from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
from torch.nn.parameter import Parameter

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes

from freegaussian.freegaussian_model import FreeGaussianModelConfig, FreeGaussianModel
from .utils import from_homogenous, get_viewmat, to_homogenous

@dataclass
class FreeGaussianControlModelConfig(FreeGaussianModelConfig):
    _target: Type = field(default_factory=lambda: FreeGaussianControlModel)


class FreeGaussianControlModel(FreeGaussianModel):
    config: FreeGaussianControlModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(*args, seed_points=seed_points, **kwargs)

    def load_deformable_checkpoint(self, checkpoint_path: Path):
        loaded_state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {(key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state["pipeline"].items()}
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}
        self.load_state_dict(model_state, strict=False)

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(int(camera.width.item()), int(camera.height.item()), self.background_color)
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            gaussian_mask = self.gaussian_mask[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            gaussian_mask = self.gaussian_mask

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop)
            sh_degree_to_use = None

        control_points = means_crop[gaussian_mask.any(-1)]
        control_points_mask = gaussian_mask[gaussian_mask.any(-1)]
        if not self.training and "cameras0" not in camera.metadata:
            control_points_d_avg = self.controller.get_atrb_vals().to(self.device)
        else:
            with torch.no_grad():
                times_0 = self.init_camera.times.expand(control_points.shape[0], -1)
                d_xyz, _, _ = self.deform(control_points, times_0)
                init_means = from_homogenous(torch.bmm(d_xyz, to_homogenous(control_points).unsqueeze(-1)).squeeze(-1))

                times = camera.times.expand(control_points.shape[0], -1)
                d_xyz, _, _ = self.deform(control_points, times)
                means = from_homogenous(torch.bmm(d_xyz, to_homogenous(control_points).unsqueeze(-1)).squeeze(-1))

                # TODO: simply means the features
                control_points_d_avg = torch.stack([(means[control_points_mask[:, i]] - init_means[control_points_mask[:, i]]).mean(0) for i in range(control_points_mask.shape[1])])

        value = control_points_mask.float() @ control_points_d_avg / control_points_mask.sum(-1, keepdim=True)
        d_xyz, d_rot, d_scale = self.control(control_points, value)

        # means = from_homogenous(torch.bmm(d_xyz, to_homogenous(control_points).unsqueeze(-1)).squeeze(-1))

        d_means = torch.zeros_like(means_crop)
        d_means[gaussian_mask.any(-1)] = d_xyz
        means = means_crop + d_means

        d_scales = torch.zeros_like(scales_crop)
        d_scales[gaussian_mask.any(-1)] = d_scale
        scales = torch.exp(scales_crop) + d_scales

        d_quats = torch.zeros_like(quats_crop)
        d_quats[gaussian_mask.any(-1)] = d_rot
        quats = quats_crop / quats_crop.norm(dim=-1, keepdim=True) + d_quats

        # TODO:
        render, alpha, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        cbs = super().get_training_callbacks(training_callback_attributes)
        return [cbs[0]]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups.pop("deform")
        return param_groups
