import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
import torch
import yaml

from gsplat.rendering import rasterization
from freegaussian.utils import get_viewmat

from freegaussian.utils import (
    get_viewmat,
    to_homogenous,
    from_homogenous,
)

np.random.seed(5)
torch.set_float32_matmul_precision("high")

argparser = argparse.ArgumentParser(description="Inference a model")
argparser.add_argument(
    "--cfg",
    type=str,
    default="outputs/seq001_Rs_int/freegaussian/2024-09-16_173635.037493/config.yml",
    help="The config file of the model",
)
argparser.add_argument("--crop", action="store_true", help="crop the gaussian pcd")
argparser.add_argument("--dynamic", action="store_true", help="use the deformable gaussian pcd")
args = argparser.parse_args()

if __name__ == "__main__":
    config_path = Path(args.cfg)
    config, pipeline, checkpoint_path, _ = eval_setup(config_path)
    datamanager, model = pipeline.datamanager, pipeline.model

    tsfms = yaml.safe_load(open(config_path.parent / "dataparser_transforms.json", "r"))

    if model.crop_box is not None and not model.training and args.crop:
        crop_ids = model.crop_box.within(model.means).squeeze()
        if crop_ids.sum() == 0:
            crop_ids = None
    else:
        crop_ids = None

    if crop_ids is not None:
        opacities_crop = model.opacities[crop_ids]
        means_crop = model.means[crop_ids]
        features_dc_crop = model.features_dc[crop_ids]
        features_rest_crop = model.features_rest[crop_ids]
        scales_crop = model.scales[crop_ids]
        quats_crop = model.quats[crop_ids]
    else:
        opacities_crop = model.opacities
        means_crop = model.means
        features_dc_crop = model.features_dc
        features_rest_crop = model.features_rest
        scales_crop = model.scales
        quats_crop = model.quats

    colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

    N = means_crop.shape[0]
    camera, data = datamanager.next_train(0)
    M = data["mask_valids"].shape[-1] - 1
    gaussian_masks = torch.zeros((N, M), dtype=torch.bool)
    num_train_cameras = len(datamanager.train_dataset)

    for i in range(num_train_cameras):
        camera, data = datamanager.next_train(i)
        image_idx = data["image_idx"]
        BLOCK_WIDTH = 16
        camera_scale_fac = model._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(camera.camera_to_worlds)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        model.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if model.config.sh_degree > 0:
            sh_degree_to_use = min(model.step // model.config.sh_degree_interval, model.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop)
            sh_degree_to_use = None

        if args.dynamic:
            times = camera.times.expand(means_crop.shape[0], -1)
            d_xyz, d_rotation, d_scaling = model.deform(means_crop.detach(), times)

            means = from_homogenous(torch.bmm(d_xyz, to_homogenous(means_crop).unsqueeze(-1)).squeeze(-1))
            scales = torch.exp(scales_crop) + d_scaling
            quats = quats_crop / quats_crop.norm(dim=-1, keepdim=True) + d_rotation
        else:
            means = means_crop
            scales = torch.exp(scales_crop)
            quats = quats_crop / quats_crop.norm(dim=-1, keepdim=True)

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
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="ED",
            sh_degree=model.config.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=model.config.rasterize_mode,
        )

        depth = render.squeeze() / tsfms["scale"]
        image_filename = datamanager.train_dataset.image_filenames[image_idx]
        depth_save_path = image_filename.parent.parent / "depth" / image_filename.with_suffix(".npy")
        depth_save_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(depth_save_path, depth.detach().cpu().numpy().astype(np.float32))

    num_eval_cameras = len(datamanager.eval_dataset)
    for i in range(num_eval_cameras):
        camera, data = datamanager.next_eval(i)
        image_idx = data["image_idx"]
        BLOCK_WIDTH = 16
        camera_scale_fac = model._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(camera.camera_to_worlds)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        model.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if model.config.sh_degree > 0:
            sh_degree_to_use = min(model.step // model.config.sh_degree_interval, model.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop)
            sh_degree_to_use = None

        if args.dynamic:
            times = camera.times.expand(means_crop.shape[0], -1)
            d_xyz, d_rotation, d_scaling = model.deform(means_crop.detach(), times)

            means = from_homogenous(torch.bmm(d_xyz, to_homogenous(means_crop).unsqueeze(-1)).squeeze(-1))
            scales = torch.exp(scales_crop) + d_scaling
            quats = quats_crop / quats_crop.norm(dim=-1, keepdim=True) + d_rotation
        else:
            means = means_crop
            scales = torch.exp(scales_crop)
            quats = quats_crop / quats_crop.norm(dim=-1, keepdim=True)

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
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="ED",
            sh_degree=model.config.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=model.config.rasterize_mode,
        )

        depth = render.squeeze() / tsfms["scale"]
        image_filename = datamanager.eval_dataset.image_filenames[image_idx]
        depth_save_path = image_filename.parent.parent / "depth" / image_filename.with_suffix(".npy").name
        np.save(depth_save_path, depth.detach().cpu().numpy().astype(np.float32))

    print("DONE! save depth to ", depth_save_path.parent)
