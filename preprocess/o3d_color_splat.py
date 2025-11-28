
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
import torch
import open3d as o3d

from gsplat.rendering import rasterization
from freegaussian.utils import get_viewmat

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


import os
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import torch
import yaml

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE


def eval_load_checkpoint(config: TrainerConfig, pipeline: Pipeline) -> Tuple[Path, int]:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    Returns:
        A tuple of the path to the loaded checkpoint and the step at which it was saved.
    """
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path, load_step


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        update_config_callback: Callback to update the config before loading the pipeline


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    # config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path, step = eval_load_checkpoint(config, pipeline)

    return config, pipeline, checkpoint_path, step

if __name__ == "__main__":
    config_path = Path(args.cfg)
    config, pipeline, checkpoint_path, _ = eval_setup(config_path)
    # __import__('ipdb').set_trace()
    datamanager, model = pipeline.datamanager, pipeline.model

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
        # if image_idx not in [
        #     180, 190, 200, 210, 220,
        #     550, 555, 560, 565, 570,
        # ]: continue
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
            # packed=False,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=model.config.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=model.config.rasterize_mode,
        )

        # HACK: filtering
        xy = info["means2d"].cpu().long()
        im = ((xy >= 0) & (xy < torch.tensor([H, W]))).all(-1)
        xy = xy[im]

        # masks
        mask = data["atrb_masks"][..., :-1] & data["mask_valids"][..., :-1][None, None, ...]
        m = mask[xy[:, 1], xy[:, 0]]

        # TODO: comfirm the means is in (x y)
        # __import__('ipdb').set_trace()
        ids = info["gaussian_ids"][im]
        ids = ids[..., None].expand(-1, M)[m].cpu()  # (n, M)
        gaussian_masks[ids, m.nonzero()[:, -1]] = True

    colors = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 1.0, 0.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0]])
    XYZ = means_crop[:, None, :].expand(-1, M, -1)[gaussian_masks].detach().cpu().numpy()  # (n)
    ci = torch.arange(M)[None, :].expand(N, -1)[gaussian_masks].numpy()
    ids = np.random.choice(XYZ.shape[0], 2000, replace=False)
    # xyz = XYZ[ids]
    xyz = XYZ
    cs = colors[ci][ids]
    bg_pts = means_crop[~gaussian_masks.sum(-1).bool()].detach().cpu().numpy()

    # NOTE: create a o3d pcd and save.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(cs)

    bg = o3d.geometry.PointCloud()
    bg.points = o3d.utility.Vector3dVector(bg_pts)
    bg.colors = o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.5]] * bg_pts.shape[0]))

    o3d.visualization.draw_geometries([pcd, bg])
    o3d.io.write_point_cloud("gaussian.ply", pcd)

