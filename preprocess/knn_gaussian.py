import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
import torch
import yaml

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

if __name__ == "__main__":
    key_frames = yaml.safe_load(open("preprocess/key_frames.yaml", "r"))
    config_path = Path(args.cfg)
    config, pipeline, checkpoint_path, _ = eval_setup(config_path)
    datamanager, model = pipeline.datamanager, pipeline.model
    current_scene_key_frame = key_frames[config.experiment_name]

    if model.crop_box is not None and not model.training and args.crop:
        crop_ids = model.crop_box.within(model.means).squeeze()
        if crop_ids.sum() == 0: crop_ids = None
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
        if image_idx not in current_scene_key_frame: continue
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
            render_mode="ED",
            sh_degree=model.config.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=model.config.rasterize_mode,
        )

        # HACK: filtering
        xy = info["means2d"].cpu().long()
        im = ((xy >= 0) & (xy < torch.tensor([W, H]))).all(-1)
        xy = xy[im]

        depth = render.squeeze()
        delta_depth = depth[xy[:, 1], xy[:, 0]] - info["depths"][im]
        dm = (-depth[xy[:, 1], xy[:, 0]] * 0.1 < delta_depth) & (delta_depth < depth[xy[:, 1], xy[:, 0]] * 1)
        # dm = torch.ones_like(dm, dtype=bool)
        xy = xy[dm.cpu()]

        # masks
        if i < num_train_cameras:
            mask = data["atrb_masks"][..., :-1] & data["mask_valids"][..., :-1][None, None, ...]
            m = mask[xy[:, 1], xy[:, 0]]  # (n m)
            ids = info["gaussian_ids"][im][dm]
            ids = ids[..., None].expand(-1, M)[m].cpu()  # (n, M)
            gaussian_masks[ids, m.nonzero()[:, -1]] = True
        # else:
        #     mask = data["atrb_masks"][..., :-1] & data["mask_valids"][..., :-1][None, None, ...]
        #     m = ~mask[yx[:, 0], yx[:, 1]]  # (n m)
        #     ids = info["gaussian_ids"][im][dm]
        #     ids = ids[..., None].expand(-1, M)[m].cpu()  # (n, M)
        #     gaussian_masks[ids, m.nonzero()[:, -1]] = False
    # clustering: https://github.com/patchy631/machine-learning/tree/main/unsupervised_learning
    # from sklearn.cluster import DBSCAN
    # from sklearn.cluster import kmeans_plusplus

    # __import__('ipdb').set_trace()
    # dbscan = DBSCAN(eps=0.5, min_samples=4)
    # dbscan.fit(XYZ)
    # ax.scatter(XYZ[:,0], XYZ[:,1], XYZ[:,2], c=dbscan.labels_)
    # centers_init, indices = kmeans_plusplus(XYZ, n_clusters=M, random_state=0)
    # plt.scatter(centers_init[:, 0], centers_init[:, 1], c="r", s=500)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    XYZ = means_crop[:, None, :].expand(-1, M, -1)[gaussian_masks].detach().cpu().numpy()  # (n)
    ci = torch.arange(M)[None, :].expand(N, -1)[gaussian_masks].numpy()
    colors = np.array(["red", "orange", "green", "blue", "cyan", "pink", "yellow", "black", "purple", "brown"])
    ids = np.random.choice(XYZ.shape[0], min(5000, XYZ.shape[0]), replace=False)
    ax.scatter(XYZ[ids, 0], XYZ[ids, 1], XYZ[ids, 2], c=colors[ci][ids])
    print(XYZ.shape)
    plt.title("DBSCAN Clustering")
    plt.show()

    save_path = config.data / "gaussian_mask_NxM_crop.npy" if args.crop else config.data / "gaussian_mask_NxM.npy"
    plt.savefig(save_path.with_suffix(".png"), dpi=256, bbox_inches="tight")

    np.save(save_path, gaussian_masks.numpy())
    print(f"DONE! save gaussian masks to {save_path}")

    # NOTE: create a o3d pcd and save.
    # import open3d as o3d
    # colors = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 1.0, 0.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0]])
    # cs = colors[ci][ids]
    # bg_pts = means_crop[~gaussian_masks.sum(-1).bool()].detach().cpu().numpy()
    #
    # # NOTE: create a o3d pcd and save.
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(XYZ)
    # pcd.colors = o3d.utility.Vector3dVector(cs)
    #
    # bg = o3d.geometry.PointCloud()
    # bg.points = o3d.utility.Vector3dVector(bg_pts)
    # bg.colors = o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.5]] * bg_pts.shape[0]))
    #
    # o3d.visualization.draw_geometries([pcd, bg])
    # o3d.io.write_point_cloud("gaussian.ply", pcd)
