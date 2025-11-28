from __future__ import annotations

from typing import Dict, Literal, Optional, Type
from pathlib import Path
from matplotlib.cbook import contiguous_regions
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
import argparse
from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
import imageio
import matplotlib.pyplot as plt
from numpy.linalg import inv

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.utils.poses import inverse, multiply, to4x4
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import Blender

from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
)

MAX_AUTO_RESOLUTION = 1600


@dataclass
class FreeGaussianSyntheticDataParserConfig(NerfstudioDataParserConfig):
    """LiveScene Synthetic dataset parser config"""

    _target: Type = field(default_factory=lambda: FreeGaussianSynthetic)
    alpha_color: Optional[str] = "white"
    """alpha color of background, when set to None, InputDataset that consumes DataparserOutputs will not attempt 
    to blend with alpha_colors using image's alpha channel data. Thus rgba image will be directly used in training. """
    load_flow: bool = False


@dataclass
class FreeGaussianSynthetic(Blender):
    """LiveScene DatasetParser for Omnigibson Behavior"""

    def __init__(self, config: FreeGaussianSyntheticDataParserConfig):
        super().__init__(config=config)

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.config.data / "transforms.json")

        image_filenames = []
        depth_filenames = []
        flow_filenames = []
        poses = []

        for frame in meta["frames"]:
            image_name = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            depth_name = self.data / Path(frame["file_path"].replace("./images", "depth") + ".npy")
            flow_name = self.data / Path(frame["file_path"].replace("./images", "flow") + ".npy")

            image_filenames.append(image_name)
            depth_filenames.append(depth_name)
            flow_filenames.append(flow_name)
            poses.append(np.array(frame["transform_matrix"]))

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        # poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        #     poses,
        #     method=self.config.orientation_method,
        #     center_method=self.config.center_method,
        # )
        poses[:, :3, 3] *= self.scale_factor
        # __import__('ipdb').set_trace()

        i_train, i_eval = get_train_eval_split_all(image_filenames)

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        cx = image_width / 2.0
        cy = image_height / 2.0

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            # dataparser_transform=transform_matrix,
            metadata={
                # NOTE: depth image is not used in any training stage, only preprocess/epipolar_flow.py loads.
                "depth_filenames": depth_filenames,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        if self.config.load_flow:
            dataparser_outputs.metadata["flow_filenames"] = flow_filenames

        return dataparser_outputs


class CustomDataset(DepthDataset):
    """Dataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, internval: int = 1) -> None:
        super().__init__(dataparser_outputs, scale_factor)
        self.internval = internval
        self.get_prev_fn = lambda idx: max(idx - internval, 0)

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        # NOTE: loading the image is not necessary
        data = {
            "image_idx": image_idx,
            # "image1": image,
            "image0_path": self._dataparser_outputs.image_filenames[self.get_prev_fn(image_idx)],
            "image1_path": self._dataparser_outputs.image_filenames[image_idx],
        }
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        if self.depth_filenames is None:
            return {"depth_image": self.depths[data["image_idx"]]}

        filepath0 = self.depth_filenames[self.get_prev_fn(data["image_idx"])]
        filepath1 = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image0 = get_depth_image_from_path(filepath=filepath0, height=height, width=width, scale_factor=scale_factor)
        depth_image1 = get_depth_image_from_path(filepath=filepath1, height=height, width=width, scale_factor=scale_factor)

        meta = {
            "depth_image0": depth_image0,
            "depth_image0_path": filepath0,
            "depth_image1": depth_image1,
            "depth_image1_path": filepath1,
            # new metadata camera
            "camera0": self.cameras[self.get_prev_fn(data["image_idx"])],
            "camera1": self.cameras[data["image_idx"]],
        }

        if "flow_filenames" in self._dataparser_outputs.metadata:
            flowpath0 = self._dataparser_outputs.metadata["flow_filenames"][self.get_prev_fn(data["image_idx"])]
            flow_image0 = np.load(flowpath0)
            meta["flow_image0"] = flow_image0
        return meta


# def opencv2gl(c2w: torch.Tensor, keep_original_world_coordinate=False) -> torch.Tensor:
#     # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL): nerfstudio-doc/nerfstudio/process_data/colmap_utils.py #445
#     c2w[0:3, 1:3] *= -1
#     if not keep_original_world_coordinate:
#         c2w = c2w[np.array([0, 2, 1, 3]), :]
#         c2w[2, :] *= -1
#     return c2w


def opengl2cv(c2w_: torch.Tensor, keep_original_world_coordinate=False) -> torch.Tensor:
    c2w = c2w_.clone()
    if c2w_.shape[0] == 3:
        c2w = to4x4(c2w)

    if not keep_original_world_coordinate:
        c2w[2, :] *= -1
        c2w = c2w[np.array([0, 2, 1, 3]), :]
    c2w[0:3, 1:3] *= -1

    if c2w_.shape[0] == 3:
        c2w = c2w[:3]
    return c2w


# Function to convert Euler angles (XYZ) to rotation matrix
def diff_2d_epipolar_flow(Z: torch.Tensor, camera0: Cameras, camera1: Cameras, opticalflow: torch.Tensor):
    """caculate the interaction opticalflow with differential epipolar constraint,
       and backproject the flow to the 3D space.

    Args:
        Z: the depth map of the current frame
        camera: the camera to use for the projection
            metadata["omega"]: the angular velocity of the camera
            metadata["veloc"]: the velocity of the camera
            metadata["opticalflows"]: the normalized optical flow of the current frame
    """
    c2w0, c2w1 = camera0.camera_to_worlds, camera1.camera_to_worlds
    c2w0 = opengl2cv(camera0.camera_to_worlds, keep_original_world_coordinate=False)
    c2w1 = opengl2cv(camera1.camera_to_worlds, keep_original_world_coordinate=False)

    R1 = c2w0[:3, :3].to(torch.float64)
    R2 = c2w1[:3, :3].to(torch.float64)
    R_relative = np.dot(np.linalg.inv(R1), R2)
    relative_angles = R.from_matrix(R_relative).as_euler("xyz", degrees=False)
    omega = torch.tensor(relative_angles)

    dT = multiply(c2w1, inverse(c2w0))
    # omega = torch.tensor([dT[0, 1], dT[0, 2], dT[1, 2]]).to(torch.float64)

    # dT = multiply(inverse(c2w0), c2w1)
    # omega = R.from_matrix(dT[:3, :3]).as_euler("zyx", degrees=False)
    # omega = R.from_matrix(dT[:3, :3]).as_rotvec()

    # euler0 = R.from_matrix(c2w0[:3, :3]).as_euler("zyx", degrees=False)
    # euler1 = R.from_matrix(c2w1[:3, :3]).as_euler("zyx", degrees=False)
    # omega = euler1 - euler0

    # rotvec0 = R.from_matrix(c2w0[:3, :3]).as_rotvec()
    # rotvec1 = R.from_matrix(c2w1[:3, :3]).as_rotvec()
    # omega = rotvec1 - rotvec0

    # veloc = dT[:3, 3].to(torch.float64)
    veloc = (c2w1[:3, 3] - c2w0[:3, 3]).to(torch.float64)

    y, x = camera0.get_image_coords(pixel_offset=0).unbind(dim=-1)
    fx, fy, cx, cy = camera0.fx, camera0.fy, camera0.cx, camera0.cy
    A = rearrange(
        torch.stack(
            [
                torch.ones_like(x) * fx,
                torch.zeros_like(x),
                cx - x,
                torch.zeros_like(y),
                torch.ones_like(y) * fy,
                cy - y,
            ],
            dim=-1,
        ),
        "h w (m n) -> h w m n",
        m=2,
        n=3,
    )
    B = rearrange(
        torch.stack(
            [
                -(x - cx) * (y - cy) / fy,
                fx + (x - cx) ** 2 / fx,
                -(y - cy) * fx / fy,
                -fy - (y - cy) ** 2 / fy,
                (x - cx) * (y - cy) / fx,
                (x - cx) * fy / fx,
            ],
            dim=-1,
        ),
        "h w (m n) -> h w m n",
        m=2,
        n=3,
    )

    print(f"\n** {[dT[0, 1], dT[0, 2], dT[1, 2]]}, \n dt: {dT}")
    print(f"\n** veloc: {veloc}, omega: {omega}")
    sceneflow = A @ veloc / Z + B @ omega
    sceneflow = sceneflow.numpy()

    interflow = opticalflow + sceneflow

    # TODO: add interflow filter
    m_inf = Z.isinf().squeeze(-1)
    interflow[m_inf] = 0.0
    sceneflow[m_inf] = 0.0
    return {
        "sceneflow": sceneflow,
        "interflow": interflow,
    }


argparser = argparse.ArgumentParser(description="Inference a model")
argparser.add_argument(
    "--cfg",
    type=str,
    default="../pretrained/gma_plus-p_8x2_120k_flyingthings3d_400x720.py",
    help="The config file of the model",
)
argparser.add_argument(
    "--ckpt",
    type=str,
    default="../pretrained/gma_plus-p_8x2_120k_flyingthings3d_400x720.pth",
    help="The checkpoint file of the model",
)
argparser.add_argument("--int", type=int, default=1, help="The interval of the image files")
argparser.add_argument("--save", action="store_true", help="Visualize the flow")
argparser.add_argument("--data", type=str, default="/DATA/LiveScene/Sim/seq001_Rs_int", help="The path of the dataset")
args = argparser.parse_args()


def cropflow(flow):
    return flow
    h, w = flow.shape[:2]
    return flow[h // 8 : -h // 8, w // 8 : -w // 8]


if __name__ == "__main__":
    interflow_path, sceneflow_path, opticalflow_path, plt_path = (
        Path(args.data) / f"interflow_n{args.int}",
        Path(args.data) / f"sceneflow_n{args.int}",
        Path(args.data) / f"opticflow_n{args.int}",
        Path(args.data) / f"plt_n{args.int}",
    )
    interflow_path.mkdir(parents=True, exist_ok=True)
    sceneflow_path.mkdir(parents=True, exist_ok=True)
    opticalflow_path.mkdir(parents=True, exist_ok=True)
    plt_path.mkdir(parents=True, exist_ok=True)

    dataparser = FreeGaussianSynthetic(
        config=FreeGaussianSyntheticDataParserConfig(data=Path(args.data), eval_mode="all", depth_unit_scale_factor=1.0)
    )
    dataset = CustomDataset(dataparser_outputs=dataparser.get_dataparser_outputs(), scale_factor=1.0, internval=args.int)
    # __import__("ipdb").set_trace()

    model = init_model(args.cfg, args.ckpt, device="cuda:0")
    for i, data in tqdm(enumerate(dataset)):
        if "flow_image0" in data:
            opticalflow = data["flow_image0"]
        else:
            opticalflow = inference_model(model, data["image0_path"], data["image1_path"])
        diffdata = diff_2d_epipolar_flow(data["depth_image0"], data["camera0"], data["camera1"], opticalflow)
        sceneflow, interflow = diffdata["sceneflow"], diffdata["interflow"]

        print(f"image0_path: {data['image0_path']}, image1_path: {data['image1_path']}")
        print(
            f"opticalflow: mean={opticalflow.mean()}, std={opticalflow.std()}, max={opticalflow.max()}, min={opticalflow.min()}"
        )
        print(f"sceneflow: mean={sceneflow.mean()}, std={sceneflow.std()}, max={sceneflow.max()}, min={sceneflow.min()}")
        print(f"interflow: mean={interflow.mean()}, std={interflow.std()}, max={interflow.max()}, min={interflow.min()}")

        if args.save:
            visualize_flow(cropflow(interflow), f"{interflow_path}/{data['image0_path'].stem}.png")
            np.save(f"{interflow_path}/{data['image1_path'].stem}.npy", interflow)

            visualize_flow(cropflow(sceneflow), f"{sceneflow_path}/{data['image0_path'].stem}.png")
            np.save(f"{sceneflow_path}/{data['image1_path'].stem}.npy", sceneflow)

            visualize_flow(cropflow(opticalflow), f"{opticalflow_path}/{data['image0_path'].stem}.png")
            np.save(f"{opticalflow_path}/{data['image1_path'].stem}.npy", opticalflow)

        y, x = data["camera0"].get_image_coords(pixel_offset=0)[::64, ::64].long().unbind(dim=-1)

        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        axs[0][0].imshow(np.linalg.norm(cropflow(opticalflow), axis=-1))
        axs[0][0].set_title("opticalflow")
        axs[0][1].imshow(np.linalg.norm(cropflow(sceneflow), axis=-1))
        axs[0][1].set_title("sceneflow")
        axs[0][2].imshow(np.linalg.norm(cropflow(interflow), axis=-1))
        axs[0][2].set_title("interflow")
        axs[1][0].quiver(
            x, y, opticalflow[::64, ::64, 0], opticalflow[::64, ::64, 1], scale=1.0, scale_units="xy", color="b"
        )
        axs[1][1].quiver(x, y, sceneflow[::64, ::64, 0], sceneflow[::64, ::64, 1], scale=1.0, scale_units="xy", color="r")
        axs[1][2].quiver(x, y, interflow[::64, ::64, 0], interflow[::64, ::64, 1], scale=1.0, scale_units="xy", color="g")


        # load visulization flows and plot in figs
        interflow_vis = imageio.v2.imread(f"{interflow_path}/{data['image0_path'].stem}.png")
        sceneflow_vis = imageio.v2.imread(f"{sceneflow_path}/{data['image0_path'].stem}.png")
        axs[2][0].imshow(opticalflow_vis)
        axs[2][0].set_title("opticalflow_vis")
        axs[2][1].imshow(sceneflow_vis)
        axs[2][1].set_title("sceneflow_vis")
        axs[2][2].imshow(interflow_vis)
        axs[2][2].set_title("interflow_vis")

        plt.savefig(plt_path / f"{data['image1_path'].stem}.png")
        plt.close()
