# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for blender dataset"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, List, Literal
import torch

import torch
import imageio
import numpy as np
import open3d as o3d
from PIL import Image
import yaml
import json
import rasterio.features
from shapely.geometry import Polygon

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType, CAMERA_MODEL_TO_TYPE
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
from nerfstudio.data.dataparsers.blender_dataparser import Blender

from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600


@dataclass
class FreeGaussianDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: FreeGaussian)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: Optional[str] = "white"
    """alpha color of background, when set to None, InputDataset that consumes DataparserOutputs will not attempt 
    to blend with alpha_colors using image's alpha channel data. Thus rgba image will be directly used in training. """
    ply_path: Optional[Path] = None
    """Path to PLY file to load 3D points from, defined relative to the dataset directory. This is helpful for
    Gaussian splatting and generally unused otherwise. If `None`, points are initialized randomly."""


@dataclass
class FreeGaussian(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: FreeGaussianDataParserConfig
    includes_time: bool = True

    def __init__(self, config: FreeGaussianDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        if self.alpha_color is not None:
            self.alpha_color_tensor = get_color(self.alpha_color)
        else:
            self.alpha_color_tensor = None

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        poses = []
        times = []
        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            times.append(frame["time"])
        poses = np.array(poses).astype(np.float32)
        times = torch.from_numpy(np.array(times).astype(np.float32))

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            times=times,
        )

        metadata = {}
        if self.config.ply_path is not None:
            metadata.update(self._load_3D_points(self.config.data / self.config.ply_path))

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=metadata,
        )

        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path):
        pcd = o3d.io.read_point_cloud(str(ply_file_path))

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32) * self.config.scale_factor)
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out


"""mask loading functions for conerf datasets"""


def load_conerf_annotation(
    dir_path: Path, fids, height: int, width: int, scale: float, num_attributes: int, cls_to_id_mapping: Dict[str, int]
):
    """
    * load annotation from json file, and create a annotation mask with shape (h, w, num + 1)
    ret:
    mask_labels (H, W, num + 1), where num is the number of attributes
    valids (num + 1), the valids of mask
    """
    atrb_masks, mask_valids = [], []
    for fid in fids:
        mask_labels = np.zeros((height, width, num_attributes + 1), dtype=np.bool_)
        path = dir_path / f"{fid}.json"
        if not path.exists() or num_attributes == 0:
            valids = np.zeros(1, dtype=np.bool_)
            mask_labels[..., -1] = True
        else:
            with open(path) as f:
                data = json.load(f)

            for datum in data["shapes"]:
                polygon = Polygon(np.array(datum["points"]) / scale)
                cls_id = cls_to_id_mapping[datum["label"]]
                mask = rasterio.features.rasterize([polygon], out_shape=(height, width))
                mask_labels[..., cls_id] = mask_labels[..., cls_id] | mask

            # * add background class
            mask_labels[mask_labels.sum(axis=-1) == 0, -1] = True
            valids = np.ones(1, dtype=np.bool_)

        atrb_masks.append(mask_labels)
        mask_valids.append(valids)
    return atrb_masks, mask_valids


def load_coco_annotations(path, fids, height: int, width: int, scale: float = 1.0, num_attributes: int = 1):
    """
    * load annotation from json file, and create a annotation mask with shape (h, w, num + 1)
    ret: mask_labels (H, W, num + 1), where num is the number of attributes
    valids (num + 1), the valids of mask
    """
    # load annotation json
    with open(path) as f:
        data = json.load(f)

    # image id to fid
    image_id_to_fid = {}
    for img_dict in data["images"]:
        fid = img_dict["file_name"].split("_")[0]
        image_id_to_fid[img_dict["id"]] = fid

    # annotations
    fid_to_annotations = {}
    for annotations in data["annotations"]:
        image_id = annotations["image_id"]
        fid = image_id_to_fid[image_id]
        if fid not in fid_to_annotations:
            fid_to_annotations[fid] = []
        fid_to_annotations[fid].append(
            {"category_id": annotations["category_id"], "points": np.array(annotations["segmentation"]).reshape(-1, 2)}
        )

    # load annotation Polygon
    atrb_masks, mask_valids = [], []
    for fid in fids:
        mask_labels = np.zeros((height, width, num_attributes + 1), dtype=np.bool_)

        if fid not in fid_to_annotations or num_attributes == 0:
            # * w/o annotation should return a mask with all 0
            valids = np.zeros(1, dtype=np.bool_)
        else:
            for annotations in fid_to_annotations[fid]:
                polygon = Polygon(annotations["points"] / scale)
                cls_id = annotations["category_id"]
                mask = rasterio.features.rasterize([polygon], out_shape=(height, width))
                mask_labels[..., cls_id] = mask_labels[..., cls_id] | mask

            # * add background class
            mask_labels[mask_labels.sum(axis=-1) == 0, -1] = True
            valids = np.ones(1, dtype=np.bool_)

        atrb_masks.append(mask_labels)
        mask_valids.append(valids)

    return atrb_masks, mask_valids


def load_blender_annotations(path: Path, fids, height: int, width: int, num_attributes: int = 1):
    """
    * load annotation from json file, and create a annotation mask with shape (h, w, num + 1)
    ret: mask_labels (H, W, num + 1), where num is the number of attributes
    valids (num + 1), the valids of mask
    """
    atrb_masks, mask_valids = [], []
    for fid in fids:
        mask_labels = np.zeros((height, width, num_attributes + 1), dtype=np.bool_)
        seg_path = path / f"{fid}_segmentation.npy"
        if not seg_path.exists() or num_attributes == 0:
            # * w/o annotation should return a mask with all 0
            valids = np.zeros(1, dtype=np.bool_)
        else:
            mask_labels[..., :num_attributes] = np.load(seg_path)[..., :num_attributes]
            # * add background class
            mask_labels[mask_labels.sum(axis=-1) == 0, -1] = 1
            valids = np.ones(1, dtype=np.bool_)

        atrb_masks.append(mask_labels)
        mask_valids.append(valids)

    return atrb_masks, mask_valids


def load_conerf_values(path: Path, fids, num_attributes, norm_vals=True):
    with open(path, "r") as f:
        annotations_in_file = yaml.safe_load(f)
    fid_to_id_mapping = {int(fid): i for i, fid in enumerate(fids)}
    atrb_vals = np.zeros((len(fids), num_attributes), dtype=np.float32)
    atrb_val_masks = np.zeros((len(fids), num_attributes + 1), dtype=np.float32)
    atrb_val_masks[..., -1] = True

    for entry in annotations_in_file:
        fid, cls = entry["frame"], entry["class"]
        if fid in fid_to_id_mapping:
            atrb_vals[fid_to_id_mapping[fid]][cls] = entry["value"]
            atrb_val_masks[fid_to_id_mapping[fid]][cls] = True
    # if norm_vals:
    #     atrb_vals = (atrb_vals - atrb_vals.min(axis=0, keepdims=True)) / (atrb_vals.max(axis=0, keepdims=True) - atrb_vals.min(axis=0, keepdims=True))
    atrb_vals = 0.5 * (atrb_vals + 1)
    atrb_vals = np.hstack([np.zeros((atrb_vals.shape[0], 1)), atrb_vals])

    return atrb_vals, atrb_val_masks


@dataclass
class FreeGaussianCoNeRFDataParserConfig(NerfstudioDataParserConfig):
    """LiveScene dataset config"""

    _target: Type = field(default_factory=lambda: FreeGaussianCoNeRFData)
    """target class to instantiate"""
    use_bbox: bool = False
    """whether use bbox for conerf"""

    """whether normalize the attribute values to [0, 1]"""
    offset: List[float] = field(default_factory=lambda: [0, 0, 0])
    """offset of the bbox"""
    random_mask_ratio: float = 0.0
    """ratio of random mask"""
    interval: int = 2
    """interval between flow frames"""
    load_flow: bool = False
    """whether load flow"""
    load_mask: bool = False
    """whether load mask"""
    dmode: Literal["conerf", "coco", "blender"] = "conerf"
    """The dataset mode to use for loading the dataset."""
    customized_split: bool = False
    """Whether to use customized split"""
    includes_time: bool = True
    """whether includes time in cameras"""


@dataclass
class FreeGaussianCoNeRFData(Nerfstudio):
    """LiveScene DatasetParser"""

    config: FreeGaussianCoNeRFDataParserConfig
    downscale_factor: Optional[int] = None
    includes_time: bool = True

    def __init__(self, config: FreeGaussianCoNeRFDataParserConfig):
        super().__init__(config=config)
        self.get_prev_fn = lambda idx: max(idx - self.config.interval, 0)
        self.includes_time = self.config.includes_time

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        data_dir = self.config.data
        meta = load_from_json(data_dir / "dataset.json")  # "dataset_highlight.json"
        all_frames = meta["ids"]

        image_filenames, mask_filenames, depth_filenames, poses, times = [], [], [], [], []
        fx, fy, cx, cy, height, width, distort = [], [], [], [], [], [], []
        flow_filenames = []

        # * sort the frames by fname
        frames = sorted(all_frames)
        for i, frame in enumerate(frames):
            intrinsics = self._read_intrinsics(Path(frame + ".json"), data_dir)
            width.append(intrinsics[0])
            height.append(intrinsics[1])
            fx.append(intrinsics[2])
            fy.append(intrinsics[3])
            cx.append(intrinsics[4])
            cy.append(intrinsics[5])
            distort.append(
                camera_utils.get_distortion_params(
                    k1=intrinsics[6][0],
                    k2=intrinsics[6][1],
                    k3=intrinsics[6][2],
                    k4=0.0,
                    p1=intrinsics[7][0],
                    p2=intrinsics[7][1],
                )
            )
            img_name = self._get_fname(Path(frame + ".png"), data_dir)
            image_filenames.append(img_name)
            depth_name = self._get_fname(Path(frame + ".npy"), data_dir, downsample_folder_suffix="_depth")
            depth_filenames.append(depth_name)
            poses.append(self._read_pose(Path(frame + ".json"), data_dir))
            # times.append((1 / len(frames)) * i)

            flow_filenames.append(data_dir / f"flow_n{self.config.interval}" / f"{frame}.npy")

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        if self.config.customized_split and f"{split}_ids" in meta:
            i_train = np.array([i for i, ids in enumerate(frames) if ids in meta["train_ids"]])
            i_eval = np.array([i for i, ids in enumerate(frames) if ids in meta["val_ids"]])
            indices = [i for i, ids in enumerate(frames) if ids in meta[f"{split}_ids"]]

            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames + [0], self.config.eval_interval)
                i_train, i_eval = i_train - 1, i_eval - 1  # *
                i_eval = i_eval[1:]
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")
            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        # * auto alignment
        # poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        #     poses,
        #     method=orientation_method,
        #     center_method=self.config.center_method,
        # )
        transform_matrix = torch.eye(4)[:3]

        # scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:  # * False
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor
        prev_ids = torch.tensor([self.get_prev_fn(i) for i in range(poses.shape[0])], dtype=torch.long)
        poses0 = poses[prev_ids].clone()

        # HACK:use the scale_factor with auto_scale_poses
        self.pcd_scale_factor = scale_factor

        # choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        poses0 = poses0[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        if self.config.use_bbox:
            aabb = (
                torch.tensor(self.scene_data["bbox"]) - torch.tensor(self.scene_data["center"]).unsqueeze(0)
            ) * self.scene_data[
                "scale"
            ]  # * aabb_scale # (2, 3)
            # * convert to openGL
            aabb = aabb[:, np.array([0, 2, 1])]
            aabb[:, 2] *= -1
            scene_box = SceneBox(aabb=aabb)
        else:
            scene_box = SceneBox(
                aabb=torch.tensor(
                    [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
                )
            )
            scene_box.aabb += scene_box.aabb[1, :] * torch.Tensor(self.config.offset)[None, :]

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        max_fid = max(int(frame) for frame in frames)
        fids = [frames[i] for i in indices]
        times = torch.tensor([int(fid) / max_fid for fid in fids], dtype=torch.float32)

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            # times=times if self.includes_time else None,
            metadata={
                "cameras0": Cameras(
                    camera_to_worlds=poses0[:, :3, :4],
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    distortion_params=distortion_params,
                    camera_type=CameraType.PERSPECTIVE,
                    # times=times.unsqueeze(-1) if self.includes_time else None,
                )
            },
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)
        cameras.metadata["cameras0"].rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        atrb_masks, mask_valids = self._read_attributes(height[0], width[0], fids, data_dir)

        if self.config.load_mask:
            atrb_masks, mask_valids = self._read_attributes(height, width, fids, self.config.data)
            dataparser_outputs.metadata["atrb_masks"] = atrb_masks
            dataparser_outputs.metadata["mask_valids"] = mask_valids

        if self.config.load_flow:
            dataparser_outputs.metadata["flow_filenames"] = flow_filenames

        if self.includes_time:
            dataparser_outputs.cameras.times = dataparser_outputs.cameras._init_get_times(times)

        return dataparser_outputs

    def _read_attributes(self, height, width, fids, data_dir: Path):
        def load_from_yaml(file: Path):
            with open(file, "r") as f:
                return yaml.safe_load(f)

        id2cls = load_from_yaml(data_dir / "mapping.yml")
        cls2id = {v: k for k, v in id2cls.items()}
        self.num_atrbs = len(cls2id)

        if self.config.dmode == "conerf":
            atrb_masks, mask_valids = load_conerf_annotation(
                data_dir / "annotations",
                fids,
                height,
                width,
                scale=1.0 / self.downscale_factor,
                num_attributes=self.num_atrbs,
                cls_to_id_mapping=cls2id,
            )
        elif self.config.dmode == "coco":
            atrb_masks, mask_valids = load_coco_annotations(
                data_dir / "annotations.coco.json",
                fids,
                height,
                width,
                scale=1.0 / self.downscale_factor,
                num_attributes=self.num_atrbs,
            )
        elif self.config.dmode == "blender":
            atrb_masks, mask_valids = load_blender_annotations(
                data_dir / f"rgb/{int(self.downscale_factor)}x", fids, height, width, num_attributes=self.num_atrbs
            )

        atrb_masks = torch.from_numpy(np.stack(atrb_masks)).bool()  # (N_images, h, w, num_atrb+1)
        mask_valids = torch.from_numpy(np.stack(mask_valids)).bool()  # (N_images, 1)
        mask_valids = mask_valids.expand(-1, self.num_atrbs + 1)  # (N_images, num_atrb+1)
        return atrb_masks, mask_valids

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_suffix="") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_suffix: suffix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / "rgb/1x" / filepath.name)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"rgb/{2**df}x" / filepath.name).exists():
                        break
                    df += 1
                self.downscale_factor = 2 ** (df - 1)
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor >= 1:
            return (
                data_dir / "rgb" / f"{self.downscale_factor}x" / (filepath.stem + downsample_folder_suffix + filepath.suffix)
            )

        return data_dir / filepath

    def _read_intrinsics(self, filepath: Path, data_dir: Path):
        meta = load_from_json(data_dir / "camera" / filepath.name)
        w, h = int(meta["image_size"][0]), int(meta["image_size"][1])
        fx, fy = float(meta["focal_length"]), float(meta["focal_length"]) * float(meta["pixel_aspect_ratio"])
        cx, cy = float(meta["principal_point"][0]), float(meta["principal_point"][1]) * float(meta["pixel_aspect_ratio"])
        radial_dists, tangential_dists = np.array(meta["radial_distortion"]), np.array(meta["tangential_distortion"])
        return w, h, fx, fy, cx, cy, radial_dists, tangential_dists

    def _read_pose(self, filepath: Path, data_dir: Path):
        meta = load_from_json(data_dir / "camera" / filepath.name)
        data = load_from_json(data_dir / "scene.json")
        self.scene_data = data
        # * conerf invert the orientation
        R, t = np.linalg.inv(np.array(meta["orientation"])), np.array(meta["position"])
        t = (t - np.array(data["center"])) * data["scale"]
        c2w = np.concatenate([np.concatenate([R, t.reshape(3, 1)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

        # * convert to openGL
        c2w[:3, 1:3] *= -1
        c2w = c2w[np.array([0, 2, 1, 3])]
        c2w[2, :] *= -1
        return c2w

    def _load_3D_points(self, ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        pcd = o3d.io.read_point_cloud(str(ply_file_path))

        # if no points found don't read in an initial point cloud
        if len(pcd.points) == 0:
            return None

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        # TODO: read_point_cloud implementation
        points3D = (points3D - np.array(self.scene_data["center"])) * self.scene_data["scale"]
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor  # self.pcd_scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out


@dataclass
class FreeGaussianRealDataParserConfig(NerfstudioDataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: FreeGaussianRealData)
    """target class to instantiate"""
    offset: List[float] = field(default_factory=lambda: [0, 0, 0])
    """offset of the bbox"""

    """whether normalize the attribute values to [0, 1]"""
    offset: List[float] = field(default_factory=lambda: [0, 0, 0])
    """offset of the bbox"""
    random_mask_ratio: float = 0.0
    """ratio of random mask"""
    interval: int = 2
    """interval between flow frames"""
    load_flow: bool = False
    """whether load flow"""
    load_mask: bool = False
    """whether load mask"""
    includes_time: bool = True
    """whether includes time in cameras"""


@dataclass
class FreeGaussianRealData(Nerfstudio):
    """LiveScene DatasetParser for PolyCam"""

    config: FreeGaussianRealDataParserConfig
    downscale_factor: Optional[int] = None
    includes_time: bool = True

    def __init__(self, config: FreeGaussianSyntheticDataParserConfig):
        super().__init__(config=config)
        self.get_prev_fn = lambda idx: max(idx - self.config.interval, 0)
        self.includes_time = self.config.includes_time

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []
        flow_filenames = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    torch.tensor(frame["distortion_params"], dtype=torch.float32)
                    if "distortion_params" in frame
                    else camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

            # NOTE: load flow filenames
            flow_filenames.append(data_dir / f"flow_n{self.config.interval}" / f"{fname.stem}.npy")

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor
        prev_ids = torch.tensor([self.get_prev_fn(i) for i in range(poses.shape[0])], dtype=torch.long)
        poses0 = poses[prev_ids].clone()

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        poses0 = poses0[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        scene_box.aabb += scene_box.aabb[1, :] * torch.Tensor(self.config.offset)[None, :]

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = (
                torch.tensor(meta["distortion_params"], dtype=torch.float32)
                if "distortion_params" in meta
                else camera_utils.get_distortion_params(
                    k1=float(meta["k1"]) if "k1" in meta else 0.0,
                    k2=float(meta["k2"]) if "k2" in meta else 0.0,
                    k3=float(meta["k3"]) if "k3" in meta else 0.0,
                    k4=float(meta["k4"]) if "k4" in meta else 0.0,
                    p1=float(meta["p1"]) if "p1" in meta else 0.0,
                    p2=float(meta["p2"]) if "p2" in meta else 0.0,
                )
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        # Only add fisheye crop radius parameter if the images are actually fisheye, to allow the same config to be used
        # for both fisheye and non-fisheye datasets.
        metadata = {}
        if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (fisheye_crop_radius is not None):
            metadata["fisheye_crop_radius"] = fisheye_crop_radius

        max_fid = max([int(Path(frame["file_path"]).stem.split("_")[-1]) for frame in meta["frames"]])
        fids = [Path(name).stem.split("_")[-1] for name in image_filenames]
        times = torch.tensor([int(fid) / max_fid for fid in fids], dtype=torch.float32)
        # __import__('ipdb').set_trace()

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            # times=times.unsqueeze(-1) if self.includes_time else None,
            metadata={
                "cameras0": Cameras(
                    camera_to_worlds=poses0[:, :3, :4],
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    camera_type=CameraType.PERSPECTIVE,
                    # times=times.unsqueeze(-1) if self.includes_time else None,
                )
            },
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)
        cameras.metadata["cameras0"].rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        # The naming is somewhat confusing, but:
        # - transform_matrix contains the transformation to dataparser output coordinates from saved coordinates.
        # - dataparser_transform_matrix contains the transformation to dataparser output coordinates from original data coordinates.
        # - applied_transform contains the transformation to saved coordinates from original data coordinates.
        applied_transform = None
        colmap_path = self.config.data / "colmap/sparse/0"
        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
        elif colmap_path.exists():
            # For converting from colmap, this was the effective value of applied_transform that was being
            # used before we added the applied_transform field to the output dataformat.
            meta["applied_transform"] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]]
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)

        if applied_transform is not None:
            dataparser_transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        else:
            dataparser_transform_matrix = transform_matrix

        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        # reinitialize metadata for dataparser_outputs
        metadata = {}

        # _generate_dataparser_outputs might be called more than once so we check if we already loaded the point cloud
        try:
            self.prompted_user
        except AttributeError:
            self.prompted_user = False

        # Load 3D points
        if self.config.load_3D_points:
            if "ply_file_path" in meta:
                ply_file_path = data_dir / meta["ply_file_path"]

            elif colmap_path.exists():
                from rich.prompt import Confirm

                # check if user wants to make a point cloud from colmap points
                if not self.prompted_user:
                    self.create_pc = Confirm.ask(
                        "load_3D_points is true, but the dataset was processed with an outdated ns-process-data that didn't convert colmap points to .ply! Update the colmap dataset automatically?"
                    )

                if self.create_pc:
                    import json

                    from nerfstudio.process_data.colmap_utils import create_ply_from_colmap

                    with open(self.config.data / "transforms.json") as f:
                        transforms = json.load(f)

                    # Update dataset if missing the applied_transform field.
                    if "applied_transform" not in transforms:
                        transforms["applied_transform"] = meta["applied_transform"]

                    ply_filename = "sparse_pc.ply"
                    create_ply_from_colmap(
                        filename=ply_filename,
                        recon_dir=colmap_path,
                        output_dir=self.config.data,
                        applied_transform=applied_transform,
                    )
                    ply_file_path = data_dir / ply_filename
                    transforms["ply_file_path"] = ply_filename

                    # This was the applied_transform value

                    with open(self.config.data / "transforms.json", "w", encoding="utf-8") as f:
                        json.dump(transforms, f, indent=4)
                else:
                    ply_file_path = None
            else:
                if not self.prompted_user:
                    CONSOLE.print(
                        "[bold yellow]Warning: load_3D_points set to true but no point cloud found. splatfacto will use random point cloud initialization."
                    )
                ply_file_path = None

            if ply_file_path:
                sparse_points = self._load_3D_points(ply_file_path, transform_matrix, scale_factor)
                if sparse_points is not None:
                    metadata.update(sparse_points)
            self.prompted_user = True

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=dataparser_transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "mask_color": self.config.mask_color,
                **metadata,
            },
        )

        if self.config.load_mask:
            atrb_masks, mask_valids = self._read_attributes(fids, self.config.data)
            dataparser_outputs.metadata["atrb_masks"] = atrb_masks
            dataparser_outputs.metadata["mask_valids"] = mask_valids

        if self.config.load_flow:
            dataparser_outputs.metadata["flow_filenames"] = flow_filenames

        if self.includes_time:
            dataparser_outputs.cameras.times = dataparser_outputs.cameras._init_get_times(times)

        return dataparser_outputs

    def _read_attributes(self, fids, data_dir: Path):
        def load_from_yaml(file: Path):
            with open(file, "r") as f:
                return yaml.safe_load(f)

        id2cls = load_from_yaml(data_dir / "mapping.yaml")
        cls2id = {v: k for k, v in id2cls.items()}
        self.num_atrbs = len(cls2id)

        atrb_masks_list = [data_dir / "masks" / f"{fid}.npy" for fid in fids]
        atrb_masks, mask_valids = [], []
        if list(filter(lambda x: x.exists(), atrb_masks_list)):
            atrb_masks = [np.load(atrb_mask_path) for atrb_mask_path in atrb_masks_list]
            # __import__('ipdb').set_trace()
            H, W, _ = atrb_masks[0].shape
            mask_valids = (np.stack(atrb_masks).sum(axis=(1, 2)) == 0) | (
                np.stack(atrb_masks).sum(axis=(1, 2)) > H * W / 300
            )  # (N_images, num_atrb+1)

        atrb_masks = torch.from_numpy(np.stack(atrb_masks)).bool() # (N_images, num_atrb+1)
        mask_valids = torch.from_numpy(mask_valids).bool()  # (N_images, num_atrb+1)

        return atrb_masks, mask_valids


@dataclass
class FreeGaussianSyntheticDataParserConfig(NerfstudioDataParserConfig):
    """LiveScene Synthetic dataset parser config"""

    _target: Type = field(default_factory=lambda: FreeGaussianSynthetic)
    alpha_color: Optional[str] = "white"
    """alpha color of background, when set to None, InputDataset that consumes DataparserOutputs will not attempt 
    to blend with alpha_colors using image's alpha channel data. Thus rgba image will be directly used in training. """

    """whether normalize the attribute values to [0, 1]"""
    offset: List[float] = field(default_factory=lambda: [0, 0, 0])
    """offset of the bbox"""
    random_mask_ratio: float = 0.0
    """ratio of random mask"""
    interval: int = 2
    """interval between flow frames"""
    load_flow: bool = False
    """whether load flow"""
    load_mask: bool = False
    """whether load mask"""
    includes_time: bool = True
    """whether includes time in cameras"""


@dataclass
class FreeGaussianSynthetic(Blender):
    """LiveScene DatasetParser for Omnigibson Behavior"""

    config: FreeGaussianSyntheticDataParserConfig
    includes_time: bool = True

    def __init__(self, config: FreeGaussianSyntheticDataParserConfig):
        super().__init__(config=config)
        self.get_prev_fn = lambda idx: max(idx - self.config.interval, 0)
        self.includes_time = self.config.includes_time

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.config.data / "transforms.json")

        image_filenames = []
        depth_filenames = []
        poses = []
        flow_filenames = []

        for frame in meta["frames"]:
            image_name = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            depth_name = self.data / Path(frame["file_path"].replace("./images", "depth") + ".npy")
            flow_name = self.data / Path(
                frame["file_path"].replace("./images", f"interflow_n{self.config.interval}") + ".npy"
            )
            image_filenames.append(image_name)
            depth_filenames.append(depth_name)
            flow_filenames.append(flow_name)
            poses.append(np.array(frame["transform_matrix"]))

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )
        poses[:, :3, 3] *= self.scale_factor
        prev_ids = torch.tensor([self.get_prev_fn(i) for i in range(poses.shape[0])], dtype=torch.long)
        poses0 = poses[prev_ids].clone()

        # find train and eval indices based on the eval_mode specified
        if self.config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
        elif self.config.eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif self.config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
        elif self.config.eval_mode == "all":
            CONSOLE.log(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices]
        flow_filenames = [flow_filenames[i] for i in indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]
        poses0 = poses0[idx_tensor]

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        scene_box.aabb += scene_box.aabb[1, :] * torch.Tensor(self.config.offset)[None, :]

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        cx = image_width / 2.0
        cy = image_height / 2.0

        max_fid = max([int(Path(frame["file_path"]).stem.split("_")[-1]) for frame in meta["frames"]])
        fids = [Path(name).stem.split("_")[-1] for name in image_filenames]
        times = torch.tensor([int(fid) / max_fid for fid in fids], dtype=torch.float32)

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            # times=times.unsqueeze(-1) if self.includes_time else None,
            metadata={
                "cameras0": Cameras(
                    camera_to_worlds=poses0[:, :3, :4],
                    fx=focal_length,
                    fy=focal_length,
                    cx=cx,
                    cy=cy,
                    camera_type=CameraType.PERSPECTIVE,
                    # times=times.unsqueeze(-1) if self.includes_time else None,
                )
            },
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                # NOTE: depth image is not used in any training stage, only preprocess/epipolar_flow.py loads.
                "depth_filenames": depth_filenames,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )

        if self.config.load_mask:
            atrb_masks, mask_valids = self._read_attributes(fids, self.data)
            dataparser_outputs.metadata["atrb_masks"] = atrb_masks
            dataparser_outputs.metadata["mask_valids"] = mask_valids

        if self.config.load_flow:
            dataparser_outputs.metadata["flow_filenames"] = flow_filenames

        if self.includes_time:
            dataparser_outputs.cameras.times = dataparser_outputs.cameras._init_get_times(times)

        return dataparser_outputs

    def _read_attributes(self, fids, data_dir: Path):
        atrb_masks_list = [data_dir / "mask" / f"{fid}.npy" for fid in fids]
        atrb_masks = [np.load(atrb_mask_path) for atrb_mask_path in atrb_masks_list]
        H, W, _ = atrb_masks[0].shape
        mask_valids = (np.stack(atrb_masks).sum(axis=(1, 2)) == 0) | (
            np.stack(atrb_masks).sum(axis=(1, 2)) > H * W / 300
        )  # (N_images, num_atrb+1)

        atrb_masks = torch.from_numpy(np.stack(atrb_masks)).bool()  # (N_images, num_atrb+1)
        mask_valids = torch.from_numpy(mask_valids).bool()  # (N_images, num_atrb+1)
        return atrb_masks, mask_valids
