"""
FreeGaussian DataManager, containing the custom FreeGaussian dataset as well
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from copy import deepcopy
import cv2
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Literal, Optional, Tuple, Type, Union, ForwardRef, cast, get_args, get_origin
from nerfstudio.utils.misc import get_orig_class
from rich.progress import track
from typing_extensions import assert_never
from functools import cached_property

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datasets.base_dataset import InputDataset


class FreeGaussianDataset(InputDataset):
    """Dataset that returns images and flows. If no flows are found, then we generate them with Zoe flow.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0) -> None:
        super().__init__(dataparser_outputs, scale_factor)
        if "flow_filenames" in self.metadata:
            self.flow_filenames = self.metadata["flow_filenames"]
        if "cameras0" in self.cameras.metadata:
            self.cameras.metadata["cameras0"].rescale_output_resolution(scaling_factor=scale_factor)

    def get_metadata(self, data: Dict) -> Dict:
        metadata = super().get_metadata(data)
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
        if "flows" in self.metadata:
            filepath = self.flow_filenames[data["image_idx"]]
            flows = get_flow_image_from_path(filepath=filepath, height=height, width=width, scale_factor=1.0)
            metadata["flows"] = flows
        if "atrb_masks" in self.metadata and "mask_valids" in self.metadata:
            metadata["atrb_masks"] = self.metadata["atrb_masks"][data["image_idx"]]
            metadata["mask_valids"] = self.metadata["mask_valids"][data["image_idx"]]
        return metadata


@dataclass
class FreeGaussianImageDatamanagerConfig(FullImageDatamanagerConfig):
    """Configuration for the FreeGaussianImageDatamanager"""

    _target: Type = field(default_factory=lambda: FreeGaussianImageDatamanager[FreeGaussianDataset])


class FreeGaussianImageDatamanager(FullImageDatamanager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    def __init__(
        self,
        config: FreeGaussianImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        undistorted_images: List[Dict[str, torch.Tensor]] = []

        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)

        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = dataset.cameras[idx].reshape(())
            assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
                f'The size of image ({data["image"].shape[1]}, {data["image"].shape[0]}) loaded '
                f"does not match the camera parameters ({camera.width.item(), camera.height.item()})"
            )
            if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
                return data
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()
            K, image, mask, flows = _undistort_image_flow(camera, distortion_params, data, image, K)

            data["image"] = torch.from_numpy(image)
            if flows is not None:
                data["flows"] = torch.from_numpy(flows)

            if mask is not None:
                data["mask"] = mask

            dataset.cameras.fx[idx] = float(K[0, 0])
            dataset.cameras.fy[idx] = float(K[1, 1])
            dataset.cameras.cx[idx] = float(K[0, 2])
            dataset.cameras.cy[idx] = float(K[1, 2])
            dataset.cameras.width[idx] = image.shape[1]
            dataset.cameras.height[idx] = image.shape[0]

            # NOTE: we assume all the camera are with the same distortion params
            if "cameras0" in dataset.cameras.metadata:
                dataset.cameras.metadata["cameras0"].fx[idx] = float(K[0, 0])
                dataset.cameras.metadata["cameras0"].fy[idx] = float(K[1, 1])
                dataset.cameras.metadata["cameras0"].cx[idx] = float(K[0, 2])
                dataset.cameras.metadata["cameras0"].cy[idx] = float(K[1, 2])
                dataset.cameras.metadata["cameras0"].width[idx] = image.shape[1]
                dataset.cameras.metadata["cameras0"].height[idx] = image.shape[0]
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )

        # Move to device.
        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
                if "depth" in cache:
                    cache["depth"] = cache["depth"].to(self.device)
                if "flows" in cache:
                    cache["flows"] = cache["flows"].to(self.device)
                self.train_cameras = self.train_dataset.cameras.to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
                self.train_cameras = self.train_dataset.cameras
        else:
            assert_never(cache_images_device)

        return undistorted_images

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = [d.copy() for d in self.cached_eval]
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            _cam = _cameras[i : i + 1]
            # _cam.flows = data[i].pop("flows").to(self.device)
            cameras.append(_cam)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[FullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is FullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) in [FullImageDatamanager, FreeGaussianImageDatamanager]:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is FullImageDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default


def get_flow_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes flow images.
    Filepath points to a 16-bit or 32-bit flow image, or a numpy array `*.npy`.

    Args:
        filepath: Path to flow image.
        height: Target flow image height.
        width: Target flow image width.
        scale_factor: Factor by which to scale flow image.
        interpolation: flow value interpolation for resizing.

    Returns:
        flow image torch tensor with shape [height, width, 2].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        raise ValueError(f"Unsupported flow image format: {filepath.suffix}")
    return torch.from_numpy(image)


def _undistort_image_flow(
    camera: Cameras, distortion_params: np.ndarray, data: dict, image: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor], Optional[np.ndarray]]:
    mask = None
    flows = None
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        assert distortion_params[3] == 0, (
            "We doesn't support the 4th Brown parameter for image undistortion, " "Only k1, k2, k3, p1, p2 can be non-zero."
        )
        # because OpenCV expects the order of distortion parameters to be (k1, k2, p1, p2, k3), we need to reorder them
        # see https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        distortion_params = np.array(
            [
                distortion_params[0],
                distortion_params[1],
                distortion_params[4],
                distortion_params[5],
                distortion_params[2],
                distortion_params[3],
                0,
                0,
            ]
        )
        # because OpenCV expects the pixel coord to be top-left, we need to shift the principal point by 0.5
        # see https://github.com/nerfstudio-project/nerfstudio/issues/3048
        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5
        if np.any(distortion_params):
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
            image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
        else:
            newK = K
            roi = 0, 0, image.shape[1], image.shape[0]
        # crop the image and update the intrinsics accordingly
        x, y, w, h = roi
        image = image[y : y + h, x : x + w]
        # update the principal point based on our cropped region of interest (ROI)
        newK[0, 2] -= x
        newK[1, 2] -= y
        if "depth_image" in data:
            data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            if np.any(distortion_params):
                mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
            mask = mask[y : y + h, x : x + w]
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        if "flows" in data:
            # NOTE: Optical flow undistortion and cropping
            flows = data["flows"]
            H, W = flows.shape[:2]

            # Generate grid of pixel coordinates
            y_grid, x_grid = np.mgrid[0:H, 0:W]
            points = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)

            # Get the distorted end points of the flow (p1 = p0 + flow)
            points_with_flow = points + flows.reshape(-1, 2)

            # Undistort the source points (p0) and flow end points (p1)
            points_undistorted = cv2.undistortPoints(np.expand_dims(points, axis=1), K, distortion_params, P=newK).reshape(
                -1, 2
            )
            points_with_flow_undistorted = cv2.undistortPoints(
                np.expand_dims(points_with_flow, axis=1), K, distortion_params, P=newK
            ).reshape(-1, 2)

            # Calculate the undistorted flow
            flow_undistorted = points_with_flow_undistorted - points_undistorted
            flow_undistorted = flow_undistorted.reshape(H, W, 2)

            # Crop the undistorted flow in the same way as the image
            flow_undistorted = flow_undistorted[y : y + h, x : x + w]
            data["flows"] = flow_undistorted

        newK[0, 2] = newK[0, 2] + 0.5
        newK[1, 2] = newK[1, 2] + 0.5
        K = newK
    else:
        raise NotImplementedError("🧏🧏: Only perspective cameras are supported")

    return K, image, mask, flows
