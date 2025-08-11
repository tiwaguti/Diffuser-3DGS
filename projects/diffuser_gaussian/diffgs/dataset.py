from pointrix.dataset.base_data import (
    DATA_FORMAT_REGISTRY,
    BaseDataPipeline,
    BaseReFormatData,
)
from pointrix.dataset.nerf_data import NerfReFormat
from pointrix.dataset.colmap_data import ColmapReFormat
from pointrix.dataset.base_data import BaseDataFormat, BaseImageDataset
from pathlib import Path
import json
import numpy as np
import os
from PIL import Image
from typing import Any, Dict, List, Optional
from pathlib import Path
from pointrix.camera.camera import Camera
from pointrix.utils.dataset.dataset_utils import fov2focal, focal2fov
from dataclasses import dataclass
from pointrix.utils.config import parse_structured
from pointrix.logger.writer import ProgressLogger, Logger
from diffuser_detector.model import DetectorUNet
import torch
import cv2
from glob import glob


def load_diffuser_masks(self) -> List[Image.Image]:
    """
    The function for loading cached images typically requires user customization.

    Parameters
    ----------
    split: The split of the data.
    """

    def convert_path(img_path):
        path = "diffuser".join(img_path.rsplit("images", 1))
        path = path.replace(path.split(".")[-1], "png")
        return path

    diff_filenames = [convert_path(f) for f in self.data_list.image_filenames]
    if not os.path.exists(diff_filenames[0]):
        print("Diffuser not available.")

    diff_mask_lists, diff_depth_lists = [], []
    cached_progress = ProgressLogger(
        description="Loading cached diffusers", suffix="images/s"
    )
    cached_progress.add_task(
        "cache_diffuser_read",
        "Loading cached masks",
        len(diff_filenames),
    )
    cached_progress.start()
    for img_filename, diff_filename in zip(
        self.data_list.image_filenames, diff_filenames
    ):
        if os.path.exists(diff_filename):
            temp_image = cv2.imread(diff_filename, -1)
            vmax = {np.dtype("uint8"): 255.0, np.dtype("uint16"): 65535.0}.get(
                temp_image.dtype, 1.0
            )
            h, w = temp_image.shape[:2]
            if temp_image.ndim == 2:
                temp_image = np.dstack([temp_image] * 3)
            new_size = (int(w * self.scale), int(h * self.scale))
            resize_image = cv2.resize(temp_image, new_size) / vmax
            diff_mask = resize_image[..., 0:1] > 0.5
            diff_depth = resize_image[..., 1:2] / np.where(
                diff_mask, resize_image[..., 2:3], 1.0
            )
            diff_mask_lists.append(diff_mask)
            diff_depth_lists.append(diff_depth)

            # temp_image = Image.open(mask_filename)
            # w, h = temp_image.size
            # resize_image = temp_image.resize((int(w * self.scale), int(h * self.scale)))
            # diff_mask_lists.append(np.array(resize_image, dtype="uint8")[..., 0])
        else:
            temp_image = Image.open(img_filename)
            w, h = temp_image.size
            new_size = (int(w * self.scale), int(h * self.scale), 1)
            diff_mask_lists.append(np.zeros(new_size, dtype=np.float32))
            diff_depth_lists.append(np.zeros(new_size, dtype=np.float32))
        cached_progress.update("cache_diffuser_read", step=1)
    cached_progress.stop()
    return diff_mask_lists, diff_depth_lists


def predict_diffuser_masks(self, device="cuda") -> List[Image.Image]:
    """
    The function for loading cached images typically requires user customization.

    Parameters
    ----------
    split: The split of the data.
    """

    detector_checkpoint = torch.load("outputs/detector_train/chkpnt_00350.pt")
    detector_model = DetectorUNet()
    detector_model.load_state_dict(detector_checkpoint["model"])
    detector_model.to(device)

    mask_lists = []
    cached_progress = ProgressLogger(
        description="Loading cached masks", suffix="images/s"
    )
    cached_progress.add_task(
        "cache_diffuser_read",
        "Predicting diffuser masks",
        len(self.data_list),
    )
    cached_progress.start()
    for img_np in self.data_list.images:
        img = (
            torch.from_numpy(img_np[..., :3] / 255.0)
            .permute(2, 0, 1)
            .float()
            .to(device)
        )
        mask = detector_model(img.unsqueeze(0)).squeeze(0)[0]
        mask_lists.append(
            np.clip(mask.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        )

        cached_progress.update("cache_diffuser_read", step=1)
    cached_progress.stop()
    return mask_lists


@DATA_FORMAT_REGISTRY.register()
class DiffuserColmapReFormat(ColmapReFormat):
    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        cached_image: bool = True,
        scale: float = 1.0,
    ):
        super().__init__(data_root, split, cached_image, scale)
        self.data_list.diffuser_masks, self.data_list.diffuser_depths = (
            load_diffuser_masks(self)
        )


@DATA_FORMAT_REGISTRY.register()
class DiffuserNerfReFormat(NerfReFormat):
    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        scale: float = 1.0,
        predict_diffuser: bool = False,
        cached_image: bool = True,
    ):
        BaseReFormatData.__init__(self, data_root, split, cached_image)
        if predict_diffuser:
            self.data_list.diffuser_masks = self.predict_diffuser_masks()
        else:
            self.data_list.diffuser_masks, self.data_list.diffuser_depths = (
                load_diffuser_masks(self)
            )

    def load_camera(self, split: str) -> List[Camera]:
        """
        The function for loading the camera typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        if split == "train":
            with open(
                os.path.join(self.data_root, "transforms_train.json")
            ) as json_file:
                json_file = json.load(json_file)
        elif split == "val":
            with open(
                os.path.join(self.data_root, "transforms_test.json")
            ) as json_file:
                json_file = json.load(json_file)

        fovx = json_file["camera_angle_x"]

        frames = json_file["frames"]
        cameras = []
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(self.data_root, frame["file_path"] + ".png")
            cam_name = glob(os.path.join(self.data_root, frame["file_path"]) + "*")[0]

            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # R is stored transposed due to 'glm' in CUDA code
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image_path = os.path.join(self.data_root, cam_name)

            image = np.array(Image.open(image_path))

            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[0])
            FovY = fovy
            FovX = fovx
            camera = Camera(
                idx=idx,
                R=R,
                T=T,
                width=image.shape[1],
                height=image.shape[0],
                rgb_file_name=image_path,
                fovX=FovX,
                fovY=FovY,
                bg=1.0,
            )
            cameras.append(camera)

        return cameras


class DiffuserDataFormat(BaseDataFormat):
    diffuser_masks: Optional[List[Image.Image]] = None
    diffuser_depths: Optional[List[Image.Image]] = None


class DiffuserImageDataset(BaseImageDataset):
    def __init__(self, format_data: DiffuserDataFormat) -> None:
        super().__init__(format_data)
        self.diffuser_masks = format_data.diffuser_masks
        self.diffuser_depths = format_data.diffuser_depths
        if self.diffuser_masks is not None:
            # Transform cached images.
            self.diffuser_masks = [
                self.to_torch(image) for image in self.diffuser_masks
            ]
            self.diffuser_depths = [
                self.to_torch(image) for image in self.diffuser_depths
            ]

    def to_torch(self, array):
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, idx):
        image_file_name = self.image_file_names[idx]
        camera = self.camera_list[idx]
        image = (
            self._load_transform_image(image_file_name)
            if self.images is None
            else self.images[idx]
        )
        diffuser_mask = self.diffuser_masks[idx]
        diffuser_depth = self.diffuser_depths[idx]
        camera.height = image.shape[1]
        camera.width = image.shape[2]
        return {
            "image": image,
            "diffuser_mask": diffuser_mask,
            "diffuser_depth": diffuser_depth,
            "camera": camera,
            "FovX": camera.fovX,
            "FovY": camera.fovY,
            "height": camera.image_height,
            "width": camera.image_width,
            "world_view_transform": camera.world_view_transform,
            "full_proj_transform": camera.full_proj_transform,
            "extrinsic_matrix": camera.extrinsic_matrix,
            "intrinsic_matrix": camera.intrinsic_matrix,
            "camera_center": camera.camera_center,
        }


class DiffuserDataPipeline(BaseDataPipeline):
    @dataclass
    class Config:
        # Datatype
        data_path: str = "data"
        data_type: str = "nerf_synthetic"
        cached_image: bool = True
        shuffle: bool = True
        batch_size: int = 1
        num_workers: int = 1
        white_bg: bool = False
        scale: float = 1.0
        predict_diffuser: bool = False
        use_dataloader: bool = True

    cfg: Config

    def __init__(self, cfg: Config, dataformat) -> None:
        self.cfg = parse_structured(self.Config, cfg)
        self._fully_initialized = True

        self.train_format_data = dataformat(
            data_root=self.cfg.data_path,
            split="train",
            cached_image=self.cfg.cached_image,
            scale=self.cfg.scale,
            predict_diffuser=self.cfg.predict_diffuser,
        ).data_list
        self.validation_format_data = dataformat(
            data_root=self.cfg.data_path,
            split="val",
            cached_image=self.cfg.cached_image,
            scale=self.cfg.scale,
            predict_diffuser=self.cfg.predict_diffuser,
        ).data_list

        self.point_cloud = self.train_format_data.PointCloud
        self.white_bg = self.cfg.white_bg
        self.use_dataloader = self.cfg.use_dataloader

        # assert not self.use_dataloader and self.cfg.batch_size == 1 and \
        #     self.cfg.cached_image, "Currently only support batch_size=1, cached_image=True when use_dataloader=False"
        self.loaddata()

        self.training_cameras = self.training_dataset.cameras

    def get_training_dataset(self) -> DiffuserImageDataset:
        """
        Return training dataset
        """
        # TODO: use registry
        self.training_dataset = DiffuserImageDataset(format_data=self.train_format_data)

    def get_validation_dataset(self) -> DiffuserImageDataset:
        """
        Return validation dataset
        """
        self.validation_dataset = DiffuserImageDataset(
            format_data=self.validation_format_data
        )
