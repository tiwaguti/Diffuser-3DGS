import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import torch
import torchvision.transforms as T
import imageio
from torch import nn
from operator import methodcaller

from pointrix.utils.losses import l1_loss
from pointrix.utils.system import mkdir_p
from pointrix.utils.gaussian_points.gaussian_utils import psnr
from pointrix.utils.visuaize import (
    visualize_rgb,
    visualize_depth,
    to8b,
)
from PIL import Image


def scale_and_clamp(img, scale=1.0, offset=0.0):
    return torch.clamp(img * scale + offset, 0, 1)


def visualize_colormap(depth, minmax=None, cmap=cv2.COLORMAP_JET, tensorboard=False):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.detach().cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        if (x > 0).any():
            mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        else:
            mi = 0
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = np.clip((x - mi) / (ma - mi + 1e-8), 0, 1)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_).flip(dims=(0,))  # (3, H, W)
    if tensorboard:
        return x_
    x = to8b(x_.detach().cpu().numpy().transpose(1, 2, 0))
    return x


def visualize_normal(norm, tensorboard=False):
    norm_color = (norm + 1) * 0.5
    if tensorboard:
        return norm_color

    x = to8b(norm_color.detach().cpu().numpy().transpose(1, 2, 0))
    return x


def visualize_render_features(render_features, tensorboard=False):
    visual_dict = {}
    for k, vis in render_features.items():
        if not isinstance(vis, torch.Tensor):
            continue
        if k.startswith("rgb_"):
            visual_dict[k] = visualize_rgb(vis.squeeze(), tensorboard=tensorboard)

    for k, vis in render_features.items():
        if k.startswith("depth_"):
            visual_dict[k] = visualize_depth(vis.squeeze(), tensorboard=tensorboard)

    for k, vis in render_features.items():
        if k.startswith("alpha_"):
            visual_dict[k] = visualize_colormap(vis.squeeze(), tensorboard=tensorboard)

    visual_dict["rgb_difference"] = visualize_rgb(
        scale_and_clamp(
            (render_features["rgb_blur"] - render_features["rgb_deblur"]).squeeze(),
            4.0,
            0.5,
        ),
        tensorboard=tensorboard,
    )

    # # diffuser detection
    # visual_dict["diff_detection"] = visualize_rgb(
    #     ((render_features["diff_detection"] + 1) * 0.5).squeeze(),
    #     tensorboard=tensorboard,
    # )
    # visual_dict["diff_acc_density"] = visualize_colormap(
    #     render_features["diff_acc_density"].squeeze(),
    #     minmax=(0.0, 5.0),
    #     tensorboard=tensorboard,
    # )
    if "diff_conatct" in render_features:
        visual_dict["diff_contact"] = visualize_colormap(
            render_features["diff_contact"].squeeze(),
            minmax=(0.0, 1.0),
            tensorboard=tensorboard,
        )
    # visual_dict["blur_org"] = visualize_colormap(
    #     render_features["blur_org"].squeeze(),
    #     minmax=(0.0, 5.0),
    #     tensorboard=tensorboard,
    # )

    if "blur_size" in render_features:
        visual_dict["blur_size"] = visualize_colormap(
            render_features["blur_size"].squeeze(),
            minmax=(0.0, 0.2),
            tensorboard=tensorboard,
        )

    if "blur_scale" in render_features:
        visual_dict["blur_scale"] = visualize_colormap(
            render_features["blur_scale"].squeeze(),
            minmax=(0.0, 0.0005),
            tensorboard=tensorboard,
        )

    if "norm_diff" in render_features:
        visual_dict["norm_diff"] = visualize_normal(
            render_features["norm_diff"].squeeze(),
            tensorboard=tensorboard,
        )
    if "view_dir" in render_features:
        visual_dict["view_dir"] = visualize_normal(
            render_features["view_dir"].squeeze(),
            tensorboard=tensorboard,
        )

    if "diff_dist" in render_features:
        visual_dict["diff_dist"] = visualize_colormap(
            render_features["diff_dist"][:, 0].squeeze(),
            minmax=(0.0, 1.0),
            tensorboard=tensorboard,
        )

    # visual_dict["embed_depth_diff_75"] = visualize_colormap(
    #     render_features["embed_depth_diff"][..., 75].permute(0, 2, 1).squeeze(),
    #     tensorboard=tensorboard,
    # )
    # visual_dict["embed_depth_trans_75"] = visualize_colormap(
    #     render_features["embed_depth_trans"][..., 75].permute(0, 2, 1).squeeze(),
    #     tensorboard=tensorboard,
    # )

    # visual_dict["embed_depth_diff_255"] = visualize_colormap(
    #     render_features["embed_depth_diff"][..., 255].permute(0, 2, 1).squeeze(),
    #     tensorboard=tensorboard,
    # )
    # visual_dict["embed_depth_trans_255"] = visualize_colormap(
    #     render_features["embed_depth_trans"][..., 255].permute(0, 2, 1).squeeze(),
    #     tensorboard=tensorboard,
    # )

    if "gt_images" in render_features:
        visual_dict["ground_truth"] = visualize_rgb(
            render_features["gt_images"].squeeze(),
            tensorboard=tensorboard,
        )
    return visual_dict


@torch.no_grad()
def test_view_render(trainer, datapipeline, output_path, device="cuda"):
    """
    Render the test view and save the images to the output path.

    Parameters
    ----------
    model : BaseModel
        The point cloud model.
    renderer : Renderer
        The renderer object.
    datapipeline : DataPipeline
        The data pipeline object.
    output_path : str
        The output path to save the images.
    """
    l1_test = 0.0
    psnr_test = 0.0
    val_dataset = datapipeline.validation_dataset
    val_dataset_size = len(val_dataset)
    progress_bar = tqdm(
        range(0, val_dataset_size),
        desc="Validation progress",
        leave=False,
    )

    mkdir_p(os.path.join(output_path, "test_view"))

    for i in range(0, val_dataset_size):
        b_i = val_dataset[i]
        render_dict = trainer.model.forward(b_i)
        opacity = trainer.model.point_cloud.get_opacity
        image_name = os.path.basename(b_i["camera"].rgb_file_name)
        render_frames = trainer.render_iter(b_i, render_dict, opacity)["render_frames"]
        render_results = [x["rendered_features_split"] for x in render_frames]
        render_features = {}
        for result in render_results:
            render_features.update({k: v.unsqueeze(0) for k, v in result.items()})
        metric_dict = trainer.model.get_metric_dict(render_features, [b_i])
        vis_features = visualize_render_features(metric_dict)

        gt_image = torch.clamp(b_i["image"].to("cuda").float(), 0.0, 1.0)
        image = torch.clamp(render_features["rgb_blur"], 0.0, 1.0)

        for feat_name, visual_feat in vis_features.items():
            if not os.path.exists(os.path.join(output_path, f"test_view_{feat_name}")):
                os.makedirs(os.path.join(output_path, f"test_view_{feat_name}"))
            imageio.imwrite(
                os.path.join(output_path, f"test_view_{feat_name}", image_name),
                visual_feat,
            )

        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        progress_bar.update(1)
    progress_bar.close()
    l1_test /= val_dataset_size
    psnr_test /= val_dataset_size
    print(f"Test results: L1 {l1_test:.5f} PSNR {psnr_test:.5f}")


@torch.no_grad()
def train_view_render(trainer, datapipeline, output_path, device="cuda"):
    """
    Render the test view and save the images to the output path.

    Parameters
    ----------
    model : BaseModel
        The point cloud model.
    renderer : Renderer
        The renderer object.
    datapipeline : DataPipeline
        The data pipeline object.
    output_path : str
        The output path to save the images.
    """
    l1_test = 0.0
    psnr_test = 0.0
    train_dataset = datapipeline.training_dataset
    train_dataset_size = len(train_dataset)
    progress_bar = tqdm(
        range(0, train_dataset_size),
        desc="Validation progress",
        leave=False,
    )

    mkdir_p(os.path.join(output_path, "train_view"))

    for i in range(0, train_dataset_size):
        b_i = train_dataset[i]
        render_dict = trainer.model.forward(b_i)
        opacity = trainer.model.point_cloud.get_opacity
        image_name = os.path.basename(b_i["camera"].rgb_file_name)
        render_frames = trainer.render_iter(b_i, render_dict, opacity)["render_frames"]
        render_results = [x["rendered_features_split"] for x in render_frames]
        render_features = {}
        for result in render_results:
            render_features.update({k: v.unsqueeze(0) for k, v in result.items()})
        metric_dict = trainer.model.get_metric_dict(render_features, [b_i])
        vis_features = visualize_render_features(metric_dict)

        gt_image = torch.clamp(b_i["image"].to("cuda").float(), 0.0, 1.0)
        image = torch.clamp(render_features["rgb_blur"], 0.0, 1.0)

        for feat_name, visual_feat in vis_features.items():
            if not os.path.exists(os.path.join(output_path, f"train_view_{feat_name}")):
                os.makedirs(os.path.join(output_path, f"train_view_{feat_name}"))
            imageio.imwrite(
                os.path.join(output_path, f"train_view_{feat_name}", image_name),
                visual_feat,
            )

        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        progress_bar.update(1)
    progress_bar.close()
    l1_test /= train_dataset_size
    psnr_test /= train_dataset_size
    print(f"Test results: L1 {l1_test:.5f} PSNR {psnr_test:.5f}")


def novel_view_render(
    trainer,
    datapipeline,
    output_path,
    novel_view_list=["Dolly", "Zoom", "Spiral"],
    device="cuda",
):
    """
    Render the novel view and save the images to the output path.

    Parameters
    ----------
    model : BaseModel
        The point cloud model.
    renderer : Renderer
        The renderer object.
    datapipeline : DataPipeline
        The data pipeline object.
    output_path : str
        The output path to save the images.
    novel_view_list : list, optional
        The list of novel views to render, by default ["Dolly", "Zoom", "Spiral"]
    """
    cameras = datapipeline.training_cameras
    print("Rendering Novel view ...............")
    for novel_view in novel_view_list:
        novel_view_camera_list = cameras.generate_camera_path(50, novel_view)

        for i, camera in enumerate(novel_view_camera_list):

            b_i = {
                "camera": camera,
                "FovX": camera.fovX,
                "FovY": camera.fovY,
                "height": int(camera.image_height),
                "width": int(camera.image_width),
                "world_view_transform": camera.world_view_transform,
                "full_proj_transform": camera.full_proj_transform,
                "extrinsic_matrix": camera.extrinsic_matrix,
                "intrinsic_matrix": camera.intrinsic_matrix,
                "camera_center": camera.camera_center,
            }
            render_dict = trainer.model.forward(b_i)
            # render_dict.update(atributes_dict)
            opacity = trainer.model.point_cloud.get_opacity
            render_frames = trainer.render_iter(b_i, render_dict, opacity)[
                "render_frames"
            ]
            render_results = [x["rendered_features_split"] for x in render_frames]
            render_features = {}
            for result in render_results:
                render_features.update({k: v.unsqueeze(0) for k, v in result.items()})
            metric_dict = trainer.model.get_metric_dict(render_features, [b_i])
            vis_features = visualize_render_features(metric_dict)

            for feat_name, visual_feat in vis_features.items():
                if not os.path.exists(
                    os.path.join(output_path, f"{novel_view}_{feat_name}")
                ):
                    os.makedirs(os.path.join(output_path, f"{novel_view}_{feat_name}"))
                imageio.imwrite(
                    os.path.join(
                        output_path, f"{novel_view}_{feat_name}", "{:0>3}.png".format(i)
                    ),
                    visual_feat,
                )
