import os
import argparse
import sys
import torch
import gradio as gr
import shutil
from glob import glob
from itertools import zip_longest
from pathlib import Path
import torch.nn as nn
import numpy as np

sys.path.append("../../")
import diffgs
from trainer import DiffuserGaussianTrainer
from pointrix.utils.config import load_config

# from pointrix.utils.visuaize import visualize_blur_org, visualize_blur_size

shared_data = {}


def render_view_comp(view_idx, *param_ui):
    eps = 1e-6
    trainer = shared_data["trainer"]
    trainer.blur_mult = 1.0
    diff_density_tensors, diff_strength_tensors, opacity_tensors = [], [], []

    for i, comp in enumerate(trainer.composition):
        point_num = (
            comp["point_num"]
            if len(trainer.composition) > 1
            else trainer.model.point_cloud.position.shape[0]
        )
        one = torch.ones((point_num, 1)).cuda()
        diff_density_tensors.append(param_ui[3 * i + 0] * one)
        diff_strength_tensors.append(param_ui[3 * i + 1] * one)
        opacity_tensors.append(param_ui[3 * i + 2] * one)

    diff_density = param_ui[3 * i + 0]
    diff_strength = param_ui[3 * i + 1]

    # trainer.model.point_cloud.diff_density = nn.Parameter(
    #     inverse_activation(torch.clamp(torch.cat(diff_density_tensors), eps, 1 - eps))
    # )

    pcd = trainer.model.point_cloud
    if pcd.diff_strength.requires_grad:
        pcd.diff_strength.requires_grad_(False)
    if pcd.diff_density.requires_grad:
        pcd.diff_density.requires_grad_(False)

    pcd.diff_strength[:] = pcd.diff_strength_inverse_activation(
        torch.clamp(torch.Tensor([diff_strength]), eps, 1 - eps)
    )
    pcd.diff_density[:] = pcd.diff_density_inverse_activation(
        torch.clamp(torch.Tensor([diff_density]), eps, 1 - eps)
    )
    # trainer.model.point_cloud.opacity = nn.Parameter(
    #     inverse_activation(torch.clamp(torch.cat(opacity_tensors), eps, 1 - eps))
    # )

    if view_idx < len(trainer.datapipeline.training_dataset):
        b_i = trainer.datapipeline.training_dataset[view_idx]
    else:
        b_i = trainer.datapipeline.validation_dataset[
            view_idx - len(trainer.datapipeline.training_dataset)
        ]
    render_dict = trainer.model.forward(b_i)
    opacity = trainer.model.point_cloud.get_opacity

    render_frames = trainer.render_iter(b_i, render_dict, opacity)["render_frames"]
    render_results = [x["rendered_features_split"] for x in render_frames]
    render_features = {}
    for result in render_results:
        render_features.update({k: v.unsqueeze(0) for k, v in result.items()})
    rgb_blur = render_features["rgb_blur"].squeeze(0)
    gt_img = torch.clamp(b_i["image"], 0.0, 1.0)

    with torch.no_grad():
        return [
            gt_img.permute(1, 2, 0).cpu().numpy(),
            rgb_blur.permute(1, 2, 0).cpu().numpy(),
        ]


def list_checkpoints(exp_dir):
    checkpoints = [file for file in os.listdir(exp_dir) if file.endswith("pth")]
    return gr.Dropdown(
        checkpoints,
        # value=checkpoints[0],
        type="value",
        interactive=True,
    )


def update_param_ui(exp_dd, chkpt_dd):
    print(os.path.join(exp_dd, chkpt_dd))
    shared_data["trainer"].load_model(os.path.join(exp_dd, chkpt_dd))
    param_ui = []

    # diffuser
    for i, comp in zip_longest(range(3), shared_data["trainer"].composition):
        with gr.Row(visible=comp is not None) as row:
            param_ui += [row]
            param_ui += [gr.Text(comp["name"] if comp else "", show_label=False)]
            param_ui += [
                gr.Slider(
                    minimum=0, maximum=1.0, step=0.01, label="density", interactive=True
                )
            ]
            param_ui += [
                gr.Slider(
                    value=0.0,
                    minimum=0.0,
                    maximum=1.0,
                    label="strength",
                    interactive=True,
                )
            ]
            param_ui += [
                gr.Slider(
                    value=1.0,
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    label="opacity",
                    interactive=True,
                )
            ]

    # blur
    for i, comp in zip_longest(range(3), shared_data["trainer"].composition):
        with gr.Row(visible=comp is not None) as row:
            param_ui += [row]
            param_ui += [gr.Text(comp["name"] if comp else "", show_label=False)]
            param_ui += [
                gr.Slider(
                    minimum=0, maximum=20.0, step=0.1, label="dist", interactive=True
                )
            ]
            param_ui += [
                gr.Slider(
                    value=0.0, minimum=0.0, maximum=0.01, label="size", interactive=True
                )
            ]
            param_ui += [
                gr.Slider(
                    value=1.0,
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    label="opacity",
                    interactive=True,
                )
            ]
    return param_ui


def load_views(config_file):
    def shortpath(longpath):
        return "/".join(Path(longpath).parts[-2:])

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default=None)
    args, extras = parser.parse_known_args()
    args.config = config_file
    cfg = load_config(args.config, cli_args=["use_timestamp=false"])
    trainer = DiffuserGaussianTrainer(
        cfg.trainer,
        "outputs/viewer_tmp",
    )
    shutil.rmtree("outputs/viewer_tmp")

    shared_data["view_names"] = [
        f"[train{i:04d}] {shortpath(filename)}"
        for i, filename in enumerate(
            trainer.datapipeline.training_dataset.image_file_names
        )
    ] + [
        f"[test{i:04d}] {shortpath(filename)}"
        for i, filename in enumerate(
            trainer.datapipeline.validation_dataset.image_file_names
        )
    ]
    shared_data["trainer"] = trainer

    return gr.Dropdown(
        shared_data["view_names"],
        value=shared_data["view_names"][0],
        type="index",
        interactive=True,
    )


config_files = sorted([x for x in glob("configs/*.yaml")])
exp_dirs = set(
    [os.path.dirname(x) for x in glob("outputs/*/*/*/*.pth")]
    + [os.path.dirname(x) for x in glob("outputs/**/*.pth")]
)
exp_dirs = sorted(list(exp_dirs), key=os.path.getmtime, reverse=True)


with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            config_dd = gr.Dropdown(config_files, label="Config")
            view_dd = gr.Dropdown(None, label="View")
            dir_dd = gr.Dropdown(exp_dirs, label="Exp dir")
            chkpt_dd = gr.Dropdown(None, label="Checkpoint")
            refresh_btn = gr.Button(value="Refresh")

        with gr.Tab(label="Compute blur from diffuser"):
            diff_param_ui = []
            diff_comp_params = []
            render_comp_btn = gr.Button(value="Render")
            for i in range(3):
                with gr.Row(visible=False) as row:
                    diff_param_ui += [row, gr.Text(show_label=False)]
                    diff_density_sl = gr.Slider(
                        minimum=0, maximum=1.0, step=0.01, label="density"
                    )
                    diff_strength_sl = gr.Slider(
                        value=0.0, minimum=0.0, maximum=1.0, label="strength"
                    )
                    opacity_sl = gr.Slider(
                        minimum=0, maximum=1, step=0.01, label="opacity"
                    )
                    diff_comp_params += [diff_density_sl, diff_strength_sl, opacity_sl]
                    diff_param_ui += [diff_density_sl, diff_strength_sl, opacity_sl]

        with gr.Row():
            gt_imgview = gr.Image(label="GT")
            diffuser_imgview = gr.Image(label="Diffuser")
        with gr.Row():
            blur_org_imgview = gr.Image(label="blur_org")
            blur_size_imgview = gr.Image(label="blur_size")

    config_dd.select(load_views, inputs=[config_dd], outputs=[view_dd])
    dir_dd.select(list_checkpoints, inputs=[dir_dd], outputs=[chkpt_dd])
    chkpt_dd.select(
        update_param_ui,
        inputs=[dir_dd, chkpt_dd],
        outputs=diff_param_ui,
    )

    refresh_btn.click(load_views, inputs=[config_dd], outputs=[view_dd])

    render_comp_btn.click(
        render_view_comp,
        inputs=[view_dd] + diff_comp_params,
        outputs=[gt_imgview, diffuser_imgview],
    )

demo.launch(server_name="0.0.0.0")
