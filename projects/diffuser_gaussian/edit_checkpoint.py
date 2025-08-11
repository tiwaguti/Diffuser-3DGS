import os
import argparse
import sys
import torch
from typing import List, Union
from pathlib import Path
from torch.optim import Optimizer
from torch import nn


sys.path.append("../../")
from pointrix.point_cloud.utils import unwarp_name
from pointrix.utils.config import load_config

# from pointrix.engine.default_trainer import DefaultTrainer
from trainer import DiffuserGaussianTrainer

import diffgs


def override_point_cloud(point_cloud, composition):
    diff_mask = torch.cat(
        [
            torch.zeros(composition[0]["point_num"], 1),
            torch.ones(composition[1]["point_num"], 1),
        ]
    ).to("cuda")
    point_cloud.diffuser_density = point_cloud.opacity * diff_mask


def edit_object(trainer) -> None:
    """
    Prune the point cloud.

    Parameters
    ----------
    step : int
        The current step.
    """

    # edit object
    eps = 0.1
    for opt in trainer.optimizer.optimizer_dict.values():
        valid_points_mask = opt.point_cloud.position[..., 0] >= -1.0
        valid_points_mask &= opt.point_cloud.position[..., 0] <= 1.0
        valid_points_mask &= opt.point_cloud.position[..., 1] >= -eps
        valid_points_mask &= opt.point_cloud.position[..., 1] <= eps
        valid_points_mask &= opt.point_cloud.position[..., 2] >= -1.0
        valid_points_mask &= opt.point_cloud.position[..., 2] <= 1.0

        opt.point_cloud.remove_points(valid_points_mask, opt.optimizer)
        opt.prune_postprocess(valid_points_mask)

    point_cloud = trainer.model.point_cloud
    point_num = point_cloud.position.shape[0]
    point_cloud.diffuse_density = point_cloud.diff_strength_inverse_activation(
        torch.ones([point_num, 1]) * 1e-5
    )


def edit_diffuser(trainer) -> None:
    """
    Prune the point cloud.

    Parameters
    ----------
    step : int
        The current step.
    """

    # edit diffuser
    eps = 0.1
    for opt in trainer.optimizer.optimizer_dict.values():
        valid_points_mask = opt.point_cloud.position[..., 0] >= -0.75
        valid_points_mask &= opt.point_cloud.position[..., 0] <= 0.75
        valid_points_mask &= opt.point_cloud.position[..., 1] >= -0.3 - eps
        valid_points_mask &= opt.point_cloud.position[..., 1] <= -0.3 + eps
        valid_points_mask &= opt.point_cloud.position[..., 2] >= -0.75
        valid_points_mask &= opt.point_cloud.position[..., 2] <= 0.75

        opt.point_cloud.remove_points(valid_points_mask, opt.optimizer)
        opt.prune_postprocess(valid_points_mask)

    point_cloud = trainer.model.point_cloud
    point_num = point_cloud.position.shape[0]
    point_cloud.diff_density = point_cloud.opacity


def main(args, extras) -> None:
    cfg = load_config(args.config, cli_args=["use_timestamp=false"])
    trainer = DiffuserGaussianTrainer(cfg.trainer, cfg.exp_dir)
    trainer.load_model(args.base_model)

    # edit_object(trainer)
    edit_diffuser(trainer)

    # save model
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    trainer.composition = [
        {
            "name": "/".join(Path(args.base_model).parts[-2:]),
            "point_num": trainer.model.point_cloud.position.shape[0],
        }
    ]
    print(trainer.composition)
    trainer.save_model(args.out_model)
    trainer.model.point_cloud.save_ply(args.out_model.replace("pth", "ply"))

    # trainer = DiffuserGaussianTrainer(cfg.trainer, cfg.exp_dir)
    # trainer.load_model(args.out_model)
    # print(trainer.composition)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default=None)
    parser.add_argument("--base_model")
    parser.add_argument("--out_model")
    args, extras = parser.parse_known_args()

    main(args, extras)
