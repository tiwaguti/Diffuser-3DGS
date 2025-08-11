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
from checkpoint_utils import extend_model

import diffgs
from diffgs.visualize import test_view_render


def merge_models(trainer, paths: List[Path] = None) -> None:
    trainer.optimizer.load_state_dict(torch.load(paths[0])["optimizer"])
    trainer.composition = []
    for model_idx, path in enumerate(paths):
        data_list = torch.load(path)

        if model_idx == 0:
            extend_model(trainer, data_list["model"], append=False)
        else:
            trainer.optimizer.load_state_dict(data_list["optimizer"], append=True)
            extend_model(trainer, data_list["model"], append=True)

        trainer.renderer.load_state_dict(data_list["renderer"])
        if len(paths) == 1:
            trainer.composition = data_list["composition"]
        else:
            trainer.composition.append(
                {
                    "name": "/".join(Path(path).parts[-2:]),
                    "point_num": data_list["model"]["point_cloud.position"].shape[0],
                }
            )

    trainer.start_steps = trainer.global_step
    for optimizer in trainer.optimizer.optimizer_dict.values():
        optimizer.step = trainer.global_step


def main(args, extras) -> None:

    cfg = load_config(args.config, cli_args=["use_timestamp=false"])
    trainer = DiffuserGaussianTrainer(cfg.trainer, cfg.exp_dir)
    merge_models(trainer, [args.base_model, args.diffuser_model])

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    trainer.save_model(args.out_model)
    trainer.model.point_cloud.save_ply(args.out_model.replace("pth", "ply"))
    print(trainer.composition)

    test_view_render(
        trainer, trainer.datapipeline, output_path=os.path.dirname(args.out_model)
    )

    # trainer = DiffuserGaussianTrainer(cfg.trainer, cfg.exp_dir)
    # trainer.load_model(args.out_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default=None)
    parser.add_argument("--base_model")
    parser.add_argument("--out_model")
    parser.add_argument("--diffuser_model")
    args, extras = parser.parse_known_args()

    main(args, extras)
