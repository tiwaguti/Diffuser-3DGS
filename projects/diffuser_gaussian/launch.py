import os
import argparse
import sys
import torch

sys.path.append("../../")
from pointrix.utils.config import load_config

# from pointrix.engine.default_trainer import DefaultTrainer
from trainer import DiffuserGaussianTrainer

import diffgs


def main(args, extras) -> None:

    cfg = load_config(args.config, cli_args=extras)
    gaussian_trainer = DiffuserGaussianTrainer(
        cfg.trainer,
        cfg.exp_dir,
    )

    if False:
        point_cloud = gaussian_trainer.model.point_cloud
        point_cloud.load_ply("outputs/merged.ply")
        num_points = len(point_cloud)

        optimizer = gaussian_trainer.optimizer.optimizer_dict["optimizer_1"]
        optimizer.max_radii2D = torch.zeros(num_points).to("cuda")
        optimizer.percent_dense = 0.01
        optimizer.pos_gradient_accum = torch.zeros((num_points, 1)).to("cuda")
        optimizer.denom = torch.zeros((num_points, 1)).to("cuda")
        optimizer.opacity_deferred = False
        attr = {
            "position": point_cloud.position,
            "opacity": point_cloud.opacity,
            "scaling": point_cloud.scaling,
            "rotation": point_cloud.rotation,
            # "blur_features": point_cloud.blur_features,
            "features_rest": point_cloud.features_rest,
        }
        optimizer.state = {}
        point_cloud.extend_optimizer(attr, optimizer)

    gaussian_trainer.train_loop()
    model_path = os.path.join(
        cfg.exp_dir, "chkpnt" + str(gaussian_trainer.global_step) + ".pth"
    )
    gaussian_trainer.save_model(path=model_path)

    gaussian_trainer.test()
    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default=None)
    args, extras = parser.parse_known_args()

    main(args, extras)
