from dataclasses import dataclass
from pointrix.engine.default_trainer import DefaultTrainer
from typing import List, Optional
import torch
import torch.nn.functional as F
from pointrix.renderer import parse_renderer
from pointrix.utils.config import parse_structured
from pointrix.optimizer import parse_optimizer, parse_scheduler
from pointrix.model import parse_model
from pointrix.logger import parse_writer
from pointrix.hook import parse_hooks
from pointrix.utils.renderer.renderer_utils import RenderFeatures
from pointrix.dataset.base_data import DATA_FORMAT_REGISTRY
from pathlib import Path
from diffgs.dataset import DiffuserDataPipeline
from diffgs.visualize import test_view_render, novel_view_render, train_view_render
import os
from checkpoint_utils import extend_model


class DiffuserGaussianTrainer(DefaultTrainer):
    @dataclass
    class Config(DefaultTrainer.Config):
        pretrain_checkpoint: Optional[str] = None
        blur_step_start: int = 7000
        blur_step_end: int = 15000
        diffser_opacity_threshold: float = 0.02

    cfg: Config

    def __init__(self, cfg: Config, exp_dir: Path, device: str = "cuda") -> None:
        # super().__init__(cfg=cfg, exp_dir=exp_dir, device=device)
        self.exp_dir = exp_dir
        self.device = device

        self.start_steps = 0
        self.global_step = 0
        self.blur_mult = 0.0

        self.composition = [{"name": "all", "point_num": 0}]

        # build config
        self.cfg = parse_structured(self.Config, cfg)
        # build hooks
        self.hooks = parse_hooks(self.cfg.hooks)
        self.call_hook("before_run")
        # build datapipeline
        self.datapipeline = parse_data_pipeline(self.cfg.dataset)

        # build render and point cloud model
        self.white_bg = self.datapipeline.white_bg
        self.renderer = parse_renderer(
            self.cfg.renderer, white_bg=self.white_bg, device=device
        )

        self.model = parse_model(self.cfg.model, self.datapipeline, device=device)

        # build optimizer and scheduler
        cameras_extent = self.datapipeline.training_dataset.radius
        self.schedulers = parse_scheduler(
            self.cfg.scheduler, cameras_extent if self.cfg.spatial_lr_scale else 1.0
        )
        self.optimizer = parse_optimizer(
            self.cfg.optimizer, self.model, cameras_extent=cameras_extent
        )

        # build logger and hooks
        self.logger = parse_writer(self.cfg.writer, exp_dir)

        self.model.setup_point_cloud()

    def gather_rendered_features(self, render_frames):
        rendered_features = {}
        for frame in render_frames:
            for result in frame:
                for k, f in result["rendered_features_split"].items():
                    if k in rendered_features:
                        rendered_features[k].append(f)
                    else:
                        rendered_features[k] = [f]

        for feature_name in rendered_features.keys():
            rendered_features[feature_name] = torch.stack(
                rendered_features[feature_name], dim=0
            )
        return rendered_features

    def render_iter(self, b_i, render_dict, opacity):

        # 1st pass
        # precompute shared geometry params
        # refr_dict = self.model.compute_refraction(**render_dict, **b_i)
        # render_dict.update(**refr_dict)
        # render_dict["position"] = refr_dict["refr_pos"]

        geo_dict = self.renderer.compute_geometry(**render_dict, **b_i)
        # geo_dict["direction"] = refr_dict["refr_dir"]
        blur_features = self.model.compute_blur_params(
            **geo_dict,
            **render_dict,
            **b_i,
            blur_mult=self.blur_mult,
        )

        rendered_blur = self.renderer.render_blur(
            RenderFeatures(
                rgb_blur=geo_dict["rgb"],
            ),
            opacity=opacity,
            **blur_features,
            **b_i,
            **render_dict,
            **geo_dict,
        )

        rendered_deblur = self.renderer.render_deblur(
            RenderFeatures(
                rgb_deblur=geo_dict["rgb"],
            ),
            opacity=opacity,
            **blur_features,
            **b_i,
            **render_dict,
            **geo_dict,
        )

        # gather results
        return {
            "render_frames": [
                rendered_blur,
                rendered_deblur,
            ],
            "viewspace_points": rendered_blur["viewspace_points"],
            "visibilitys": rendered_blur["visibility_filter"].unsqueeze(0),
            "radii": rendered_blur["radii"].unsqueeze(0),
        }

    def render_batch(self, batch):
        render_frames, viewspace_points, visibilitys, radii = [], [], [], []
        render_dict = self.model.forward(batch)
        opacity = self.model.point_cloud.get_opacity
        for b_i in batch:
            results = self.render_iter(b_i, render_dict, opacity)
            render_frames.append(results["render_frames"])
            viewspace_points.append(results["viewspace_points"])
            visibilitys.append(results["visibilitys"])
            radii.append(results["radii"])

        return {
            **self.gather_rendered_features(render_frames),
            "viewspace_points": viewspace_points,
            "visibility": torch.cat(visibilitys).any(dim=0),
            "radii": torch.cat(radii, 0).max(dim=0).values,
        }

    def update_blur_multiplier(self, iteration):
        i_start, i_end = self.cfg.blur_step_start, self.cfg.blur_step_end
        self.blur_mult = max(0.0, min((iteration - i_start) / (i_end - i_start), 1.0))

    def train_step(self, batch: List[dict]) -> None:
        """
        The training step for the model.

        Parameters
        ----------
        batch : dict
            The batch data.
        """
        render_results = self.render_batch(batch)
        self.loss_dict = self.model.get_loss_dict(render_results, batch, self.blur_mult)
        self.loss_dict["loss"].backward()

        self.optimizer_dict = self.model.get_optimizer_dict(
            self.loss_dict, render_results, self.white_bg
        )

    def train_loop(self) -> None:
        """
        The training loop for the model.
        """
        loop_range = range(self.start_steps, self.cfg.max_steps + 1)
        self.call_hook("before_train")

        for iteration in loop_range:
            self.call_hook("before_train_iter")

            batch = self.datapipeline.next_train(self.global_step)
            self.renderer.update_sh_degree(iteration)
            self.schedulers.step(self.global_step, self.optimizer)

            self.update_blur_multiplier(iteration)
            self.train_step(batch)

            self.optimizer.update_model(**self.optimizer_dict)
            self.call_hook("after_train_iter")
            self.global_step += 1
            if (
                iteration + 1
            ) % self.cfg.val_interval == 0 or iteration + 1 == self.cfg.max_steps:
                self.call_hook("before_val")
                self.validation()
                self.call_hook("after_val")
        self.call_hook("after_train")

    @torch.no_grad()
    def validation(self):
        self.val_dataset_size = len(self.datapipeline.validation_dataset)
        for i in range(0, self.val_dataset_size):
            self.call_hook("before_val_iter")
            batch = self.datapipeline.next_val(i)
            render_results = self.render_batch(batch)
            self.metric_dict = self.model.get_metric_dict(render_results, batch)
            self.call_hook("after_val_iter")

    def test(self, model_path=None) -> None:
        """
        The testing method for the model.
        """
        # self.model.load_ply(model_path)
        self.load_model(model_path)
        self.model.to(self.device)
        self.renderer.active_sh_degree = self.renderer.cfg.max_sh_degree
        test_view_render(
            self,
            self.datapipeline,
            output_path=self.cfg.output_path,
        )
        train_view_render(
            self,
            self.datapipeline,
            output_path=self.cfg.output_path,
        )
        # novel_view_render(
        #     self,
        #     self.datapipeline,
        #     output_path=self.cfg.output_path,
        # )

    def load_model(self, path: Path = None) -> None:
        if path is None:
            path = os.path.join(self.exp_dir, "chkpnt" + str(self.global_step) + ".pth")

        self.composition = []
        data_list = torch.load(path)

        self.optimizer.load_state_dict(data_list["optimizer"])
        extend_model(self, data_list["model"], append=False)

        self.renderer.load_state_dict(data_list["renderer"])
        self.composition = data_list["composition"]

        self.start_steps = self.global_step
        for optimizer in self.optimizer.optimizer_dict.values():
            optimizer.step = self.global_step

    def save_model(self, path: Path = None) -> None:
        if path is None:
            path = os.path.join(self.exp_dir, "chkpnt" + str(self.global_step) + ".pth")
        data_list = {
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.get_state_dict(),
            "renderer": self.renderer.state_dict(),
            "composition": self.composition,
        }
        torch.save(data_list, path)

    def sample_feature_map(self, feature_map, uv, width, height):
        # feature_map: [C, H, W]
        # uv: [N, 2]

        size = torch.Tensor([width, height]).cuda()
        uv_norm = (uv / size[None, :]) * 2 - 1
        samples = F.grid_sample(
            feature_map.unsqueeze(0),
            uv_norm.unsqueeze(0).unsqueeze(0),
            align_corners=False,
        )
        return samples[0, :, 0].T


def parse_data_pipeline(cfg: dict):
    """
    Parse the data pipeline.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    if len(cfg) == 0:
        return None
    data_type = cfg.data_type
    dataformat = DATA_FORMAT_REGISTRY.get(data_type)

    return DiffuserDataPipeline(cfg, dataformat)
