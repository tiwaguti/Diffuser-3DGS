from pointrix.optimizer.optimizer import OPTIMIZER_REGISTRY
from pointrix.optimizer.gs_optimizer import GaussianSplattingOptimizer
from pointrix.utils.config import C
from dataclasses import dataclass
import torch


@OPTIMIZER_REGISTRY.register()
class DiffuserGaussianSplattingOptimizer(GaussianSplattingOptimizer):
    @dataclass
    class Config:
        # Densification
        control_module: str = "point_cloud"
        percent_dense: float = 0.01
        split_num: int = 2
        densify_stop_iter: int = 15000
        densify_start_iter: int = 500
        prune_interval: int = 100
        duplicate_interval: int = 100
        opacity_reset_interval: int = 3000
        densify_grad_threshold: float = 0.0002
        min_opacity: float = 0.005
        min_diffuser_density: float = 0.005

    cfg: Config

    def update_hypers(self) -> None:
        """
        Update the hyperparameters of the optimizer.
        """
        super().update_hypers()
        self.min_diffuser_density = C(self.cfg.min_diffuser_density, 0, self.step)

    def prune(self, step: int) -> None:
        """
        Prune the point cloud.

        Parameters
        ----------
        step : int
            The current step.
        """
        # TODO: fix me
        size_threshold = 20 if step > self.opacity_reset_interval else None
        cameras_extent = self.cameras_extent

        prune_filter = (
            (self.point_cloud.get_opacity < self.min_opacity)
            & (self.point_cloud.get_diff_density < self.min_diffuser_density)
        ).squeeze()
        if size_threshold:
            big_points_vs = self.max_radii2D > size_threshold
            big_points_ws = (
                self.point_cloud.get_scaling.max(dim=1).values > 0.1 * cameras_extent
            )
            prune_filter = torch.logical_or(prune_filter, big_points_vs)
            prune_filter = torch.logical_or(prune_filter, big_points_ws)

        valid_points_mask = ~prune_filter
        self.point_cloud.remove_points(valid_points_mask, self.optimizer)
        self.prune_postprocess(valid_points_mask)
