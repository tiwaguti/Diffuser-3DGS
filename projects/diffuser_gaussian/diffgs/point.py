import numpy as np
import torch.nn as nn
import torch
from dataclasses import dataclass
from pointrix.model.gaussian_points.gaussian_points import GaussianPointCloud
from pointrix.point_cloud import PointCloud, POINTSCLOUD_REGISTRY


@POINTSCLOUD_REGISTRY.register()
class DiffuserGaussianPointCloud(GaussianPointCloud):
    @dataclass
    class Config(GaussianPointCloud.Config):
        max_diff_strength: float = 0.1

    cfg: Config

    def setup(self, point_cloud=None):
        super().setup(point_cloud)
        self.diff_density_activation = torch.sigmoid
        self.diff_density_inverse_activation = self.inverse_opacity_activation
        # self.diff_strength_activation = torch.exp
        # self.diff_strength_inverse_activation = torch.log
        self.diff_strength_activation = torch.exp
        self.diff_strength_inverse_activation = torch.log

        num_points = len(self.position)

        self.register_atribute(
            "diff_density",
            self.diff_strength_inverse_activation(torch.ones(num_points, 1) * 0.1),
        )
        self.register_atribute(
            "diff_strength",
            self.diff_strength_inverse_activation(torch.ones(num_points, 1) * 0.1),
        )

    def re_init(self, num_points):
        super().re_init(num_points)

        self.register_atribute(
            "diff_density",
            self.diff_strength_inverse_activation(torch.ones(num_points, 1) * 0.1),
        )
        self.register_atribute(
            "diff_strength",
            self.diff_strength_inverse_activation(torch.ones(num_points, 1) * 0.1),
        )

    @property
    def get_diff_density(self):
        return self.diff_density_activation(self.diff_density)

    @property
    def get_diff_strength(self):
        return self.diff_strength_activation(self.diff_strength)
