from pointrix.model.base_model import BaseModel, MODEL_REGISTRY
from dataclasses import dataclass
from pointrix.point_cloud import parse_point_cloud
import torch
import torch.nn as nn
from pointrix.model.loss import l1_loss, ssim, psnr
from pointrix.utils.gaussian_points.gaussian_utils import inverse_sigmoid

from typing import Mapping, Any, List


@MODEL_REGISTRY.register()
class DiffuserGaussianSplatting(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        lambda_diff_fidelity: float = 0.1
        lambda_deblur_tv: float = 0.1
        diff_threshold: float = 0.01
        init_diff_strength: float = 0.5e-4
        diff_strength_min: float = 0.1e-4
        diff_strength_max: float = 1.0e-4
        optimize_diff_strength: bool = True
        plane_axes: str = "xz"
        obj_position: List[float] = (0, 0, 0)
        diff_size: List[List[float]] = ((-2, -2), (2, 2))
        obj_size: List[List[float]] = ((-2, -2), (2, 2))
        diff_distance: float = 0.2

    cfg: Config

    def setup(self, datapipeline, device="cuda"):
        self.point_cloud = parse_point_cloud(self.cfg.point_cloud, datapipeline).to(
            device
        )
        self.point_cloud.set_prefix_name("point_cloud")

        # define object/diffuser geometry
        self.plane_axes = self.cfg.plane_axes
        self.obj_position = torch.Tensor(self.cfg.obj_position)
        self.diff_normal = self.diffuser_vec[2]
        self.diff_position = (
            self.obj_position + self.cfg.diff_distance * self.diff_normal
        )
        self.diff_size = self.cfg.diff_size
        self.obj_size = self.cfg.obj_size

        # diffuser parameter
        diff_strength_range = self.cfg.diff_strength_max - self.cfg.diff_strength_min
        eps = 1e-9
        self.diff_strength_activation = (
            lambda x: torch.sigmoid(x) * diff_strength_range
            + self.cfg.diff_strength_min
            + eps
        )
        diff_strength_inverse_activation = lambda x: inverse_sigmoid(
            (x - self.cfg.diff_strength_min - eps) / diff_strength_range
        )

        self.diff_strength = nn.Parameter(
            diff_strength_inverse_activation(
                torch.FloatTensor([self.cfg.init_diff_strength])
            ).to("cuda"),
            requires_grad=self.cfg.optimize_diff_strength,
        )

        # refraction parameter
        eta_min = 0.5
        eta_max = 1.2
        eta_range = eta_max - eta_min
        eps = 1e-9
        self.eta_activation = lambda x: torch.sigmoid(x) * eta_range + eta_min + eps
        eta_inverse_activation = lambda x: inverse_sigmoid(
            (x - eta_min - eps) / eta_range
        )

        self.eta = nn.Parameter(
            eta_inverse_activation(torch.FloatTensor([1.0])).to("cuda"),
            requires_grad=True,
        )

        self.device = device

    @property
    def get_diff_strength(self):
        return self.diff_strength_activation(self.diff_strength)

    @property
    def get_eta(self):
        return self.eta_activation(self.eta)

    @property
    def diffuser_vec(self):
        """
        Return diffuser vector
        0/1: first/second axis, 2: normal axis
        """
        axes = {"xy": [0, 1, 2], "yz": [1, 2, 0], "xz": [0, 2, 1]}[self.plane_axes]
        eigen_vec = torch.eye(3)[axes]
        return eigen_vec

    def setup_point_cloud(self):
        pcd = self.point_cloud

        num_obj = pcd.position.shape[0]
        num_all = pcd.position.shape[0]
        eps = 1e-6

        pcd.position.requires_grad_(False)
        self.setup_object_point_cloud(pcd.position[:num_obj])
        self.setup_diffuser_point_cloud(pcd.position[num_obj:])
        pcd.position = pcd.position.to("cuda")
        pcd.position.requires_grad_(True)

        eigen_vec = self.diffuser_vec
        s = torch.Tensor(self.diff_size).max() * 0.01
        scaling = s * eigen_vec[0] + s * eigen_vec[1] + 0.1 * s * eigen_vec[2]
        pcd.scaling.requires_grad_(False)
        pcd.scaling[:] = pcd.scaling_inverse_activation(scaling).expand_as(pcd.scaling)
        pcd.scaling = pcd.scaling.to("cuda")
        pcd.scaling.requires_grad_(True)

        pcd.rotation.requires_grad_(False)
        pcd.rotation[:] = torch.Tensor([0, 0, 0, 1]).expand_as(pcd.rotation)
        pcd.rotation.requires_grad_(True)

        pcd.opacity.requires_grad_(False)
        pcd.opacity[num_obj:] = pcd.inverse_opacity_activation(
            torch.Tensor([0.0 + eps])
        )
        pcd.opacity.requires_grad_(True)

        obj_mask = torch.Tensor([1] * num_obj + [0] * (num_all - num_obj)).to("cuda")

        inverse_activation = pcd.inverse_opacity_activation

        pcd.diff_density.requires_grad_(False)
        pcd.diff_density[num_obj:] = inverse_activation(torch.Tensor([1.0 - eps]))
        pcd.diff_density[:num_obj] = inverse_activation(torch.Tensor([0.0 + eps]))

        plane_mask = torch.cat(
            [
                (eigen_vec[0] + eigen_vec[1]).expand(num_obj, 3),
                torch.zeros(num_all - num_obj, 3),
            ]
        ).to("cuda")
        rot_mask = torch.cat(
            [
                torch.cat([eigen_vec[2], torch.Tensor([0])]).expand(num_obj, 4),
                torch.zeros(num_all - num_obj, 4),
            ]
        ).to("cuda")
        pcd.position.register_hook(lambda g: g * plane_mask)
        pcd.scaling.register_hook(lambda g: g * plane_mask)
        pcd.rotation.register_hook(lambda g: g * rot_mask)
        pcd.opacity.register_hook(lambda g: g * obj_mask[..., None])
        pcd.features.register_hook(lambda g: g * obj_mask[..., None, None])
        pcd.features_rest.register_hook(lambda g: g * obj_mask[..., None, None])

    def setup_object_point_cloud(self, position):
        e = self.diffuser_vec
        bb_min = self.diff_size[0][0] * e[0] + self.diff_size[0][1] * e[1]
        bb_max = self.diff_size[1][0] * e[0] + self.diff_size[1][1] * e[1]
        for i in range(3):
            position[:, i] = torch.Tensor((position.shape[0])).uniform_(
                self.obj_position[i] + bb_min[i], self.obj_position[i] + bb_max[i]
            )

    def setup_diffuser_point_cloud(self, position):
        e = self.diffuser_vec
        bb_min = self.diff_size[0][0] * e[0] + self.diff_size[0][1] * e[1]
        bb_max = self.diff_size[1][0] * e[0] + self.diff_size[1][1] * e[1]
        for i in range(3):
            position[:, i] = torch.Tensor((position.shape[0])).uniform_(
                self.diff_position[i] + bb_min[i], self.diff_position[i] + bb_max[i]
            )

    def intersect(self, position, camera_center):
        # compute intersection
        n = torch.Tensor(self.diff_normal).float().to("cuda")

        # distance to diffuser along normal axis
        d = torch.Tensor(self.diff_position).float().to("cuda")
        o = camera_center.to("cuda")
        do = d[None, :] - o
        d_dot_n = (do * n[None, :]).sum(dim=1, keepdim=True)

        # projected ray on normal axis
        p = torch.nn.functional.normalize(position - o, p=2, dim=1)
        p_dot_n = (p * n[None, :]).sum(dim=1, keepdim=True)

        t = torch.where(p_dot_n * p_dot_n > 0, d_dot_n / p_dot_n, -1)
        itsc = o + p * t

        axes = {"xy": [0, 1], "yz": [1, 2], "xz": [0, 2]}[self.plane_axes]
        is_diff_ray = (
            (itsc[:, axes] >= torch.Tensor(self.diff_size[0]).to("cuda")).all(dim=1)
            & (itsc[:, axes] <= torch.Tensor(self.diff_size[1]).to("cuda")).all(dim=1)
            & (t[..., 0] > 0)
        )[..., None]

        return itsc, is_diff_ray, p_dot_n, t

    def compute_refraction(self, position, camera_center, **kwargs):
        eta = self.get_eta
        po = position - camera_center.to("cuda")
        itsc, is_diff_ray, p_dot_n, _ = self.intersect(position, camera_center)
        n = torch.Tensor(self.diff_normal).float().to("cuda")
        p_n = nn.functional.normalize(po, p=2, dim=1)

        k = 1.0 - eta * eta * (1 - p_dot_n * p_dot_n)
        mask = (k > 0.0) & is_diff_ray

        k_sq = torch.where(mask, k.sqrt(), 0.0)
        refr_dist = torch.linalg.norm(position - itsc, dim=-1, keepdims=True)
        refr_dir = nn.functional.normalize(
            torch.where(mask, eta * p_n + (k_sq - eta * p_dot_n) * n, p_n), p=2, dim=1
        )
        refr_pos = torch.where(mask, itsc + refr_dist * refr_dir, position)
        return {"refr_pos": refr_pos, "refr_dir": refr_dir}
        # return {"refr_pos": position, "refr_dir": direction}

    def compute_blur_params(self, position, camera_center, blur_mult, **kwargs):
        _, is_diff_ray, v_dot_n, t = self.intersect(position, camera_center)
        d = torch.linalg.norm(
            position - camera_center.to("cuda"), dim=-1, keepdims=True
        )

        if is_diff_ray.sum() < 1:
            print("No diffuser ray!")
        # print(is_diff_ray.sum() / is_diff_ray.shape[0])

        cos = torch.where((-v_dot_n > 0) & is_diff_ray, -v_dot_n, 1)
        # z = torch.where(is_diff_ray, torch.clamp_min(t - d, 0), 0.0)
        z = torch.where(is_diff_ray, torch.clamp_min(d - t, 0), 0.0)
        # blur_scale = torch.clamp_min(z * self.get_diff_strength / cos, 0.0)
        blur_scale = torch.where(
            is_diff_ray,
            torch.clamp_min(z * self.get_diff_strength / cos / d, 0.0),
            0.0,
        )

        return {"blur_scale": blur_scale * blur_mult}

    def forward(self, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.

        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        render_dict = {
            "position": self.point_cloud.position,
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation,
            "shs": self.point_cloud.get_shs,
        }
        return render_dict

    def sigmoid_threshold(self, tensor, threshold):
        return torch.sigmoid((tensor - threshold * 0.5) / threshold)

    def total_variation(self, img):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

    def get_loss_dict(self, render_results, batch, blur_mult=1.0) -> dict:
        """
        Get the loss dictionary.

        Parameters
        ----------
        render_results : dict
            The render results which is the output of the renderer.
        batch : dict
            The batch of data which contains the ground truth images.

        Returns
        -------
        dict
            The loss dictionary which contain loss for backpropagation.
        """
        gt_images = torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))], dim=0
        )
        # diff_masks = torch.stack(
        #     [batch[i]["diffuser_mask"].to(self.device) for i in range(len(batch))],
        #     dim=0,
        # )
        L1_loss = l1_loss(render_results["rgb_blur"], gt_images)
        ssim_loss = 1.0 - ssim(render_results["rgb_blur"], gt_images)
        loss_coefs = {"ssim_loss": (ssim_loss, self.cfg.lambda_dssim)}

        # for logging
        loss_coefs["diff_strength"] = (self.get_diff_strength, 0.0)

        # for logging
        loss_coefs["eta"] = (self.get_eta, 0.0)

        if "rgb_deblur" in render_results:
            deblur_tv_loss = self.total_variation(render_results["rgb_deblur"])
            loss_coefs["deblur_tv_loss"] = (
                deblur_tv_loss * 1.0,
                self.cfg.lambda_deblur_tv,
            )

        loss_coefs["L1_loss"] = (
            L1_loss,
            1.0 - sum([v[1] for v in loss_coefs.values()]),
        )
        loss = sum([v[0] * v[1] for v in loss_coefs.values() if not v[0].isnan()])
        loss_dict = {k: v[0] for k, v in loss_coefs.items()}
        loss_dict["loss"] = loss

        if torch.isnan(loss).any():
            print("Loss contains nan")
        return loss_dict

    @torch.no_grad()
    def get_metric_dict(self, render_results, batch) -> dict:
        """
        Get the metric dictionary.

        Parameters
        ----------
        render_results : dict
            The render results which is the output of the renderer.
        batch : dict
            The batch of data which contains the ground truth images.

        Returns
        -------
        dict
            The metric dictionary which contains the metrics for evaluation.
        """
        metric_dict = {"rgb_file_name": batch[0]["camera"].rgb_file_name}
        if "image" in batch[0]:
            gt_images = torch.clamp(
                torch.stack(
                    [batch[i]["image"].to(self.device) for i in range(len(batch))],
                    dim=0,
                ),
                0.0,
                1.0,
            )
            L1_loss = l1_loss(render_results["rgb_blur"], gt_images).mean().double()
            psnr_test = (
                psnr(render_results["rgb_blur"].squeeze(), gt_images.squeeze())
                .mean()
                .double()
            )
            ssims_test = (
                ssim(render_results["rgb_blur"], gt_images, size_average=True)
                .mean()
                .item()
            )
            # lpips_test = lpips(
            #     render_results['images'],
            #     gt_images,
            #     net_type='vgg'
            # ).mean().item()

            metric_dict.update(
                {
                    "gt_images": gt_images,
                    "L1_loss": L1_loss,
                    "psnr": psnr_test,
                    "ssims": ssims_test,
                }
            )

        if "diff_acc_density" in render_results:
            diff_acc_density = torch.clamp(render_results["diff_acc_density"], 0, 1)
            metric_dict["diff_acc_density"] = diff_acc_density

        if "norm_diff" in render_results:
            norm_diff = render_results["norm_diff"]
            # nn.functional.normalize(render_results["norm_diff"], p=2, dim=1)
            metric_dict["norm_diff"] = norm_diff

        if "view_dir" in render_results:
            view_dir = render_results["view_dir"]
            # nn.functional.normalize(render_results["norm_diff"], p=2, dim=1)
            metric_dict["view_dir"] = view_dir

        if "diff_contact" in render_results:
            metric_dict["diff_contact"] = render_results["diff_contact"]

        if "diff_dist" in render_results:
            metric_dict["diff_dist"] = render_results["diff_dist"]

        rgb_images = {
            k: torch.clamp(img, 0.0, 1.0)
            for k, img in render_results.items()
            if k.startswith("rgb_")
        }
        blur_images = {
            k: img for k, img in render_results.items() if k.startswith("blur_")
        }
        depth_images = {
            k: img for k, img in render_results.items() if k.startswith("depth_")
        }
        alpha_images = {
            k: img for k, img in render_results.items() if k.startswith("alpha_")
        }

        metric_dict.update(
            {
                **rgb_images,
                **blur_images,
                **depth_images,
                **alpha_images,
            }
        )

        return metric_dict

    def get_state_dict(self):
        # additional_info = {"diffuser_model": self.diffuser_model.state_dict()}
        # return {**super().get_state_dict(), **additional_info}
        return super().get_state_dict()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # diffuser_model = state_dict.pop("diffuser_model")

        # if diffuser_model:
        #     self.diffuser_model.load_state_dict(diffuser_model)

        return super().load_state_dict(state_dict, strict, assign)
