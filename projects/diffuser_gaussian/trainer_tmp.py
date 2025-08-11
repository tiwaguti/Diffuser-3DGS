import torch
import torch.nn.functional as F
import dptr.gs as gs
from pointrix.renderer.dptr import DPTRRender, RENDERER_REGISTRY
from pointrix.utils.renderer.renderer_utils import RenderFeatures


@RENDERER_REGISTRY.register()
class DiffuserDPTRRender(DPTRRender):
    def compute_geometry(
        self,
        position,
        scaling,
        rotation,
        shs,
        camera_center,
        extrinsic_matrix,
        intrinsic_matrix,
        height,
        width,
        **kwargs,
    ):

        (uv, depth) = gs.project_point(
            position, intrinsic_matrix.cuda(), extrinsic_matrix.cuda(), width, height
        )
        visible = depth != 0

        # compute cov3d
        cov3d = gs.compute_cov3d(scaling, rotation, visible)
        direction = position.cuda() - camera_center.repeat(position.shape[0], 1).cuda()
        direction = direction / direction.norm(dim=1, keepdim=True)
        rgb = gs.compute_sh(shs, 3, direction)
        return {
            "uv": uv,
            "depth": depth,
            "visible": visible,
            "direction": direction,
            "cov3d": cov3d,
            "rgb": rgb,
        }

    def render_diffuser_features(
        self,
        diffuser_features,
        bg_color,
        depth,
        uv,
        visible,
        height,
        width,
        extrinsic_matrix,
        intrinsic_matrix,
        cov3d,
        direction,
        position,
        diffuse_density,
        **kwargs,
    ) -> dict:
        """
        Render the point cloud for one iteration

        Parameters
        ----------
        FovX : float
            The field of view in the x-axis.
        FovY : float
            The field of view in the y-axis.
        height : float
            The height of the image.
        width : float
            The width of the image.
        world_view_transform : torch.Tensor
            The world view transformation matrix.
        full_proj_transform : torch.Tensor
            The full projection transformation matrix.
        camera_center : torch.Tensor
            The camera center.
        position : torch.Tensor
            The position of the point cloud.
        opacity : torch.Tensor
            The opacity of the point cloud.
        scaling : torch.Tensor
            The scaling of the point cloud.
        rotation : torch.Tensor
            The rotation of the point cloud.
        shs : torch.Tensor
            The spherical harmonics of the point cloud.
        scaling_modifier : float, optional
            The scaling modifier, by default 1.0
        render_xyz : bool, optional
            Whether to render the xyz or not, by default False

        Returns
        -------
        dict
            The rendered image, the viewspace points,
            the visibility filter, the radii, the xyz,
            the color, the rotation, the scales, and the xy.
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # import pdb; pdb.set_trace()
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.

        (conic, radius, tiles_touched) = gs.ewa_project(
            position,
            cov3d,
            intrinsic_matrix.cuda(),
            extrinsic_matrix.cuda(),
            uv,
            width,
            height,
            visible,
        )

        # sort
        (gaussian_ids_sorted, tile_range) = gs.sort_gaussian(
            uv, depth, width, height, radius, tiles_touched
        )

        ndc = torch.zeros_like(uv, requires_grad=True)
        # alpha blending
        try:
            ndc.retain_grad()
        except:
            raise ValueError("ndc does not have grad")

        rendered_features = gs.alpha_blending(
            uv,
            conic,
            diffuse_density,
            diffuser_features.combine(),
            gaussian_ids_sorted,
            tile_range,
            bg_color,
            width,
            height,
            ndc,
        )
        rendered_features_split = diffuser_features.split(rendered_features)

        return {
            "rendered_features_split": rendered_features_split,
            "viewspace_points": ndc,
            "visibility_filter": radius > 0,
            "radii": radius,
        }

    def render_blur(
        self,
        Render_Features,
        height,
        width,
        extrinsic_matrix,
        intrinsic_matrix,
        position,
        opacity,
        shs,
        cov3d,
        direction,
        uv,
        depth,
        visible,
        blur_scale,
        bg_color=None,
        **kwargs,
    ):
        # ewa project with blur
        fx, fy = intrinsic_matrix[0], intrinsic_matrix[1]
        blur_opacity = 1.0 / (1.0 + blur_scale * torch.sqrt(fx * fx + fy * fy))
        (conic, radius, tiles_touched) = gs.ewa_project_blur(
            position,
            cov3d,
            intrinsic_matrix.cuda(),
            extrinsic_matrix.cuda(),
            uv,
            blur_scale,
            width,
            height,
            visible,
        )

        # sort
        (gaussian_ids_sorted, tile_range) = gs.sort_gaussian(
            uv, depth, width, height, radius, tiles_touched
        )

        ndc = torch.zeros_like(uv, requires_grad=True)
        # alpha blending
        try:
            ndc.retain_grad()
        except:
            raise ValueError("ndc does not have grad")

        rendered_features = gs.alpha_blending(
            uv,
            conic,
            opacity,  # * blur_opacity,
            Render_Features.combine(),
            gaussian_ids_sorted,
            tile_range,
            bg_color if bg_color else self.bg_color,
            width,
            height,
            ndc,
        )
        rendered_features_split = Render_Features.split(rendered_features)

        return {
            "rendered_features_split": rendered_features_split,
            "viewspace_points": ndc,
            "visibility_filter": radius > 0,
            "radii": radius,
        }

    def render_deblur(
        self,
        Render_Features,
        height,
        width,
        extrinsic_matrix,
        intrinsic_matrix,
        position,
        opacity,
        shs,
        cov3d,
        direction,
        uv,
        depth,
        visible,
        bg_color=None,
        **kwargs,
    ):
        (conic, radius, tiles_touched) = gs.ewa_project(
            position,
            cov3d,
            intrinsic_matrix.cuda(),
            extrinsic_matrix.cuda(),
            uv,
            width,
            height,
            visible,
        )

        # sort
        (gaussian_ids_sorted, tile_range) = gs.sort_gaussian(
            uv, depth, width, height, radius, tiles_touched
        )

        ndc = torch.zeros_like(uv, requires_grad=True)
        # alpha blending
        try:
            ndc.retain_grad()
        except:
            raise ValueError("ndc does not have grad")

        rendered_features = gs.alpha_blending(
            uv,
            conic,
            opacity,
            Render_Features.combine(),
            gaussian_ids_sorted,
            tile_range,
            bg_color if bg_color else self.bg_color,
            width,
            height,
            ndc,
        )
        rendered_features_split = Render_Features.split(rendered_features)

        return {
            "rendered_features_split": rendered_features_split,
            "viewspace_points": ndc,
            "visibility_filter": radius > 0,
            "radii": radius,
        }
