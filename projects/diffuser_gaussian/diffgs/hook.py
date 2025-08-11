import os
from pointrix.hook import HOOK_REGISTRY, LogHook

from diffgs.visualize import visualize_render_features
import torch


@HOOK_REGISTRY.register()
class DeblurLogHook(LogHook):
    """
    A hook to log the training and validation losses.
    """

    @torch.no_grad()
    def after_val_iter(self, trainner) -> None:
        self.progress_bar.update("validation", step=1)
        for key, value in trainner.metric_dict.items():
            if key in self.losses_test:
                self.losses_test[key] += value

        image_name = os.path.basename(trainner.metric_dict["rgb_file_name"])
        iteration = trainner.global_step

        visual_dict = visualize_render_features(trainner.metric_dict, tensorboard=True)
        for feature_name, vis in visual_dict.items():
            trainner.logger.write_image(
                "test" + f"_view_{image_name}/{feature_name}",
                vis.squeeze(),
                step=iteration,
            )

        trainner.logger.write_image(
            "test" + f"_view_{image_name}/ground_truth",
            trainner.metric_dict["gt_images"].squeeze(),
            step=iteration,
        )
