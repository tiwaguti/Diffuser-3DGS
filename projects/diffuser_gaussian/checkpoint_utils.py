import torch
from typing import List, Union
from pathlib import Path
from torch.optim import Optimizer
from torch import nn
import sys
from itertools import zip_longest

sys.path.append("../../")
from pointrix.point_cloud.utils import unwarp_name


def replace_optimizer(points, new_atributes: dict, optimizer: Optimizer) -> dict:
    """
    replace the point cloud in optimizer with new atribute.

    Parameters
    ----------
    new_atributes: dict
        The dict of new atributes.
    optimizer: Optimizer
        The optimizer for the point cloud.
    """
    optimizable_tensors = {}
    org_optimizer_state = optimizer.state.copy()
    for group, state in zip_longest(optimizer.param_groups, org_optimizer_state):
        for key, replace_tensor in new_atributes.items():
            if group["name"] == points.prefix_name + key:
                # stored_state = optimizer.state.get(group["params"][0], None)
                if state is None:
                    # if state is missing, fill it with the state of first element
                    stored_state = optimizer.state[
                        optimizer.param_groups[0]["params"][0]
                    ].copy()
                else:
                    stored_state = optimizer.state.get(state, None)

                stored_state["exp_avg"] = torch.zeros_like(replace_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(replace_tensor)

                if state is not None:
                    del optimizer.state[state]
                group["params"][0] = nn.Parameter(
                    replace_tensor.contiguous().requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[unwarp_name(group["name"])] = group["params"][0]

    return optimizable_tensors


def replace_points(
    points, new_atributes: dict, optimizer: Union[Optimizer, None] = None
) -> None:
    """
    replace atribute of the point cloud with new atribute.

    Parameters
    ----------
    new_atributes: dict
        The dict of new atributes.
    optimizer: Optimizer
        The optimizer for the point cloud.
    """
    if optimizer is not None:
        replace_tensor = replace_optimizer(points, new_atributes, optimizer)
        for key, value in replace_tensor.items():
            setattr(points, key, value)
    else:
        for key, value in new_atributes.items():
            name = key
            value = getattr(points, name)
            replace_atribute = nn.Parameter(value.contiguous().requires_grad_(True))
            setattr(points, key, replace_atribute)


def extend_model(trainer, state_dict, append):
    point_attributes = {
        k.split(".")[-1]: v for k, v in state_dict.items() if "point_cloud." in k
    }
    if not append:
        trainer.model.load_state_dict(state_dict)
        replace_points(
            trainer.model.point_cloud,
            point_attributes,
            trainer.optimizer.optimizer_dict["optimizer_1"].optimizer,
        )
        return

    # print("add", point_attributes["position"].shape)
    trainer.model.point_cloud.extand_points(
        point_attributes,
        trainer.optimizer.optimizer_dict["optimizer_1"].optimizer,
    )
    return
