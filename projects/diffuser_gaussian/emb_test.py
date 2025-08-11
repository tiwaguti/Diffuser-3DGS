import torch
import torch.nn as nn


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_value = self.kwargs["max_value"]
        N_kernels = self.kwargs["num_kernels"]

        shifts = torch.linspace(0.0, max_value, steps=N_kernels)

        for shift in shifts:
            embed_fns.append(
                lambda x, k_fn=self.kwargs["kernel_fn"], shift=shift: k_fn(
                    (x / self.kwargs["max_value"] - shift),
                    w=self.kwargs["width"],
                )
                / self.kwargs["width"]
            )
            out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    # kde
    embed_kwargs = {
        "include_input": False,
        "input_dims": i,
        "max_value": torch.pi,
        "num_kernels": multires,
        "log_sampling": False,
        "width": 0.08,
        "kernel_fn": lambda x, w: (-x.pow(2) / (2.0 * w**2.0)).exp()
        / (2.0 * torch.pi * w**2) ** 0.5,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class KDEEmbedding:
    def __init__(self, kernel, xmin=0.0, xmax=1.0, width=0.05, deg=10) -> None:
        self.deg = deg
        self.shifts = torch.arange(xmin, xmax, step=(xmax-xmin)/deg)
        self.embed_fns = [
            lambda x, s=s: kernel((x-s) / width) / width for s in self.shifts
        ]

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


    def get_estimator(self, coefs):
        return lambda x: sum([coefs[i] * fn(x) for i, fn in enumerate(self.embed_fns)])


def gaussian_kernel(x):
    return (-0.5 * x.pow(2)).exp() / (2.0 * torch.pi) ** 0.5

def pillbox_kernel(x):
    # mask = (x >= -w * 0.5) & (x <= w * 0.5)
    mask = (x >= 0) & (x < 1)
    return 1.0  * mask


import matplotlib.pyplot as plt

deg = 100
width = 1 / deg
gaussian_embedder = KDEEmbedding(kernel=gaussian_kernel, deg=deg, width=width)

pillbox_embedder = KDEEmbedding(kernel=pillbox_kernel, deg=deg, width=width)

# xi = torch.linspace(0, torch.pi, 32)
yi = torch.cat([torch.randn((1000)) * 0.02 + 0.3, torch.randn((1000)) * 0.2 + 0.7, torch.randn((1000)) * 0.04 + 0.5])
# yi = torch.exp(-20.0 * (xi - 1.0) ** 2.0)
# plt.figure()
# plt.plot(xi.numpy(), yi.numpy())


x_grid = torch.linspace(0, 1.0, 1000)

f_g = gaussian_embedder.embed(yi[:, None])  # [0]
f_sum_g = torch.mean(f_g, dim=0)
estimator_g = gaussian_embedder.get_estimator(f_sum_g)
y_est_g = estimator_g(x_grid) / x_grid.shape[0]


f_p = pillbox_embedder.embed(yi[:, None])  # [0]
f_sum_p = torch.mean(f_p, dim=0)
estimator_p = pillbox_embedder.get_estimator(f_sum_p)
y_est_p = estimator_p(x_grid) / x_grid.shape[0]
print(y_est_p)

import numpy as np
np.savetxt("tmp.txt", torch.cat([x_grid.view(-1, 1), y_est_p.view(-1, 1)], dim=-1).cpu().numpy())

plt.figure()
plt.plot(x_grid.numpy(), y_est_g.numpy())
plt.plot(x_grid.numpy(), y_est_p.numpy())
plt.show()
