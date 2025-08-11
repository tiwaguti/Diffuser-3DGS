from plyfile import PlyData, PlyElement
import argparse
import numpy as np
import numpy.ma as ma
import torch


def load_ply(src):
    plydata = PlyData.read(src)
    return plydata


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_ply")
    parser.add_argument("--diffuser_ply")
    args = parser.parse_args()

    scene_pcd = load_ply(args.scene_ply)
    scene_pcd["vertex"]["blur_features_0"][:] = inverse_sigmoid(
        scene_pcd["vertex"]["blur_features_0"] * 0.0 + 1e-6
    )
    scene_pcd["vertex"]["blur_features_1"][:] = inverse_sigmoid(
        scene_pcd["vertex"]["blur_features_1"] * 0.0 + 1e-6
    )

    diffuser_pcd = load_ply(args.diffuser_ply)
    mask = (
        (diffuser_pcd["vertex"]["x"] >= -0.5)
        & (diffuser_pcd["vertex"]["x"] <= 0.5)
        & (diffuser_pcd["vertex"]["z"] >= -0.5)
        & (diffuser_pcd["vertex"]["z"] <= 0.5)
        & (diffuser_pcd["vertex"]["y"] <= -0.9)
    )

    diffuser_pcd["vertex"]["blur_features_0"][:] = inverse_sigmoid(
        np.zeros_like(diffuser_pcd["vertex"]["blur_features_0"]) + 1e-6
    )
    diffuser_pcd["vertex"]["blur_features_1"][:] = inverse_sigmoid(
        np.ones_like(diffuser_pcd["vertex"]["blur_features_1"]) * 0.5
    )
    diffuser_pcd["vertex"]["opacity_0"][:] = inverse_sigmoid(
        diffuser_pcd["vertex"]["opacity_0"] * 0.0 + 1e-6
    )

    merged = np.concatenate(
        [scene_pcd["vertex"].data, diffuser_pcd["vertex"].data[mask]]
    )
    scene_pcd["vertex"].data = merged
    scene_pcd.write("outputs/merged.ply")


if __name__ == "__main__":
    main()
