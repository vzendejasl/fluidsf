"""
Compare the legacy 3D implementation against the MPI backend.

Run with:
    mpirun -launcher fork -n 4 python examples/python_scripts/ex_3d_mpi_compare.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import fluidsf


def build_data():
    x = np.linspace(0.0, 1.0, 12)
    y = np.linspace(0.0, 1.0, 12)
    z = np.linspace(0.0, 1.0, 12)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")
    scalar = u + 2.0 * v
    return x, y, z, u, v, w, scalar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs"),
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, z, u, v, w, scalar = build_data()
    sf_type = ["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"]
    boundary = ["periodic-x", "periodic-y"]

    mpi_sf = fluidsf.generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=sf_type,
        scalar=scalar,
        boundary=boundary,
        backend="mpi",
        comm=comm,
    )

    if rank != 0:
        return

    serial_sf = fluidsf.generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=sf_type,
        scalar=scalar,
        boundary=boundary,
    )

    comparison_keys = [
        "SF_advection_velocity_x",
        "SF_advection_scalar_x",
        "SF_LL_x",
        "SF_TT_x",
        "SF_SS_x",
        "SF_LLL_x",
        "SF_LTT_x",
        "SF_LSS_x",
    ]
    for key in comparison_keys:
        diff = float(np.max(np.abs(serial_sf[key] - mpi_sf[key])))
        print(f"{key}: max abs diff = {diff:.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis_index, axis_name in enumerate(["x", "y", "z"]):
        axes[axis_index].plot(
            serial_sf[f"x-diffs" if axis_name == "x" else f"{axis_name}-diffs"],
            serial_sf[f"SF_LL_{axis_name}"],
            color="C0",
            linestyle="-",
            label="SF_LL serial",
        )
        axes[axis_index].plot(
            mpi_sf[f"x-diffs" if axis_name == "x" else f"{axis_name}-diffs"],
            mpi_sf[f"SF_LL_{axis_name}"],
            color="C0",
            linestyle="--",
            label="SF_LL mpi",
        )
        axes[axis_index].plot(
            serial_sf[f"x-diffs" if axis_name == "x" else f"{axis_name}-diffs"],
            serial_sf[f"SF_advection_velocity_{axis_name}"],
            color="C1",
            linestyle="-",
            label="ASF_V serial",
        )
        axes[axis_index].plot(
            mpi_sf[f"x-diffs" if axis_name == "x" else f"{axis_name}-diffs"],
            mpi_sf[f"SF_advection_velocity_{axis_name}"],
            color="C1",
            linestyle="--",
            label="ASF_V mpi",
        )
        axes[axis_index].set_title(f"3D Along {axis_name}")
        axes[axis_index].set_xlabel("Separation distance")
        axes[axis_index].set_ylabel("Structure function")
        axes[axis_index].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "ex_3d_mpi_compare.png", dpi=150)
    plt.close(fig)

    np.savez(
        output_dir / "ex_3d_mpi_compare.npz",
        x_diffs=serial_sf["x-diffs"],
        y_diffs=serial_sf["y-diffs"],
        z_diffs=serial_sf["z-diffs"],
        **{f"serial_{key}": serial_sf[key] for key in comparison_keys},
        **{f"mpi_{key}": mpi_sf[key] for key in comparison_keys},
    )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
