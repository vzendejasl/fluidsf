"""
Compare the legacy 2D implementation against the MPI backend.

Run with:
    mpirun -launcher fork -n 4 python examples/python_scripts/ex_2d_mpi_compare.py
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
    x = np.linspace(0.0, 1.0, 96)
    y = np.linspace(0.0, 1.0, 96)
    X, _ = np.meshgrid(x, y)
    u = X
    v = 0.5 * X
    scalar = 0.25 * X
    return x, y, u, v, scalar


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

    x, y, u, v, scalar = build_data()
    sf_type = ["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"]
    boundary = "periodic-x"

    mpi_sf = fluidsf.generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=sf_type,
        scalar=scalar,
        boundary=boundary,
        backend="mpi",
        comm=comm,
    )

    if rank != 0:
        return

    serial_sf = fluidsf.generate_structure_functions_2d(
        u,
        v,
        x,
        y,
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
    for key, color in zip(["SF_LL_x", "SF_LLL_x", "SF_LTT_x"], ["C0", "C1", "C2"]):
        axes[0].plot(serial_sf["x-diffs"], serial_sf[key], color=color, linestyle="-", label=f"{key} serial")
        axes[0].plot(mpi_sf["x-diffs"], mpi_sf[key], color=color, linestyle="--", label=f"{key} mpi")
    for key, color in zip(["SF_advection_velocity_x", "SF_advection_scalar_x", "SF_LSS_x"], ["C3", "C4", "C5"]):
        axes[1].plot(serial_sf["x-diffs"], serial_sf[key], color=color, linestyle="-", label=f"{key} serial")
        axes[1].plot(mpi_sf["x-diffs"], mpi_sf[key], color=color, linestyle="--", label=f"{key} mpi")
    axes[0].set_title("Velocity SFs Along x")
    axes[1].set_title("Advective/Scalar SFs Along x")
    for ax in axes:
        ax.set_xlabel("Separation distance")
        ax.set_ylabel("Structure function")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "ex_2d_mpi_compare.png", dpi=150)
    plt.close(fig)

    np.savez(
        output_dir / "ex_2d_mpi_compare.npz",
        x_diffs=serial_sf["x-diffs"],
        y_diffs=serial_sf["y-diffs"],
        **{f"serial_{key}": serial_sf[key] for key in comparison_keys},
        **{f"mpi_{key}": mpi_sf[key] for key in comparison_keys},
    )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
