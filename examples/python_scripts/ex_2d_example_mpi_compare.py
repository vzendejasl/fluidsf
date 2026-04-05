"""
Compare the main 2D example workflow against the MPI backend on the sample dataset.

Run with:
    mpirun -launcher fork -n 4 python examples/python_scripts/ex_2d_example_mpi_compare.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pooch
from mpi4py import MPI

import fluidsf


def load_data():
    file_path = pooch.retrieve(
        url="https://zenodo.org/records/15278227/files/2layer_128.jld2",
        known_hash="a04abc602ca3bbc4ff9a868a96848b6815a17f697202fb12e3ff40762de92ec6",
    )
    f = h5py.File(file_path, "r")
    grid = f["grid"]
    snapshots = f["snapshots"]
    x = grid["x"][()]
    y = grid["y"][()]
    u = snapshots["u"]["20050"][0]
    v = snapshots["v"]["20050"][0]
    q = snapshots["q"]["20050"][0]
    return x, y, u, v, q


def _compare_case(name, serial_sf, mpi_sf, keys):
    diffs = {}
    for key in keys:
        diffs[key] = float(np.max(np.abs(serial_sf[key] - mpi_sf[key])))
        print(f"{name} {key}: max abs diff = {diffs[key]:.3e}")
    return diffs


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

    x, y, u, v, q = load_data()

    default_mpi = fluidsf.generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        backend="mpi",
        comm=comm,
    )
    full_mpi = fluidsf.generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LLL", "LL", "LTT", "LSS"],
        scalar=q,
        backend="mpi",
        comm=comm,
    )

    if rank != 0:
        return

    default_serial = fluidsf.generate_structure_functions_2d(u, v, x, y)
    full_serial = fluidsf.generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LLL", "LL", "LTT", "LSS"],
        scalar=q,
    )

    default_keys = [
        "SF_advection_velocity_x",
        "SF_advection_velocity_y",
    ]
    full_keys = [
        "SF_advection_velocity_x",
        "SF_advection_velocity_y",
        "SF_advection_scalar_x",
        "SF_advection_scalar_y",
        "SF_LLL_x",
        "SF_LLL_y",
        "SF_LL_x",
        "SF_LL_y",
        "SF_LTT_x",
        "SF_LTT_y",
        "SF_LSS_x",
        "SF_LSS_y",
    ]

    default_diffs = _compare_case("default", default_serial, default_mpi, default_keys)
    full_diffs = _compare_case("full", full_serial, full_mpi, full_keys)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].semilogx(
        default_serial["x-diffs"],
        default_serial["SF_advection_velocity_x"],
        color="C0",
        linestyle="-",
        label="ASF_V x serial",
    )
    axes[0].semilogx(
        default_mpi["x-diffs"],
        default_mpi["SF_advection_velocity_x"],
        color="C0",
        linestyle="--",
        label="ASF_V x mpi",
    )
    axes[0].semilogx(
        default_serial["y-diffs"],
        default_serial["SF_advection_velocity_y"],
        color="C1",
        linestyle="-",
        label="ASF_V y serial",
    )
    axes[0].semilogx(
        default_mpi["y-diffs"],
        default_mpi["SF_advection_velocity_y"],
        color="C1",
        linestyle="--",
        label="ASF_V y mpi",
    )

    axes[1].semilogx(full_serial["x-diffs"], full_serial["SF_LL_x"], color="C2", linestyle="-", label="LL x serial")
    axes[1].semilogx(full_mpi["x-diffs"], full_mpi["SF_LL_x"], color="C2", linestyle="--", label="LL x mpi")
    axes[1].semilogx(full_serial["y-diffs"], full_serial["SF_LLL_y"], color="C3", linestyle="-", label="LLL y serial")
    axes[1].semilogx(full_mpi["y-diffs"], full_mpi["SF_LLL_y"], color="C3", linestyle="--", label="LLL y mpi")

    axes[2].semilogx(full_serial["x-diffs"], full_serial["SF_LTT_x"], color="C4", linestyle="-", label="LTT x serial")
    axes[2].semilogx(full_mpi["x-diffs"], full_mpi["SF_LTT_x"], color="C4", linestyle="--", label="LTT x mpi")
    axes[2].semilogx(full_serial["y-diffs"], full_serial["SF_LSS_y"], color="C5", linestyle="-", label="LSS y serial")
    axes[2].semilogx(full_mpi["y-diffs"], full_mpi["SF_LSS_y"], color="C5", linestyle="--", label="LSS y mpi")

    for ax in axes:
        ax.set_xlabel("Separation distance")
        ax.set_ylabel("Structure function")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "ex_2d_example_mpi_compare.png", dpi=150)
    plt.close(fig)

    np.savez(
        output_dir / "ex_2d_example_mpi_compare.npz",
        x_diffs_default=default_serial["x-diffs"],
        y_diffs_default=default_serial["y-diffs"],
        x_diffs_full=full_serial["x-diffs"],
        y_diffs_full=full_serial["y-diffs"],
        **{f"default_serial_{key}": default_serial[key] for key in default_keys},
        **{f"default_mpi_{key}": default_mpi[key] for key in default_keys},
        **{f"full_serial_{key}": full_serial[key] for key in full_keys},
        **{f"full_mpi_{key}": full_mpi[key] for key in full_keys},
    )
    print(f"Saved outputs to {output_dir}")
    print(f"Largest default diff = {max(default_diffs.values()):.3e}")
    print(f"Largest full diff = {max(full_diffs.values()):.3e}")


if __name__ == "__main__":
    main()
