"""
Compare the main 3D example workflow against the MPI backend on the sample dataset.

Run with:
    mpirun -launcher fork -n 4 python examples/python_scripts/ex_3d_example_mpi_compare.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pooch
import xarray as xr
from mpi4py import MPI

import fluidsf


def load_data():
    file_path = pooch.retrieve(
        url="https://zenodo.org/records/15278227/files/langmuir_fields.nc",
        known_hash="d32f5f4c02791ddc584abecc572a5f06a948638ba4e45c4d6952a0723c3c1e40",
    )
    ds = xr.load_dataset(file_path).isel(time=1)
    # The sample file stores fields in (z, y, x). Convert to the public
    # (x, y, z) layout expected by the MPI backend so serial and MPI compare
    # the same physical subset directly.
    u = np.transpose(ds.u.values[-60:, :, :], (2, 1, 0))
    v = np.transpose(ds.v.values[-60:, :, :], (2, 1, 0))
    w = np.transpose(ds.w.values[-60:, :, :], (2, 1, 0))
    return (u, v, w, ds.xF.values[:], ds.yF.values[:], ds.zF.values[-60:])


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

    u, v, w, x, y, z = load_data()
    sf_type = ["ASF_V", "LL", "LLL", "LTT"]
    boundary = ["periodic-x", "periodic-y"]

    mpi_sf = fluidsf.generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=sf_type,
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
        boundary=boundary,
    )

    comparison_keys = [
        "SF_advection_velocity_x",
        "SF_advection_velocity_y",
        "SF_advection_velocity_z",
        "SF_LL_x",
        "SF_LL_y",
        "SF_LL_z",
        "SF_LLL_x",
        "SF_LLL_y",
        "SF_LLL_z",
        "SF_LTT_x",
        "SF_LTT_y",
        "SF_LTT_z",
    ]
    diffs = {}
    for key in comparison_keys:
        diffs[key] = float(np.max(np.abs(serial_sf[key] - mpi_sf[key])))
        print(f"{key}: max abs diff = {diffs[key]:.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis_index, axis_name in enumerate(["x", "y", "z"]):
        diff_key = f"{axis_name}-diffs"
        axes[axis_index].plot(
            serial_sf[diff_key],
            serial_sf[f"SF_LL_{axis_name}"],
            color="C0",
            linestyle="-",
            label="LL serial",
        )
        axes[axis_index].plot(
            mpi_sf[diff_key],
            mpi_sf[f"SF_LL_{axis_name}"],
            color="C0",
            linestyle="--",
            label="LL mpi",
        )
        axes[axis_index].plot(
            serial_sf[diff_key],
            serial_sf[f"SF_advection_velocity_{axis_name}"],
            color="C1",
            linestyle="-",
            label="ASF_V serial",
        )
        axes[axis_index].plot(
            mpi_sf[diff_key],
            mpi_sf[f"SF_advection_velocity_{axis_name}"],
            color="C1",
            linestyle="--",
            label="ASF_V mpi",
        )
        axes[axis_index].set_title(f"Example 3D Along {axis_name}")
        axes[axis_index].set_xlabel("Separation distance [m]")
        axes[axis_index].set_ylabel("Structure function")
        axes[axis_index].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "ex_3d_example_mpi_compare.png", dpi=150)
    plt.close(fig)

    np.savez(
        output_dir / "ex_3d_example_mpi_compare.npz",
        x_diffs=serial_sf["x-diffs"],
        y_diffs=serial_sf["y-diffs"],
        z_diffs=serial_sf["z-diffs"],
        **{f"serial_{key}": serial_sf[key] for key in comparison_keys},
        **{f"mpi_{key}": mpi_sf[key] for key in comparison_keys},
    )
    print(f"Saved outputs to {output_dir}")
    print(f"Largest diff = {max(diffs.values()):.3e}")


if __name__ == "__main__":
    main()
