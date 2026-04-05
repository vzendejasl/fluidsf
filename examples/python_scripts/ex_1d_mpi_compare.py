"""
Compare the legacy 1D implementation against the MPI backend.

Run with:
    mpirun -launcher fork -n 4 python examples/python_scripts/ex_1d_mpi_compare.py
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
    x = np.linspace(0.0, 1.0, 128)
    u = x
    v = 0.5 * x
    scalar = 0.25 * x
    return x, u, v, scalar


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

    x, u, v, scalar = build_data()
    sf_type = ["LL", "LLL", "LTT", "LSS"]
    mpi_sf = fluidsf.generate_structure_functions_1d(
        u,
        x,
        sf_type=sf_type,
        v=v,
        scalar=scalar,
        boundary=None,
        backend="mpi",
        comm=comm,
    )

    if rank != 0:
        return

    serial_sf = fluidsf.generate_structure_functions_1d(
        u,
        x,
        sf_type=sf_type,
        v=v,
        scalar=scalar,
        boundary=None,
    )

    comparison_keys = ["SF_LL", "SF_LLL", "SF_LTT", "SF_LSS"]
    for key in comparison_keys:
        diff = float(np.max(np.abs(serial_sf[key] - mpi_sf[key])))
        print(f"{key}: max abs diff = {diff:.3e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color in zip(comparison_keys, ["C0", "C1", "C2", "C3"]):
        ax.plot(serial_sf["x-diffs"], serial_sf[key], color=color, linestyle="-", label=f"{key} serial")
        ax.plot(mpi_sf["x-diffs"], mpi_sf[key], color=color, linestyle="--", label=f"{key} mpi")
    ax.set_xlabel("Separation distance")
    ax.set_ylabel("Structure function")
    ax.set_title("1D Serial vs MPI")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ex_1d_mpi_compare.png", dpi=150)
    plt.close(fig)

    np.savez(
        output_dir / "ex_1d_mpi_compare.npz",
        x_diffs=serial_sf["x-diffs"],
        serial_SF_LL=serial_sf["SF_LL"],
        mpi_SF_LL=mpi_sf["SF_LL"],
        serial_SF_LLL=serial_sf["SF_LLL"],
        mpi_SF_LLL=mpi_sf["SF_LLL"],
        serial_SF_LTT=serial_sf["SF_LTT"],
        mpi_SF_LTT=mpi_sf["SF_LTT"],
        serial_SF_LSS=serial_sf["SF_LSS"],
        mpi_SF_LSS=mpi_sf["SF_LSS"],
    )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
