"""
Compare the cascade-rate tutorial workflow in serial and MPI on sample data.

Run with:
    mpirun -launcher fork -n 4 python examples/python_scripts/ex_cascade_serial_vs_mpi_sample_data.py
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
from scipy.stats import bootstrap

import fluidsf


BOOTSTRAP_KWARGS = {
    "confidence_level": 0.5,
    "axis": 0,
    "random_state": 0,
}


def max_abs_diff_ignore_nonfinite(serial_values, mpi_values):
    serial_values = np.asarray(serial_values)
    mpi_values = np.asarray(mpi_values)
    valid = np.isfinite(serial_values) & np.isfinite(mpi_values)
    if not np.any(valid):
        return 0.0
    return float(np.max(np.abs(serial_values[valid] - mpi_values[valid])))


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
    u = snapshots["u"]
    v = snapshots["v"]
    return x, y, u, v


def compute_sf_snapshots(u, v, x, y, snapshot_keys, *, backend="serial", comm=None):
    kwargs = {}
    if backend == "mpi":
        kwargs = {"backend": "mpi", "comm": comm}

    return [
        fluidsf.generate_structure_functions_2d(
            u[snapshot][0],
            v[snapshot][0],
            x,
            y,
            sf_type=["ASF_V", "LLL"],
            boundary="periodic-all",
            **kwargs,
        )
        for snapshot in snapshot_keys
    ]


def bootstrap_sf_statistics(sfs_list):
    sf_asf_x = np.asarray([sf["SF_advection_velocity_x"] for sf in sfs_list])
    sf_asf_y = np.asarray([sf["SF_advection_velocity_y"] for sf in sfs_list])
    sf_lll_x = np.asarray([sf["SF_LLL_x"] for sf in sfs_list])
    sf_lll_y = np.asarray([sf["SF_LLL_y"] for sf in sfs_list])

    boot_asf_x = bootstrap((sf_asf_x,), np.mean, **BOOTSTRAP_KWARGS)
    boot_asf_y = bootstrap((sf_asf_y,), np.mean, **BOOTSTRAP_KWARGS)
    boot_lll_x = bootstrap((sf_lll_x,), np.mean, **BOOTSTRAP_KWARGS)
    boot_lll_y = bootstrap((sf_lll_y,), np.mean, **BOOTSTRAP_KWARGS)

    return {
        "boot_ASF_x_mean": boot_asf_x.bootstrap_distribution.mean(axis=1),
        "boot_ASF_y_mean": boot_asf_y.bootstrap_distribution.mean(axis=1),
        "boot_LLL_x_mean": boot_lll_x.bootstrap_distribution.mean(axis=1),
        "boot_LLL_y_mean": boot_lll_y.bootstrap_distribution.mean(axis=1),
        "boot_ASF_x_conf_low": boot_asf_x.confidence_interval[0],
        "boot_ASF_x_conf_high": boot_asf_x.confidence_interval[1],
        "boot_ASF_y_conf_low": boot_asf_y.confidence_interval[0],
        "boot_ASF_y_conf_high": boot_asf_y.confidence_interval[1],
        "boot_LLL_x_conf_low": boot_lll_x.confidence_interval[0],
        "boot_LLL_x_conf_high": boot_lll_x.confidence_interval[1],
        "boot_LLL_y_conf_low": boot_lll_y.confidence_interval[0],
        "boot_LLL_y_conf_high": boot_lll_y.confidence_interval[1],
    }


def estimate_cascade_rates(stats, x_diffs, y_diffs):
    return {
        "epsilon_LLL_x_mean": -2 * stats["boot_LLL_x_mean"] / (3 * x_diffs),
        "epsilon_LLL_y_mean": -2 * stats["boot_LLL_y_mean"] / (3 * y_diffs),
        "epsilon_ASF_x_mean": -stats["boot_ASF_x_mean"] / 2,
        "epsilon_ASF_y_mean": -stats["boot_ASF_y_mean"] / 2,
        "epsilon_LLL_x_conf_low": -2 * stats["boot_LLL_x_conf_low"] / (3 * x_diffs),
        "epsilon_LLL_x_conf_high": -2 * stats["boot_LLL_x_conf_high"] / (3 * x_diffs),
        "epsilon_LLL_y_conf_low": -2 * stats["boot_LLL_y_conf_low"] / (3 * y_diffs),
        "epsilon_LLL_y_conf_high": -2 * stats["boot_LLL_y_conf_high"] / (3 * y_diffs),
        "epsilon_ASF_x_conf_low": -stats["boot_ASF_x_conf_low"] / 2,
        "epsilon_ASF_x_conf_high": -stats["boot_ASF_x_conf_high"] / 2,
        "epsilon_ASF_y_conf_low": -stats["boot_ASF_y_conf_low"] / 2,
        "epsilon_ASF_y_conf_high": -stats["boot_ASF_y_conf_high"] / 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs"),
    )
    parser.add_argument(
        "--mpi-only",
        action="store_true",
        help="Only compute and save the MPI-side arrays for notebook comparisons.",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, u, v = load_data()
    snapshot_keys = sorted(u.keys(), key=int)

    # The notebook launches this helper under mpirun because the notebook kernel
    # itself stays single-process. Rank 0 writes the MPI results back to disk so
    # the notebook can reload them as ordinary NumPy arrays.
    mpi_sfs_list = compute_sf_snapshots(
        u,
        v,
        x,
        y,
        snapshot_keys,
        backend="mpi",
        comm=comm,
    )
    participating_ranks = comm.gather(rank, root=0)

    if rank != 0:
        return

    print(f"MPI ranks participating: {participating_ranks}")

    mpi_stats = bootstrap_sf_statistics(mpi_sfs_list)
    x_diffs = mpi_sfs_list[0]["x-diffs"]
    y_diffs = mpi_sfs_list[0]["y-diffs"]
    mpi_cascade = estimate_cascade_rates(mpi_stats, x_diffs, y_diffs)

    if args.mpi_only:
        np.savez(
            output_dir / "ex_cascade_serial_vs_mpi_sample_data.npz",
            x_diffs=x_diffs,
            y_diffs=y_diffs,
            **{f"mpi_{key}": mpi_stats[key] for key in mpi_stats},
            **{f"mpi_{key}": mpi_cascade[key] for key in mpi_cascade},
        )
        print(f"Saved MPI-only outputs to {output_dir}")
        return

    serial_sfs_list = compute_sf_snapshots(
        u,
        v,
        x,
        y,
        snapshot_keys,
        backend="serial",
    )

    serial_stats = bootstrap_sf_statistics(serial_sfs_list)
    serial_cascade = estimate_cascade_rates(serial_stats, x_diffs, y_diffs)

    comparison_keys = sorted(
        {
            *serial_stats.keys(),
            *serial_cascade.keys(),
        }
    )
    diffs = {}
    for key in comparison_keys:
        serial_values = serial_stats.get(key, serial_cascade.get(key))
        mpi_values = mpi_stats.get(key, mpi_cascade.get(key))
        diffs[key] = max_abs_diff_ignore_nonfinite(serial_values, mpi_values)
        print(f"{key}: max abs diff = {diffs[key]:.3e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    ax2.semilogx(
        x_diffs,
        serial_cascade["epsilon_LLL_x_mean"],
        label=r"LLL$_x$ serial",
        color="tab:blue",
    )
    ax2.semilogx(
        y_diffs,
        serial_cascade["epsilon_LLL_y_mean"],
        label=r"LLL$_y$ serial",
        color="tab:red",
        linestyle="dashed",
    )
    ax2.semilogx(
        x_diffs,
        mpi_cascade["epsilon_LLL_x_mean"],
        label=r"LLL$_x$ mpi",
        color="tab:blue",
        linestyle=":",
    )
    ax2.semilogx(
        y_diffs,
        mpi_cascade["epsilon_LLL_y_mean"],
        label=r"LLL$_y$ mpi",
        color="tab:red",
        linestyle="-.",
    )

    ax1.semilogx(
        x_diffs,
        serial_cascade["epsilon_ASF_x_mean"],
        label=r"x-dir serial",
        color="tab:blue",
    )
    ax1.semilogx(
        y_diffs,
        serial_cascade["epsilon_ASF_y_mean"],
        label=r"y-dir serial",
        color="tab:red",
        linestyle="dashed",
    )
    ax1.semilogx(
        x_diffs,
        mpi_cascade["epsilon_ASF_x_mean"],
        label=r"x-dir mpi",
        color="tab:blue",
        linestyle=":",
    )
    ax1.semilogx(
        y_diffs,
        mpi_cascade["epsilon_ASF_y_mean"],
        label=r"y-dir mpi",
        color="tab:red",
        linestyle="-.",
    )

    ax1.fill_between(
        x_diffs,
        serial_cascade["epsilon_ASF_x_conf_low"],
        serial_cascade["epsilon_ASF_x_conf_high"],
        color="tab:blue",
        alpha=0.25,
        edgecolor=None,
    )
    ax1.fill_between(
        y_diffs,
        serial_cascade["epsilon_ASF_y_conf_low"],
        serial_cascade["epsilon_ASF_y_conf_high"],
        color="tab:red",
        alpha=0.25,
        edgecolor=None,
    )
    ax2.fill_between(
        x_diffs,
        serial_cascade["epsilon_LLL_x_conf_low"],
        serial_cascade["epsilon_LLL_x_conf_high"],
        color="tab:blue",
        alpha=0.25,
        edgecolor=None,
    )
    ax2.fill_between(
        y_diffs,
        serial_cascade["epsilon_LLL_y_conf_low"],
        serial_cascade["epsilon_LLL_y_conf_high"],
        color="tab:red",
        alpha=0.25,
        edgecolor=None,
    )

    ax1.set_ylabel(r"Inverse energy cascade rate $\epsilon$")
    ax1.set_xlabel(r"Separation distance")
    ax2.set_xlabel(r"Separation distance")
    ax1.set_xlim(3e-2, 3e0)
    ax2.set_xlim(3e-2, 3e0)
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    ax1.hlines(0, 3e-2, 3e0, color="k", lw=1, zorder=0)
    ax2.hlines(0, 3e-2, 3e0, color="k", lw=1, zorder=0)
    fig.tight_layout()
    fig.savefig(output_dir / "ex_cascade_serial_vs_mpi_sample_data.png", dpi=150)
    plt.close(fig)

    np.savez(
        output_dir / "ex_cascade_serial_vs_mpi_sample_data.npz",
        x_diffs=x_diffs,
        y_diffs=y_diffs,
        **{f"serial_{key}": serial_stats[key] for key in serial_stats},
        **{f"mpi_{key}": mpi_stats[key] for key in mpi_stats},
        **{f"serial_{key}": serial_cascade[key] for key in serial_cascade},
        **{f"mpi_{key}": mpi_cascade[key] for key in mpi_cascade},
    )
    print(f"Saved outputs to {output_dir}")
    print(f"Largest diff = {max(diffs.values()):.3e}")


if __name__ == "__main__":
    main()
