"""MPI runner for the Phase 2 z-slab 3D SF-grid generator."""

from __future__ import annotations

import argparse

import numpy as np

from fluidsf.mpi import (
    extract_local_z_slab,
    generate_sf_grid_3d_periodic_z_slab_mpi,
)
from fluidsf.mpi.generate_sf_3d_mpi import _compute_velocity_structure_functions


def _build_reference_grid(u, v, w, x, y, z):
    nx_half = len(x) // 2
    ny_half = len(y) // 2
    nz_half = len(z) // 2
    output = {
        "x-diffs": x[:nx_half] - float(x[0]),
        "y-diffs": y[:ny_half] - float(y[0]),
        "z-diffs": z[:nz_half] - float(z[0]),
        "SF_LL_grid": np.zeros((nx_half, ny_half, nz_half), dtype=np.float64),
        "SF_TT_grid": np.zeros((nx_half, ny_half, nz_half), dtype=np.float64),
        "SF_LLL_grid": np.zeros((nx_half, ny_half, nz_half), dtype=np.float64),
        "SF_LTT_grid": np.zeros((nx_half, ny_half, nz_half), dtype=np.float64),
    }

    for shift_x in range(nx_half):
        for shift_y in range(ny_half):
            for shift_z in range(nz_half):
                if shift_x == 0 and shift_y == 0 and shift_z == 0:
                    continue
                reduced = _compute_velocity_structure_functions(
                    u,
                    v,
                    w,
                    shift_x,
                    shift_y,
                    shift_z,
                    ("LL", "TT", "LLL", "LTT"),
                    boundary="periodic-all",
                )
                output["SF_LL_grid"][shift_x, shift_y, shift_z] = reduced["SF_LL"]
                output["SF_TT_grid"][shift_x, shift_y, shift_z] = reduced["SF_TT"]
                output["SF_LLL_grid"][shift_x, shift_y, shift_z] = reduced["SF_LLL"]
                output["SF_LTT_grid"][shift_x, shift_y, shift_z] = reduced["SF_LTT"]

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--field", choices=("linear", "asymmetric"), default="linear")
    args = parser.parse_args()

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nx = args.nside if args.nx is None else args.nx
    ny = args.nside if args.ny is None else args.ny
    nz = args.nside if args.nz is None else args.nz
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    z = np.arange(nz, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    if args.field == "asymmetric":
        u = 2.0 * xx + 0.1 * yy
        v = 3.0 * yy + 0.2 * zz
        w = 5.0 * zz + 0.3 * xx
    else:
        u = xx
        v = yy
        w = zz

    u_local = extract_local_z_slab(u, size, rank)
    v_local = extract_local_z_slab(v, size, rank)
    w_local = extract_local_z_slab(w, size, rank)

    slab_sf = generate_sf_grid_3d_periodic_z_slab_mpi(
        u_local,
        v_local,
        w_local,
        x,
        y,
        z,
        sf_type=("LL", "TT", "LLL", "LTT"),
        comm=comm,
    )

    ref_sf = None
    if rank == 0:
        ref_sf = _build_reference_grid(u, v, w, x, y, z)

    if rank == 0:
        np.savez(
            args.output,
            x_diffs=slab_sf["x-diffs"],
            y_diffs=slab_sf["y-diffs"],
            z_diffs=slab_sf["z-diffs"],
            slab_ll=slab_sf["SF_LL_grid"],
            slab_tt=slab_sf["SF_TT_grid"],
            slab_lll=slab_sf["SF_LLL_grid"],
            slab_ltt=slab_sf["SF_LTT_grid"],
            ref_ll=ref_sf["SF_LL_grid"],
            ref_tt=ref_sf["SF_TT_grid"],
            ref_lll=ref_sf["SF_LLL_grid"],
            ref_ltt=ref_sf["SF_LTT_grid"],
        )


if __name__ == "__main__":
    main()
