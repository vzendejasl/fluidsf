"""MPI runner for the Phase 2 z-slab 3D SF-grid generator."""

from __future__ import annotations

import argparse

import numpy as np

from fluidsf.mpi import (
    extract_local_z_slab,
    generate_sf_grid_3d_mpi,
    generate_sf_grid_3d_periodic_z_slab_mpi,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--nside", type=int, default=8)
    args = parser.parse_args()

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    px = 1 if size <= 2 else 2

    x = np.arange(args.nside, dtype=float)
    y = np.arange(args.nside, dtype=float)
    z = np.arange(args.nside, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]

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

    ref_sf = generate_sf_grid_3d_mpi(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LL", "TT", "LLL", "LTT"],
        px=px,
        boundary="periodic-all",
        comm=comm,
    )

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
