"""MPI runner for the Phase 2 z-slab reduction primitive."""

from __future__ import annotations

import argparse

import numpy as np

from fluidsf.mpi.reducers_3d import (
    compute_velocity_sf_reduction_3d,
    finalize_velocity_sf_reduction,
)
from fluidsf.mpi.slab_decomp_3d import (
    compute_velocity_sf_reduction_3d_periodic_z_slab_mpi,
    extract_local_z_slab,
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

    x = np.arange(args.nside, dtype=float)
    y = np.arange(args.nside, dtype=float)
    z = np.arange(args.nside, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]

    u_local = extract_local_z_slab(u, size, rank)
    v_local = extract_local_z_slab(v, size, rank)
    w_local = extract_local_z_slab(w, size, rank)

    shifts = [(1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]
    sf_type = ("LL", "TT", "LLL", "LTT")
    out = {}

    for idx, shift in enumerate(shifts):
        slab_sf = compute_velocity_sf_reduction_3d_periodic_z_slab_mpi(
            u_local,
            v_local,
            w_local,
            *shift,
            sf_type,
            comm=comm,
        )
        if rank == 0:
            full_reduction = compute_velocity_sf_reduction_3d(
                u,
                v,
                w,
                *shift,
                sf_type,
                boundary="periodic-all",
            )
            full_sf = finalize_velocity_sf_reduction(full_reduction, sf_type)
            for name in sf_type:
                out[f"slab_{idx}_{name.lower()}"] = slab_sf[f"SF_{name}"]
                out[f"full_{idx}_{name.lower()}"] = full_sf[f"SF_{name}"]

    if rank == 0:
        np.savez(args.output, **out)


if __name__ == "__main__":
    main()
