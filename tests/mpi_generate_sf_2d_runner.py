"""Small MPI runner used by pytest integration tests for 2D."""

from __future__ import annotations

import argparse

import numpy as np

from fluidsf import generate_structure_functions_2d
from fluidsf.mpi.slab_decomp_2d import compute_slab_bounds_1d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("nonperiodic", "periodic-x", "periodic-y", "periodic-all"),
        default="nonperiodic",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--field", choices=("linear", "asymmetric"), default="linear")
    args = parser.parse_args()

    nx = args.nside if args.nx is None else args.nx
    ny = args.nside if args.ny is None else args.ny
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    if args.field == "asymmetric":
        u = 2.0 * xx + 0.3 * yy
        v = 5.0 * yy + 0.4 * xx
        scalar = 7.0 * xx + 11.0 * yy
    else:
        u = xx
        v = 0.5 * xx
        scalar = 0.25 * xx

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    start, stop = compute_slab_bounds_1d(len(x), comm.Get_size(), comm.Get_rank())
    u_local = u[:, start:stop]
    v_local = v[:, start:stop]
    scalar_local = scalar[:, start:stop]
    if args.mode == "nonperiodic":
        sf = generate_structure_functions_2d(
            u_local,
            v_local,
            x,
            y,
            sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
            scalar=scalar_local,
            boundary=None,
            nbins=2,
            backend="mpi",
            comm=comm,
        )
    else:
        if args.mode == "periodic-x":
            boundary = "periodic-x"
        elif args.mode == "periodic-y":
            boundary = "periodic-y"
        else:
            boundary = "periodic-all"
        sf = generate_structure_functions_2d(
            u_local,
            v_local,
            x,
            y,
            sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
            scalar=scalar_local,
            boundary=boundary,
            nbins=2,
            backend="mpi",
            comm=comm,
        )

    if comm.Get_rank() == 0:
        np.savez(
            args.output,
            x_diffs=sf["x-diffs"],
            y_diffs=sf["y-diffs"],
            sf_adv_v_x=sf["SF_advection_velocity_x"],
            sf_adv_v_y=sf["SF_advection_velocity_y"],
            sf_adv_s_x=sf["SF_advection_scalar_x"],
            sf_adv_s_y=sf["SF_advection_scalar_y"],
            sf_ll_x=sf["SF_LL_x"],
            sf_ll_y=sf["SF_LL_y"],
            sf_tt_x=sf["SF_TT_x"],
            sf_tt_y=sf["SF_TT_y"],
            sf_ss_x=sf["SF_SS_x"],
            sf_ss_y=sf["SF_SS_y"],
            sf_lll_x=sf["SF_LLL_x"],
            sf_lll_y=sf["SF_LLL_y"],
            sf_ltt_x=sf["SF_LTT_x"],
            sf_ltt_y=sf["SF_LTT_y"],
            sf_lss_x=sf["SF_LSS_x"],
            sf_lss_y=sf["SF_LSS_y"],
        )


if __name__ == "__main__":
    main()
