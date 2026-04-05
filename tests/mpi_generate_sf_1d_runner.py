"""Small MPI runner used by pytest integration tests for 1D."""

from __future__ import annotations

import argparse

import numpy as np

from fluidsf import generate_structure_functions_1d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("nonperiodic", "periodic-latlon"), default="nonperiodic")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nside", type=int, default=8)
    args = parser.parse_args()

    x = np.arange(args.nside, dtype=float)
    u = x.copy()
    v = 2.0 * x
    scalar = 3.0 * x

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if args.mode == "nonperiodic":
        sf = generate_structure_functions_1d(
            u,
            x,
            sf_type=["LL", "TT", "SS", "LLL", "LTT", "LSS"],
            v=v,
            scalar=scalar,
            boundary=None,
            nbins=2,
            backend="mpi",
            comm=comm,
        )
    else:
        sf = generate_structure_functions_1d(
            u,
            x,
            sf_type=["LL", "TT", "SS", "LLL", "LTT", "LSS"],
            v=v,
            y=x,
            scalar=scalar,
            boundary="Periodic",
            grid_type="latlon",
            nbins=2,
            backend="mpi",
            comm=comm,
        )

    if comm.Get_rank() == 0:
        np.savez(
            args.output,
            x_diffs=sf["x-diffs"],
            sf_ll=sf["SF_LL"],
            sf_tt=sf["SF_TT"],
            sf_ss=sf["SF_SS"],
            sf_lll=sf["SF_LLL"],
            sf_ltt=sf["SF_LTT"],
            sf_lss=sf["SF_LSS"],
        )


if __name__ == "__main__":
    main()
