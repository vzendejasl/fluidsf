"""Small MPI runner used by pytest integration tests."""

from __future__ import annotations

import argparse

import numpy as np

from fluidsf import generate_structure_functions_3d
from fluidsf.mpi import compute_slab_bounds_1d, generate_sf_grid_3d_mpi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("grid", "public", "public-scalar"), default="grid"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--nside", type=int, default=4)
    parser.add_argument("--px", type=int, default=1)
    args = parser.parse_args()

    x = np.arange(args.nside, dtype=float)
    y = np.arange(args.nside, dtype=float)
    z = np.arange(args.nside, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    if args.mode == "grid":
        sf = generate_sf_grid_3d_mpi(
            xx,
            yy,
            zz,
            x,
            y,
            z,
            sf_type=["LL", "TT", "LLL", "LTT"],
            px=args.px,
            boundary=None,
        )

        if sf["x-diffs"] is not None:
            np.savez(
                args.output,
                x_diffs=sf["x-diffs"],
                y_diffs=sf["y-diffs"],
                z_diffs=sf["z-diffs"],
                sf_ll_grid=sf["SF_LL_grid"],
                sf_tt_grid=sf["SF_TT_grid"],
                sf_lll_grid=sf["SF_LLL_grid"],
                sf_ltt_grid=sf["SF_LTT_grid"],
            )
        return

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    start, stop = compute_slab_bounds_1d(len(x), comm.Get_size(), comm.Get_rank())

    if args.mode == "public":
        sf = generate_structure_functions_3d(
            xx[start:stop, :, :],
            yy[start:stop, :, :],
            zz[start:stop, :, :],
            x,
            y,
            z,
            sf_type=["LL", "TT", "LLL", "LTT"],
            boundary="periodic-all",
            backend="mpi",
            px=args.px,
            comm=comm,
        )

        if sf["x-diffs"] is not None:
            np.savez(
                args.output,
                x_diffs=sf["x-diffs"],
                y_diffs=sf["y-diffs"],
                z_diffs=sf["z-diffs"],
                sf_ll_x=sf["SF_LL_x"],
                sf_ll_y=sf["SF_LL_y"],
                sf_ll_z=sf["SF_LL_z"],
                sf_tt_x=sf["SF_TT_x"],
                sf_tt_y=sf["SF_TT_y"],
                sf_tt_z=sf["SF_TT_z"],
                sf_lll_x=sf["SF_LLL_x"],
                sf_lll_y=sf["SF_LLL_y"],
                sf_lll_z=sf["SF_LLL_z"],
                sf_ltt_x=sf["SF_LTT_x"],
                sf_ltt_y=sf["SF_LTT_y"],
                sf_ltt_z=sf["SF_LTT_z"],
            )
        return

    scalar = xx + 2.0 * yy
    sf = generate_structure_functions_3d(
        xx[start:stop, :, :],
        yy[start:stop, :, :],
        zz[start:stop, :, :],
        x,
        y,
        z,
        sf_type=["SS", "LSS"],
        scalar=scalar[start:stop, :, :],
        boundary="periodic-all",
        backend="mpi",
        px=args.px,
        comm=comm,
    )

    if sf["x-diffs"] is not None:
        np.savez(
            args.output,
            x_diffs=sf["x-diffs"],
            y_diffs=sf["y-diffs"],
            z_diffs=sf["z-diffs"],
            sf_ss_x=sf["SF_SS_x"],
            sf_ss_y=sf["SF_SS_y"],
            sf_ss_z=sf["SF_SS_z"],
            sf_lss_x=sf["SF_LSS_x"],
            sf_lss_y=sf["SF_LSS_y"],
            sf_lss_z=sf["SF_LSS_z"],
        )


if __name__ == "__main__":
    main()
