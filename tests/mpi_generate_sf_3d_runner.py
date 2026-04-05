"""Small MPI runner used by pytest integration tests."""

from __future__ import annotations

import argparse

import numpy as np

from fluidsf import generate_structure_functions_3d
from fluidsf.mpi import compute_slab_bounds_1d, generate_sf_grid_3d_mpi


def _assert_rank_consistent(comm, payload):
    gathered = comm.gather(payload, root=0)
    if comm.Get_rank() == 0:
        reference = gathered[0]
        for other in gathered[1:]:
            if reference.keys() != other.keys():
                raise AssertionError("MPI ranks produced different output keys.")
            for key in reference:
                np.testing.assert_allclose(other[key], reference[key], atol=1e-13)


def _save_payload_root_only(comm, output, payload):
    if comm.Get_rank() == 0:
        np.savez(output, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "grid",
            "public",
            "public-scalar",
            "public-adv",
            "public-full",
            "public-periodic-x-full",
            "public-binned",
            "public-nonperiodic-binned",
            "public-mixed-binned",
        ),
        default="grid",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--nside", type=int, default=4)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--field", choices=("linear", "asymmetric"), default="linear")
    parser.add_argument("--px", type=int, default=1)
    args = parser.parse_args()

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
        scalar = 7.0 * xx + 11.0 * zz
    else:
        u = xx
        v = yy
        w = zz
        scalar = xx + 2.0 * yy

    if args.mode == "grid":
        sf = generate_sf_grid_3d_mpi(
            u,
            v,
            w,
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
            u[start:stop, :, :],
            v[start:stop, :, :],
            w[start:stop, :, :],
            x,
            y,
            z,
            sf_type=["LL", "TT", "LLL", "LTT"],
            boundary="periodic-all",
            backend="mpi",
            px=args.px,
            comm=comm,
        )

        payload = {
            "x_diffs": sf["x-diffs"],
            "y_diffs": sf["y-diffs"],
            "z_diffs": sf["z-diffs"],
            "sf_ll_x": sf["SF_LL_x"],
            "sf_ll_y": sf["SF_LL_y"],
            "sf_ll_z": sf["SF_LL_z"],
            "sf_tt_x": sf["SF_TT_x"],
            "sf_tt_y": sf["SF_TT_y"],
            "sf_tt_z": sf["SF_TT_z"],
            "sf_lll_x": sf["SF_LLL_x"],
            "sf_lll_y": sf["SF_LLL_y"],
            "sf_lll_z": sf["SF_LLL_z"],
            "sf_ltt_x": sf["SF_LTT_x"],
            "sf_ltt_y": sf["SF_LTT_y"],
            "sf_ltt_z": sf["SF_LTT_z"],
        }
        _assert_rank_consistent(comm, payload)
        _save_payload_root_only(comm, args.output, payload)
        return

    if args.mode == "public-scalar":
        sf = generate_structure_functions_3d(
            u[start:stop, :, :],
            v[start:stop, :, :],
            w[start:stop, :, :],
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

        payload = {
            "x_diffs": sf["x-diffs"],
            "y_diffs": sf["y-diffs"],
            "z_diffs": sf["z-diffs"],
            "sf_ss_x": sf["SF_SS_x"],
            "sf_ss_y": sf["SF_SS_y"],
            "sf_ss_z": sf["SF_SS_z"],
            "sf_lss_x": sf["SF_LSS_x"],
            "sf_lss_y": sf["SF_LSS_y"],
            "sf_lss_z": sf["SF_LSS_z"],
        }
        _assert_rank_consistent(comm, payload)
        _save_payload_root_only(comm, args.output, payload)
        return

    if args.mode == "public-adv":
        sf = generate_structure_functions_3d(
            u[start:stop, :, :],
            v[start:stop, :, :],
            w[start:stop, :, :],
            x,
            y,
            z,
            sf_type=["ASF_V", "ASF_S"],
            scalar=scalar[start:stop, :, :],
            boundary="periodic-all",
            backend="mpi",
            px=args.px,
            comm=comm,
        )

        payload = {
            "x_diffs": sf["x-diffs"],
            "y_diffs": sf["y-diffs"],
            "z_diffs": sf["z-diffs"],
            "sf_adv_v_x": sf["SF_advection_velocity_x"],
            "sf_adv_v_y": sf["SF_advection_velocity_y"],
            "sf_adv_v_z": sf["SF_advection_velocity_z"],
            "sf_adv_s_x": sf["SF_advection_scalar_x"],
            "sf_adv_s_y": sf["SF_advection_scalar_y"],
            "sf_adv_s_z": sf["SF_advection_scalar_z"],
        }
        _assert_rank_consistent(comm, payload)
        _save_payload_root_only(comm, args.output, payload)
        return

    if args.mode == "public-full":
        sf = generate_structure_functions_3d(
            u[start:stop, :, :],
            v[start:stop, :, :],
            w[start:stop, :, :],
            x,
            y,
            z,
            sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
            scalar=scalar[start:stop, :, :],
            boundary="periodic-all",
            backend="mpi",
            px=args.px,
            comm=comm,
        )

        payload = {
            "x_diffs": sf["x-diffs"],
            "y_diffs": sf["y-diffs"],
            "z_diffs": sf["z-diffs"],
            "sf_adv_v_x": sf["SF_advection_velocity_x"],
            "sf_adv_v_y": sf["SF_advection_velocity_y"],
            "sf_adv_v_z": sf["SF_advection_velocity_z"],
            "sf_adv_s_x": sf["SF_advection_scalar_x"],
            "sf_adv_s_y": sf["SF_advection_scalar_y"],
            "sf_adv_s_z": sf["SF_advection_scalar_z"],
            "sf_ll_x": sf["SF_LL_x"],
            "sf_ll_y": sf["SF_LL_y"],
            "sf_ll_z": sf["SF_LL_z"],
            "sf_tt_x": sf["SF_TT_x"],
            "sf_tt_y": sf["SF_TT_y"],
            "sf_tt_z": sf["SF_TT_z"],
            "sf_ss_x": sf["SF_SS_x"],
            "sf_ss_y": sf["SF_SS_y"],
            "sf_ss_z": sf["SF_SS_z"],
            "sf_lll_x": sf["SF_LLL_x"],
            "sf_lll_y": sf["SF_LLL_y"],
            "sf_lll_z": sf["SF_LLL_z"],
            "sf_ltt_x": sf["SF_LTT_x"],
            "sf_ltt_y": sf["SF_LTT_y"],
            "sf_ltt_z": sf["SF_LTT_z"],
            "sf_lss_x": sf["SF_LSS_x"],
            "sf_lss_y": sf["SF_LSS_y"],
            "sf_lss_z": sf["SF_LSS_z"],
        }
        _assert_rank_consistent(comm, payload)
        _save_payload_root_only(comm, args.output, payload)
        return

    if args.mode == "public-periodic-x-full":
        sf = generate_structure_functions_3d(
            u[start:stop, :, :],
            v[start:stop, :, :],
            w[start:stop, :, :],
            x,
            y,
            z,
            sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
            scalar=scalar[start:stop, :, :],
            boundary="periodic-x",
            backend="mpi",
            px=args.px,
            comm=comm,
        )

        payload = {
            "x_diffs": sf["x-diffs"],
            "y_diffs": sf["y-diffs"],
            "z_diffs": sf["z-diffs"],
            "sf_adv_v_x": sf["SF_advection_velocity_x"],
            "sf_adv_v_y": sf["SF_advection_velocity_y"],
            "sf_adv_v_z": sf["SF_advection_velocity_z"],
            "sf_adv_s_x": sf["SF_advection_scalar_x"],
            "sf_adv_s_y": sf["SF_advection_scalar_y"],
            "sf_adv_s_z": sf["SF_advection_scalar_z"],
            "sf_ll_x": sf["SF_LL_x"],
            "sf_ll_y": sf["SF_LL_y"],
            "sf_ll_z": sf["SF_LL_z"],
            "sf_tt_x": sf["SF_TT_x"],
            "sf_tt_y": sf["SF_TT_y"],
            "sf_tt_z": sf["SF_TT_z"],
            "sf_ss_x": sf["SF_SS_x"],
            "sf_ss_y": sf["SF_SS_y"],
            "sf_ss_z": sf["SF_SS_z"],
            "sf_lll_x": sf["SF_LLL_x"],
            "sf_lll_y": sf["SF_LLL_y"],
            "sf_lll_z": sf["SF_LLL_z"],
            "sf_ltt_x": sf["SF_LTT_x"],
            "sf_ltt_y": sf["SF_LTT_y"],
            "sf_ltt_z": sf["SF_LTT_z"],
            "sf_lss_x": sf["SF_LSS_x"],
            "sf_lss_y": sf["SF_LSS_y"],
            "sf_lss_z": sf["SF_LSS_z"],
        }
        _assert_rank_consistent(comm, payload)
        _save_payload_root_only(comm, args.output, payload)
        return

    if args.mode == "public-binned":
        sf = generate_structure_functions_3d(
            u[start:stop, :, :],
            v[start:stop, :, :],
            w[start:stop, :, :],
            x,
            y,
            z,
            sf_type=["ASF_V", "ASF_S", "LL", "SS", "LLL", "LSS"],
            scalar=scalar[start:stop, :, :],
            boundary="periodic-all",
            nbins=2,
            backend="mpi",
            px=args.px,
            comm=comm,
        )
    elif args.mode == "public-mixed-binned":
        sf = generate_structure_functions_3d(
            u[start:stop, :, :],
            v[start:stop, :, :],
            w[start:stop, :, :],
            x,
            y,
            z,
            sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
            scalar=scalar[start:stop, :, :],
            boundary=["periodic-x", "periodic-y"],
            nbins=2,
            backend="mpi",
            px=args.px,
            comm=comm,
        )
    else:
        sf = generate_structure_functions_3d(
            u[start:stop, :, :],
            v[start:stop, :, :],
            w[start:stop, :, :],
            x,
            y,
            z,
            sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
            scalar=scalar[start:stop, :, :],
            boundary=None,
            nbins=2,
            backend="mpi",
            px=args.px,
            comm=comm,
        )

    payload = {
        "x_diffs": sf["x-diffs"],
        "y_diffs": sf["y-diffs"],
        "z_diffs": sf["z-diffs"],
        "sf_adv_v_x": sf["SF_advection_velocity_x"],
        "sf_adv_v_y": sf["SF_advection_velocity_y"],
        "sf_adv_v_z": sf["SF_advection_velocity_z"],
        "sf_adv_s_x": sf["SF_advection_scalar_x"],
        "sf_adv_s_y": sf["SF_advection_scalar_y"],
        "sf_adv_s_z": sf["SF_advection_scalar_z"],
        "sf_ll_x": sf["SF_LL_x"],
        "sf_ll_y": sf["SF_LL_y"],
        "sf_ll_z": sf["SF_LL_z"],
        "sf_ss_x": sf["SF_SS_x"],
        "sf_ss_y": sf["SF_SS_y"],
        "sf_ss_z": sf["SF_SS_z"],
        "sf_lll_x": sf["SF_LLL_x"],
        "sf_lll_y": sf["SF_LLL_y"],
        "sf_lll_z": sf["SF_LLL_z"],
        "sf_lss_x": sf["SF_LSS_x"],
        "sf_lss_y": sf["SF_LSS_y"],
        "sf_lss_z": sf["SF_LSS_z"],
    }
    if "SF_TT_x" in sf:
        payload["sf_tt_x"] = sf["SF_TT_x"]
        payload["sf_tt_y"] = sf["SF_TT_y"]
        payload["sf_tt_z"] = sf["SF_TT_z"]
    if "SF_LTT_x" in sf:
        payload["sf_ltt_x"] = sf["SF_LTT_x"]
        payload["sf_ltt_y"] = sf["SF_LTT_y"]
        payload["sf_ltt_z"] = sf["SF_LTT_z"]
    _assert_rank_consistent(comm, payload)
    _save_payload_root_only(comm, args.output, payload)


if __name__ == "__main__":
    main()
