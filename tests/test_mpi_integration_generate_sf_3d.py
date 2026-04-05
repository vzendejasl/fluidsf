import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fluidsf import generate_structure_functions_3d


RUN_MPI_TESTS = os.environ.get("FLUIDSF_RUN_MPI_TESTS") == "1"


def _rank_config(ranks):
    if ranks <= 2:
        return 4, 1
    if ranks == 4:
        return 4, 2
    return 8, 2


def _run_mpi_case(tmp_path, mode, ranks, shape=None, field="linear"):
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun is not available in PATH.")

    nside, px = _rank_config(ranks)
    output_file = tmp_path / f"mpi_{mode}_{ranks}.npz"
    runner = Path(__file__).with_name("mpi_generate_sf_3d_runner.py")
    env = os.environ.copy()
    env["HYDRA_LAUNCHER"] = "fork"

    nx, ny, nz = (None, None, None) if shape is None else shape
    completed = subprocess.run(
        [
            mpirun,
            "-launcher",
            "fork",
            "-n",
            str(ranks),
            sys.executable,
            str(runner),
            "--mode",
            mode,
            "--nside",
            str(nside),
            "--px",
            str(px),
            "--output",
            str(output_file),
            "--field",
            field,
            *([] if nx is None else ["--nx", str(nx), "--ny", str(ny), "--nz", str(nz)]),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(
            f"{ranks}-rank MPI integration test for mode={mode} failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if not output_file.exists():
        raise AssertionError("MPI runner did not write the expected output file.")

    return np.load(output_file), (nside, nside, nside) if shape is None else shape


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_sf_grid_3d_mpi_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(tmp_path, "grid", ranks)
    nside = shape[0]

    x_diffs = data["x_diffs"]
    y_diffs = data["y_diffs"]
    z_diffs = data["z_diffs"]
    sf_ll = data["sf_ll_grid"]
    sf_tt = data["sf_tt_grid"]
    sf_lll = data["sf_lll_grid"]
    sf_ltt = data["sf_ltt_grid"]

    nx_half = nside // 2
    np.testing.assert_allclose(x_diffs, np.arange(nx_half, dtype=float))
    np.testing.assert_allclose(y_diffs, np.arange(nx_half, dtype=float))
    np.testing.assert_allclose(z_diffs, np.arange(nx_half, dtype=float))

    expected_ll = np.zeros_like(sf_ll)
    expected_tt = np.zeros_like(sf_tt)
    expected_lll = np.zeros_like(sf_lll)
    expected_ltt = np.zeros_like(sf_ltt)

    for ix in range(sf_ll.shape[0]):
        for iy in range(sf_ll.shape[1]):
            for iz in range(sf_ll.shape[2]):
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                radius = np.sqrt(float(ix * ix + iy * iy + iz * iz))
                expected_ll[ix, iy, iz] = radius**2
                expected_lll[ix, iy, iz] = radius**3

    np.testing.assert_allclose(sf_ll, expected_ll)
    np.testing.assert_allclose(sf_tt, expected_tt, atol=1e-14)
    np.testing.assert_allclose(sf_lll, expected_lll)
    np.testing.assert_allclose(sf_ltt, expected_ltt, atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_3d_public_mpi_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(tmp_path, "public", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    z = np.arange(nside, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")
    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LL", "TT", "LLL", "LTT"],
        boundary="periodic-all",
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["z_diffs"], serial["z-diffs"])
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_z"], serial["SF_LL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_z"], serial["SF_TT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_z"], serial["SF_LLL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_z"], serial["SF_LTT_z"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_3d_public_mpi_periodic_all_noncubic_real_ranks(
    tmp_path, ranks
):
    data, shape = _run_mpi_case(tmp_path, "public-full", ranks, shape=(8, 6, 4))

    x = np.arange(shape[0], dtype=float)
    y = np.arange(shape[1], dtype=float)
    z = np.arange(shape[2], dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")
    scalar = u + 2.0 * v
    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["z_diffs"], serial["z-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_z"], serial["SF_advection_velocity_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_z"], serial["SF_advection_scalar_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_z"], serial["SF_LL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_z"], serial["SF_TT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_z"], serial["SF_SS_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_z"], serial["SF_LLL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_z"], serial["SF_LTT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_z"], serial["SF_LSS_z"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_3d_public_mpi_asymmetric_noncubic_real_ranks(
    tmp_path, ranks
):
    data, shape = _run_mpi_case(
        tmp_path,
        "public-full",
        ranks,
        shape=(8, 6, 4),
        field="asymmetric",
    )

    x = np.arange(shape[0], dtype=float)
    y = np.arange(shape[1], dtype=float)
    z = np.arange(shape[2], dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u = 2.0 * xx + 0.1 * yy
    v = 3.0 * yy + 0.2 * zz
    w = 5.0 * zz + 0.3 * xx
    scalar = 7.0 * xx + 11.0 * zz
    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["z_diffs"], serial["z-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_z"], serial["SF_advection_velocity_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_z"], serial["SF_advection_scalar_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_z"], serial["SF_LL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_z"], serial["SF_TT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_z"], serial["SF_SS_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_z"], serial["SF_LLL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_z"], serial["SF_LTT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_z"], serial["SF_LSS_z"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_3d_public_mpi_asymmetric_periodic_x_real_ranks(
    tmp_path, ranks
):
    data, shape = _run_mpi_case(
        tmp_path,
        "public-periodic-x-full",
        ranks,
        shape=(8, 6, 4),
        field="asymmetric",
    )

    x = np.arange(shape[0], dtype=float)
    y = np.arange(shape[1], dtype=float)
    z = np.arange(shape[2], dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u = 2.0 * xx + 0.1 * yy
    v = 3.0 * yy + 0.2 * zz
    w = 5.0 * zz + 0.3 * xx
    scalar = 7.0 * xx + 11.0 * zz
    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary="periodic-x",
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["z_diffs"], serial["z-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_z"], serial["SF_advection_velocity_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_z"], serial["SF_advection_scalar_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_z"], serial["SF_LL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_z"], serial["SF_TT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_z"], serial["SF_SS_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_z"], serial["SF_LLL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_z"], serial["SF_LTT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_z"], serial["SF_LSS_z"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_3d_public_mpi_scalar_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(tmp_path, "public-scalar", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    z = np.arange(nside, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")
    scalar = u + 2.0 * v
    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["SS", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["z_diffs"], serial["z-diffs"])
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_z"], serial["SF_SS_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_z"], serial["SF_LSS_z"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_3d_public_mpi_advective_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(tmp_path, "public-adv", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    z = np.arange(nside, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")
    scalar = u + 2.0 * v
    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S"],
        scalar=scalar,
        boundary="periodic-all",
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["z_diffs"], serial["z-diffs"])
    np.testing.assert_allclose(
        data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13
    )
    np.testing.assert_allclose(
        data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13
    )
    np.testing.assert_allclose(
        data["sf_adv_v_z"], serial["SF_advection_velocity_z"], atol=1e-13
    )
    np.testing.assert_allclose(
        data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13
    )
    np.testing.assert_allclose(
        data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13
    )
    np.testing.assert_allclose(
        data["sf_adv_s_z"], serial["SF_advection_scalar_z"], atol=1e-13
    )


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_3d_public_mpi_binned_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(tmp_path, "public-binned", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    z = np.arange(nside, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")
    scalar = u + 2.0 * v
    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "SS", "LLL", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["z_diffs"], serial["z-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_z"], serial["SF_advection_velocity_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_z"], serial["SF_advection_scalar_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_z"], serial["SF_LL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_z"], serial["SF_SS_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_z"], serial["SF_LLL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_z"], serial["SF_LSS_z"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_3d_public_mpi_nonperiodic_binned_real_ranks(
    tmp_path, ranks
):
    data, shape = _run_mpi_case(tmp_path, "public-nonperiodic-binned", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    z = np.arange(nside, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")
    scalar = u + 2.0 * v
    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary=None,
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["z_diffs"], serial["z-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_z"], serial["SF_advection_velocity_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_z"], serial["SF_advection_scalar_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_z"], serial["SF_LL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_z"], serial["SF_TT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_z"], serial["SF_SS_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_z"], serial["SF_LLL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_z"], serial["SF_LTT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_z"], serial["SF_LSS_z"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_3d_public_mpi_mixed_binned_real_ranks(
    tmp_path, ranks
):
    data, shape = _run_mpi_case(tmp_path, "public-mixed-binned", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    z = np.arange(nside, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")
    scalar = u + 2.0 * v
    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary=["periodic-x", "periodic-y"],
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["z_diffs"], serial["z-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_z"], serial["SF_advection_velocity_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_z"], serial["SF_advection_scalar_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_z"], serial["SF_LL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_z"], serial["SF_TT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_z"], serial["SF_SS_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_z"], serial["SF_LLL_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_z"], serial["SF_LTT_z"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_z"], serial["SF_LSS_z"], atol=1e-13)
