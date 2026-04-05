import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fluidsf import generate_structure_functions_2d


RUN_MPI_TESTS = os.environ.get("FLUIDSF_RUN_MPI_TESTS") == "1"


def _run_mpi_case(tmp_path, mode, ranks, nside=8, shape=None, field="linear"):
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun is not available in PATH.")

    output_file = tmp_path / f"mpi_2d_{mode}_{ranks}.npz"
    runner = Path(__file__).with_name("mpi_generate_sf_2d_runner.py")
    env = os.environ.copy()
    env["HYDRA_LAUNCHER"] = "fork"

    nx, ny = (None, None) if shape is None else shape
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
            "--output",
            str(output_file),
            "--field",
            field,
            *([] if nx is None else ["--nx", str(nx), "--ny", str(ny)]),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(
            f"{ranks}-rank MPI integration test for 2D mode={mode} failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if not output_file.exists():
        raise AssertionError("MPI runner did not write the expected 2D output file.")

    return np.load(output_file), (nside, nside) if shape is None else shape


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_2d_public_mpi_nonperiodic_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(tmp_path, "nonperiodic", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    u = np.meshgrid(x, y)[0]
    v = 0.5 * np.meshgrid(x, y)[0]
    scalar = 0.25 * np.meshgrid(x, y)[0]
    serial = generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary=None,
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_2d_public_mpi_periodic_x_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(tmp_path, "periodic-x", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    u = np.meshgrid(x, y)[0]
    v = 0.5 * np.meshgrid(x, y)[0]
    scalar = 0.25 * np.meshgrid(x, y)[0]
    serial = generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary="periodic-x",
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_2d_public_mpi_periodic_y_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(tmp_path, "periodic-y", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    u = np.meshgrid(x, y)[0]
    v = 0.5 * np.meshgrid(x, y)[0]
    scalar = 0.25 * np.meshgrid(x, y)[0]
    serial = generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary="periodic-y",
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_2d_public_mpi_periodic_all_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(tmp_path, "periodic-all", ranks)
    nside = shape[0]

    x = np.arange(nside, dtype=float)
    y = np.arange(nside, dtype=float)
    u = np.meshgrid(x, y)[0]
    v = 0.5 * np.meshgrid(x, y)[0]
    scalar = 0.25 * np.meshgrid(x, y)[0]
    serial = generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_2d_public_mpi_asymmetric_noncubic_real_ranks(tmp_path, ranks):
    data, shape = _run_mpi_case(
        tmp_path,
        "periodic-all",
        ranks,
        shape=(10, 6),
        field="asymmetric",
    )

    x = np.arange(shape[0], dtype=float)
    y = np.arange(shape[1], dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    u = 2.0 * xx + 0.3 * yy
    v = 5.0 * yy + 0.4 * xx
    scalar = 7.0 * xx + 11.0 * yy
    serial = generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["y_diffs"], serial["y-diffs"])
    np.testing.assert_allclose(data["sf_adv_v_x"], serial["SF_advection_velocity_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_v_y"], serial["SF_advection_velocity_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_x"], serial["SF_advection_scalar_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_adv_s_y"], serial["SF_advection_scalar_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_x"], serial["SF_LL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ll_y"], serial["SF_LL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_x"], serial["SF_TT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt_y"], serial["SF_TT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_x"], serial["SF_SS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss_y"], serial["SF_SS_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_x"], serial["SF_LLL_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll_y"], serial["SF_LLL_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_x"], serial["SF_LTT_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt_y"], serial["SF_LTT_y"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_x"], serial["SF_LSS_x"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss_y"], serial["SF_LSS_y"], atol=1e-13)
