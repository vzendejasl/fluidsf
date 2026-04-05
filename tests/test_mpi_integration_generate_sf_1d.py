import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fluidsf import generate_structure_functions_1d


RUN_MPI_TESTS = os.environ.get("FLUIDSF_RUN_MPI_TESTS") == "1"


def _run_mpi_case(tmp_path, mode, ranks, nside=8):
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun is not available in PATH.")

    output_file = tmp_path / f"mpi_1d_{mode}_{ranks}.npz"
    runner = Path(__file__).with_name("mpi_generate_sf_1d_runner.py")
    env = os.environ.copy()
    env["HYDRA_LAUNCHER"] = "fork"

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
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(
            f"{ranks}-rank MPI integration test for 1D mode={mode} failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if not output_file.exists():
        raise AssertionError("MPI runner did not write the expected 1D output file.")

    return np.load(output_file), nside


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_1d_public_mpi_nonperiodic_real_ranks(tmp_path, ranks):
    data, nside = _run_mpi_case(tmp_path, "nonperiodic", ranks)

    x = np.arange(nside, dtype=float)
    u = x.copy()
    v = 2.0 * x
    scalar = 3.0 * x
    serial = generate_structure_functions_1d(
        u,
        x,
        sf_type=["LL", "TT", "SS", "LLL", "LTT", "LSS"],
        v=v,
        scalar=scalar,
        boundary=None,
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["sf_ll"], serial["SF_LL"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt"], serial["SF_TT"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss"], serial["SF_SS"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll"], serial["SF_LLL"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt"], serial["SF_LTT"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss"], serial["SF_LSS"], atol=1e-13)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_structure_functions_1d_public_mpi_periodic_latlon_real_ranks(tmp_path, ranks):
    data, nside = _run_mpi_case(tmp_path, "periodic-latlon", ranks)

    x = np.arange(nside, dtype=float)
    u = x.copy()
    v = 2.0 * x
    scalar = 3.0 * x
    serial = generate_structure_functions_1d(
        u,
        x,
        sf_type=["LL", "TT", "SS", "LLL", "LTT", "LSS"],
        v=v,
        y=x,
        scalar=scalar,
        boundary="Periodic",
        grid_type="latlon",
        nbins=2,
    )

    np.testing.assert_allclose(data["x_diffs"], serial["x-diffs"])
    np.testing.assert_allclose(data["sf_ll"], serial["SF_LL"], atol=1e-13)
    np.testing.assert_allclose(data["sf_tt"], serial["SF_TT"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ss"], serial["SF_SS"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lll"], serial["SF_LLL"], atol=1e-13)
    np.testing.assert_allclose(data["sf_ltt"], serial["SF_LTT"], atol=1e-13)
    np.testing.assert_allclose(data["sf_lss"], serial["SF_LSS"], atol=1e-13)
