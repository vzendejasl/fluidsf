import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


RUN_MPI_TESTS = os.environ.get("FLUIDSF_RUN_MPI_TESTS") == "1"


def _nside_for_ranks(ranks):
    return max(8, ranks)


@pytest.mark.skipif(not RUN_MPI_TESTS, reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.")
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
def test_generate_sf_grid_3d_periodic_z_slab_mpi_matches_reference_grid(tmp_path, ranks):
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun is not available in PATH.")

    output_file = tmp_path / f"slab_grid_{ranks}.npz"
    runner = Path(__file__).with_name("mpi_slab_grid_3d_runner.py")
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
            "--nside",
            str(_nside_for_ranks(ranks)),
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
            f"{ranks}-rank slab-grid MPI integration test failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    data = np.load(output_file)
    np.testing.assert_allclose(data["x_diffs"], np.arange(_nside_for_ranks(ranks) // 2))
    np.testing.assert_allclose(data["y_diffs"], np.arange(_nside_for_ranks(ranks) // 2))
    np.testing.assert_allclose(data["z_diffs"], np.arange(_nside_for_ranks(ranks) // 2))
    np.testing.assert_allclose(data["slab_ll"], data["ref_ll"], atol=1e-13)
    np.testing.assert_allclose(data["slab_tt"], data["ref_tt"], atol=1e-13)
    np.testing.assert_allclose(data["slab_lll"], data["ref_lll"], atol=1e-13)
    np.testing.assert_allclose(data["slab_ltt"], data["ref_ltt"], atol=1e-13)
