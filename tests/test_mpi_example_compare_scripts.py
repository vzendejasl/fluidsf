import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


RUN_MPI_TESTS = os.environ.get("FLUIDSF_RUN_MPI_TESTS") == "1"


def _run_example_script(tmp_path, script_name):
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun is not available in PATH.")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "examples" / "python_scripts" / script_name
    env = os.environ.copy()
    env["HYDRA_LAUNCHER"] = "fork"

    completed = subprocess.run(
        [
            mpirun,
            "-launcher",
            "fork",
            "-n",
            "2",
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(
            f"MPI example compare script failed: {script_name}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    stem = script_name.removesuffix(".py")
    npz_path = output_dir / f"{stem}.npz"
    png_path = output_dir / f"{stem}.png"
    assert npz_path.exists(), f"Missing expected npz output for {script_name}"
    assert png_path.exists(), f"Missing expected png output for {script_name}"
    return np.load(npz_path)


def _max_serial_mpi_diff(npz_data):
    diffs = []
    for key in npz_data.files:
        pair_prefixes = [
            ("serial_", "mpi_"),
            ("default_serial_", "default_mpi_"),
            ("full_serial_", "full_mpi_"),
        ]
        for serial_prefix, mpi_prefix in pair_prefixes:
            if not key.startswith(serial_prefix):
                continue
            mpi_key = mpi_prefix + key[len(serial_prefix) :]
            if mpi_key in npz_data.files:
                serial_values = npz_data[key]
                mpi_values = npz_data[mpi_key]
                valid = np.isfinite(serial_values) & np.isfinite(mpi_values)
                if np.any(valid):
                    diffs.append(float(np.max(np.abs(serial_values[valid] - mpi_values[valid]))))
            break
    assert diffs, "No serial/mpi comparison arrays were found in the saved output."
    return max(diffs)


@pytest.mark.skipif(
    not RUN_MPI_TESTS,
    reason="Set FLUIDSF_RUN_MPI_TESTS=1 to enable MPI integration tests.",
)
@pytest.mark.parametrize(
    "script_name, tolerance",
    [
        ("ex_1d_serial_vs_mpi_synthetic_data.py", 1e-14),
        ("ex_2d_serial_vs_mpi_sample_data.py", 1e-12),
        ("ex_3d_serial_vs_mpi_sample_data.py", 1e-12),
        ("ex_cascade_serial_vs_mpi_sample_data.py", 1e-12),
    ],
)
def test_mpi_example_compare_scripts_match_serial(tmp_path, script_name, tolerance):
    data = _run_example_script(tmp_path, script_name)
    max_diff = _max_serial_mpi_diff(data)
    assert max_diff <= tolerance, (
        f"{script_name} exceeded tolerance: max_diff={max_diff:.3e}, "
        f"tolerance={tolerance:.3e}"
    )
