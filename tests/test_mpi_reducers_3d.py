import numpy as np

from fluidsf.mpi.generate_sf_3d_mpi import _compute_velocity_structure_functions
from fluidsf.mpi.reducers_3d import (
    compute_scalar_sf_reduction_3d,
    compute_velocity_sf_reduction_3d,
    finalize_structure_function_reduction,
    finalize_velocity_sf_reduction,
)


def test_compute_velocity_sf_reduction_3d_linear_field_nonperiodic():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]

    reductions = compute_velocity_sf_reduction_3d(
        u, v, w, 1, 0, 0, ("LL", "TT", "LLL", "LTT"), boundary=None
    )

    assert reductions["SF_LL_count"] == 48
    assert reductions["SF_TT_count"] == 48
    assert reductions["SF_LLL_count"] == 48
    assert reductions["SF_LTT_count"] == 48
    assert np.isclose(reductions["SF_LL_sum"], 48.0)
    assert np.isclose(reductions["SF_TT_sum"], 0.0)
    assert np.isclose(reductions["SF_LLL_sum"], 48.0)
    assert np.isclose(reductions["SF_LTT_sum"], 0.0)


def test_compute_velocity_sf_reduction_3d_linear_field_periodic():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]

    reductions = compute_velocity_sf_reduction_3d(
        u, v, w, 1, 0, 0, ("LL", "TT", "LLL", "LTT"), boundary="periodic-all"
    )

    assert reductions["SF_LL_count"] == 64
    assert reductions["SF_TT_count"] == 64
    assert reductions["SF_LLL_count"] == 64
    assert reductions["SF_LTT_count"] == 64
    assert np.isclose(reductions["SF_LL_sum"], 192.0)
    assert np.isclose(reductions["SF_TT_sum"], 0.0)
    assert np.isclose(reductions["SF_LLL_sum"], -384.0)
    assert np.isclose(reductions["SF_LTT_sum"], 0.0)


def test_finalize_velocity_sf_reduction_matches_structure_function_helper():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]
    sf_type = ("LL", "TT", "LLL", "LTT")

    reductions = compute_velocity_sf_reduction_3d(
        u, v, w, 1, 1, 0, sf_type, boundary=None
    )
    reduced_means = finalize_velocity_sf_reduction(reductions, sf_type)
    direct_means = _compute_velocity_structure_functions(
        u, v, w, 1, 1, 0, sf_type, boundary=None
    )

    for key in direct_means:
        np.testing.assert_allclose(reduced_means[key], direct_means[key], atol=1e-14)


def test_compute_scalar_sf_reduction_3d_linear_field_periodic():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]
    scalar = u + 2.0 * v

    reductions = compute_scalar_sf_reduction_3d(
        u, v, w, scalar, 1, 0, 0, ("SS", "LSS"), boundary="periodic-all"
    )
    reduced_means = finalize_structure_function_reduction(reductions, ("SS", "LSS"))

    assert reductions["SF_SS_count"] == 64
    assert reductions["SF_LSS_count"] == 64
    assert np.isclose(reduced_means["SF_SS"], 3.0)
    assert np.isclose(reduced_means["SF_LSS"], -6.0)
