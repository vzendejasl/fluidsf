import importlib

import numpy as np

from fluidsf.calculate_sf_maps_2d import calculate_sf_maps_2d

generate_sf_maps_2d_module = importlib.import_module("fluidsf.generate_sf_maps_2d")


def test_calculate_sf_maps_2d_uses_exact_sf_type_matches():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    u = np.tile(x, (len(y), 1))
    v = np.zeros_like(u)
    adv_x = np.zeros_like(u)
    adv_y = np.zeros_like(u)

    output = calculate_sf_maps_2d(
        u,
        v,
        x,
        y,
        adv_x,
        adv_y,
        1,
        0,
        ["LLL"],
    )

    assert set(output) == {"SF_LLL_xy"}


def test_generate_sf_maps_2d_only_reads_requested_keys(monkeypatch):
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    u = np.zeros((len(y), len(x)))
    v = np.zeros_like(u)

    def fake_calculate_sf_maps_2d(*args, **kwargs):
        return {"SF_LLL_xy": 3.0}

    monkeypatch.setattr(
        generate_sf_maps_2d_module,
        "calculate_sf_maps_2d",
        fake_calculate_sf_maps_2d,
    )

    output = generate_sf_maps_2d_module.generate_sf_maps_2d(u, v, x, y, sf_type=["LLL"])

    assert "SF_LL_xy" not in output
    np.testing.assert_allclose(output["SF_LLL_xy"], 3.0)


def test_generate_sf_maps_2d_uses_scalar_advection_output(monkeypatch):
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    u = np.zeros((len(y), len(x)))
    v = np.zeros_like(u)
    scalar = np.ones_like(u)

    def fake_calculate_advection_2d(u, v, x, y, dx=None, dy=None, grid_type="uniform", scalar=None):
        if scalar is None:
            return np.full_like(u, 10.0), np.full_like(u, 20.0)
        return np.full_like(u, 30.0)

    def fake_calculate_sf_maps_2d(*args, **kwargs):
        return {
            "SF_advection_velocity_xy": 1.0,
            "SF_advection_scalar_xy": 2.0,
        }

    monkeypatch.setattr(
        generate_sf_maps_2d_module,
        "calculate_advection_2d",
        fake_calculate_advection_2d,
    )
    monkeypatch.setattr(
        generate_sf_maps_2d_module,
        "calculate_sf_maps_2d",
        fake_calculate_sf_maps_2d,
    )

    output = generate_sf_maps_2d_module.generate_sf_maps_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S"],
        scalar=scalar,
    )

    np.testing.assert_allclose(output["SF_advection_velocity_xy"], 1.0)
    np.testing.assert_allclose(output["SF_advection_scalar_xy"], 2.0)
