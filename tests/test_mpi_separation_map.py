import numpy as np
import pytest

from fluidsf.mpi.separation_map import (
    compute_axis_index_list,
    compute_rank_coordinates,
    compute_separation_map,
    compute_separation_pairs_for_rank,
    validate_processor_grid,
)


def test_validate_processor_grid():
    assert validate_processor_grid(16, 12, 2, 4) == (8, 6)


@pytest.mark.parametrize(
    "nx, ny, px, nprocs",
    [
        (15, 12, 2, 4),
        (16, 11, 2, 4),
        (16, 12, 3, 4),
        (16, 10, 2, 6),
    ],
)
def test_validate_processor_grid_errors(nx, ny, px, nprocs):
    with pytest.raises(ValueError):
        validate_processor_grid(nx, ny, px, nprocs)


def test_compute_rank_coordinates():
    assert compute_rank_coordinates(0, 2) == (0, 0)
    assert compute_rank_coordinates(1, 2) == (0, 1)
    assert compute_rank_coordinates(2, 2) == (1, 0)
    assert compute_rank_coordinates(3, 2) == (1, 1)


def test_compute_axis_index_list_matches_fastsf_pattern():
    np.testing.assert_array_equal(compute_axis_index_list(8, 2, 0), [0, 7, 4, 3])
    np.testing.assert_array_equal(compute_axis_index_list(8, 2, 1), [1, 6, 5, 2])


def test_compute_axis_index_list_single_index_per_rank():
    np.testing.assert_array_equal(compute_axis_index_list(4, 4, 0), [0])
    np.testing.assert_array_equal(compute_axis_index_list(4, 4, 3), [3])


def test_compute_axis_index_list_odd_list_size_falls_back_to_unique_strided_assignment():
    np.testing.assert_array_equal(compute_axis_index_list(5, 1, 0), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(compute_axis_index_list(10, 2, 0), [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(compute_axis_index_list(10, 2, 1), [1, 3, 5, 7, 9])


def test_compute_separation_pairs_for_rank():
    pairs = compute_separation_pairs_for_rank(8, 8, 2, 4, rank=0)
    expected = np.array([[0, 0], [0, 3], [3, 0], [3, 3]])
    np.testing.assert_array_equal(pairs, expected)


def test_compute_separation_map_covers_all_pairs_once():
    rank_pairs = compute_separation_map(8, 8, 2, 4)
    all_pairs = np.vstack(rank_pairs)
    expected = {(x, y) for x in range(4) for y in range(4)}
    actual = {tuple(pair) for pair in all_pairs.tolist()}

    assert len(all_pairs) == 16
    assert len(actual) == 16
    assert actual == expected


def test_compute_separation_map_covers_all_pairs_once_with_odd_owner_counts():
    rank_pairs = compute_separation_map(12, 10, 2, 2)
    all_pairs = np.vstack(rank_pairs)
    expected = {(x, y) for x in range(6) for y in range(5)}
    actual = {tuple(pair) for pair in all_pairs.tolist()}

    assert len(all_pairs) == 30
    assert len(actual) == 30
    assert actual == expected
