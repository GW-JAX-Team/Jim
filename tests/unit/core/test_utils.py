"""Tests for the shared core utilities."""

import pytest

from jimgw.core.utils import round_up_to_power_of_two


@pytest.mark.parametrize(
    ("n", "expected"),
    [(-3, 1), (0, 1), (1, 1), (2, 2), (3, 4), (8, 8), (9, 16)],
)
def test_round_up_to_power_of_two(n, expected):
    result = round_up_to_power_of_two(n)
    assert result == expected
    assert type(result) is int
