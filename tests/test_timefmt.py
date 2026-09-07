"""Timestamp formatting is the contract between transcripts and citations."""

from __future__ import annotations

import pytest

from companion.utils.timefmt import (
    format_duration, format_range, format_timestamp, parse_timestamp,
)


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0:00"), (5, "0:05"), (65, "1:05"), (600, "10:00"),
        (3599, "59:59"), (3600, "1:00:00"), (3725, "1:02:05"),
        (12.9, "0:12"), (-5, "0:00"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_format_range_uses_en_dash():
    assert format_range(750.2, 790.4) == "12:30–13:10"


@pytest.mark.parametrize("value", ["0:00", "1:05", "59:59", "1:02:05"])
def test_parse_roundtrip(value):
    assert format_timestamp(parse_timestamp(value)) == value


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_timestamp("not a time")


def test_format_duration():
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3720) == "1h 02m"
