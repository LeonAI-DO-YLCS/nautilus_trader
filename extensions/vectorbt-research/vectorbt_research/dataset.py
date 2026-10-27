"""
Exact reader for the yearly Volatility 75 Parquet files.

Reads one time window and returns rows in the form that input.parse_quote_rows accepts.
Each price is taken from the shortest decimal text of its double and must sit on the
half-cent grid (0.005). Nothing is rounded.
"""

import csv
import math
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vectorbt_research.input import _utc_nanoseconds

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_HALF_CENTS_PER_UNIT = 200
_PLAIN_PRICE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_DATE = "<DATE>"
_TIME = "<TIME>"
_BID = "<BID>"
_ASK = "<ASK>"
_PARQUET_COLUMNS = [_DATE, _TIME, _BID, _ASK]


def read_quote_window(
    path: str | Path, start_ns: int, end_ns: int
) -> list[dict[str, object]]:
    """
    Return the quotes with start_ns <= timestamp < end_ns, in source order.

    Each row holds timestamp (UTC text), ordinal (0-based row position in the file), and bid
    and ask (exact price text, or None when empty). Row groups whose DATE range misses the
    window are not read.
    """
    if end_ns <= start_ns:
        raise ValueError(f"end_ns must be after start_ns, got {start_ns} and {end_ns}")
    first_date = _date_text(start_ns)
    last_date = _date_text(end_ns - 1)
    rows: list[dict[str, object]] = []
    with pq.ParquetFile(path) as parquet:
        metadata = parquet.metadata
        date_index = parquet.schema.names.index(_DATE)
        row_offset = 0
        for group in range(metadata.num_row_groups):
            group_metadata = metadata.row_group(group)
            statistics = group_metadata.column(date_index).statistics
            if _may_overlap(statistics, first_date, last_date):
                table = parquet.read_row_group(group, columns=_PARQUET_COLUMNS)
                rows.extend(_window_rows(table, row_offset, start_ns, end_ns))
            row_offset += group_metadata.num_rows
    return rows


def check_window_against_csv(
    parquet_path: str | Path, csv_path: str | Path, start_ns: int, end_ns: int
) -> int:
    """
    Check that the Parquet window matches the exporter CSV, and return its row count.

    Rows must match in count, order, UTC time, and half-cent integers for BID and ASK. An
    empty price must be empty on both sides. Raises ValueError on the first mismatch.
    """
    expected = read_quote_window(parquet_path, start_ns, end_ns)
    actual = _read_csv_window(csv_path, start_ns, end_ns)
    if len(expected) != len(actual):
        raise ValueError(
            f"row count differs in window: parquet {len(expected)}, csv {len(actual)}"
        )
    for want, got in zip(expected, actual, strict=True):
        timestamp = str(want["timestamp"])
        if _utc_nanoseconds(timestamp) != _utc_nanoseconds(str(got["timestamp"])):
            raise ValueError(
                f"timestamp differs: parquet {timestamp}, csv {got['timestamp']}"
            )
        for label, key in (("BID", "bid"), ("ASK", "ask")):
            wanted = _half_cents_of(want[key], label, timestamp)
            if wanted != _half_cents_of(got[key], label, timestamp):
                raise ValueError(
                    f"{label} differs at {timestamp}: parquet {want[key]}, csv {got[key]}"
                )
    return len(expected)


def _may_overlap(statistics: object, first_date: str, last_date: str) -> bool:
    """
    Return False only when the row group's DATE statistics prove it misses the window.
    """
    if statistics is None or not statistics.has_min_max:
        return True
    return statistics.min <= last_date and statistics.max >= first_date


def _window_rows(
    table: pa.Table, row_offset: int, start_ns: int, end_ns: int
) -> Iterator[dict[str, object]]:
    rows = zip(
        table.column(_DATE).to_pylist(),
        table.column(_TIME).to_pylist(),
        table.column(_BID).to_pylist(),
        table.column(_ASK).to_pylist(),
        strict=True,
    )
    for index, (date, clock, bid, ask) in enumerate(rows):
        timestamp = _timestamp_text(date, clock)
        if start_ns <= _utc_nanoseconds(timestamp) < end_ns:
            yield {
                "timestamp": timestamp,
                "ordinal": row_offset + index,
                "bid": _price_text(bid, "BID", timestamp),
                "ask": _price_text(ask, "ASK", timestamp),
            }


def _read_csv_window(
    path: str | Path, start_ns: int, end_ns: int
) -> list[dict[str, object]]:
    """
    Return the exporter CSV rows in the window as text, in file order.

    The file is tab-separated, starts with DATE and TIME, and is sorted by time. Lines on
    dates outside the window are skipped without parsing.
    """
    first_date = _date_text(start_ns)
    last_date = _date_text(end_ns - 1)
    rows: list[dict[str, object]] = []
    previous: int | None = None
    with open(path, newline="", encoding="utf-8") as handle:
        header = next(csv.reader([handle.readline()], delimiter="\t"))
        if header[:2] != [_DATE, _TIME] or not {_BID, _ASK} <= set(header):
            raise ValueError(f"unexpected CSV header in {path}: {header}")
        columns = {name: index for index, name in enumerate(header)}
        for line in handle:
            date = line.partition("\t")[0]
            if date < first_date:
                continue
            if date > last_date:
                break
            fields = next(csv.reader([line], delimiter="\t"))
            timestamp = _timestamp_text(fields[0], fields[1])
            moment = _utc_nanoseconds(timestamp)
            if moment < start_ns:
                continue
            if moment >= end_ns:
                break
            if previous is not None and moment < previous:
                raise ValueError(f"CSV rows are out of time order at {timestamp}")
            previous = moment
            rows.append(
                {
                    "timestamp": timestamp,
                    "bid": fields[columns[_BID]] or None,
                    "ask": fields[columns[_ASK]] or None,
                }
            )
    return rows


def _price_text(value: object, label: str, timestamp: str) -> str | None:
    """
    Return the exact decimal text of a Parquet double, or None when the cell is empty.

    The text is the shortest form that reads back as the same double, so 50197.9 stays
    50197.9 and no digit is rounded. A price off the half-cent grid raises.
    """
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} at {timestamp} is not a finite price: {value!r}")
    text = format(Decimal(repr(value)), "f")
    _half_cents_of(text, label, timestamp)
    return text


def _half_cents_of(text: object, label: str, timestamp: str) -> int | None:
    """
    Return a plain decimal price as an integer count of half cents (0.005 units).

    This is the exact test of price times 200 being whole, done on the digits, so no
    rounding can hide an off-grid price. Returns None for a missing price.
    """
    if text is None:
        return None
    if type(text) is not str or _PLAIN_PRICE.fullmatch(text) is None:
        raise ValueError(f"{label} at {timestamp} is not a plain price: {text!r}")
    whole, _, fraction = text.partition(".")
    scaled = int(whole + fraction) * _HALF_CENTS_PER_UNIT
    scale = 10 ** len(fraction)
    if scaled % scale:
        raise ValueError(f"{label} at {timestamp} is off the 0.005 grid: {text}")
    return scaled // scale


def _timestamp_text(date: object, clock: object) -> str:
    """
    Join the file's DATE (2026.08.18) and TIME (03:20:00.241) into UTC text.
    """
    if type(date) is not str or type(clock) is not str:
        raise TypeError(f"DATE and TIME must be text, got {date!r} and {clock!r}")
    return f"{date.replace('.', '-')}T{clock}Z"


def _date_text(moment_ns: int) -> str:
    """
    Return the UTC calendar date of an instant in the file's DATE format.
    """
    seconds = moment_ns // 1_000_000_000
    return (_EPOCH + timedelta(seconds=seconds)).strftime("%Y.%m.%d")
