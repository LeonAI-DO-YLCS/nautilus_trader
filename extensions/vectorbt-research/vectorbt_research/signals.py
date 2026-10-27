"""
Frozen hand-written signals as a vectorbt signal matrix, and back.

Start times are integer UTC nanoseconds. Rows follow the given quote-time order and
columns follow the frozen list order, so signals at the same time keep separate columns.
"""

import numpy as np
import pandas as pd
import vectorbt  # noqa: F401 registers the .vbt accessor on pandas objects


def frozen_signals_to_vectorbt(
    frozen_signals: list[dict[str, object]], quote_times_ns: list[int]
) -> pd.DataFrame:
    """
    Build a boolean matrix: one row per quote time, one column per frozen signal.

    A signal's column is True only on the row of its start time. The column label keeps
    the signal id, direction and availability ordinal so they can be read back.
    """
    quote_index = pd.Index(
        _require_ns_ints("quote time", quote_times_ns), dtype="int64"
    )
    starts = _require_ns_ints(
        "start time", [signal["start_time_ns"] for signal in frozen_signals]
    )
    if not quote_index.is_unique:
        raise ValueError("quote times must be unique")
    rows = quote_index.get_indexer(starts)
    if (rows < 0).any():
        raise ValueError("every frozen signal start time must be a quote time")
    matrix = np.zeros((len(quote_index), len(starts)), dtype=bool)
    matrix[rows, np.arange(len(starts))] = True
    columns = pd.MultiIndex.from_arrays(
        [
            [signal["signal_id"] for signal in frozen_signals],
            [signal["direction"] for signal in frozen_signals],
            [signal["availability_ordinal"] for signal in frozen_signals],
        ],
        names=["signal_id", "direction", "availability_ordinal"],
    )
    return pd.DataFrame(matrix, index=quote_index, columns=columns)


def signals_from_vectorbt(vectorbt_signals: pd.DataFrame) -> list[dict[str, object]]:
    """
    Read one record per column, in column order. Each column must mark exactly one row.
    """
    matrix = vectorbt_signals.to_numpy()
    records = []
    for position, (signal_id, direction, ordinal) in enumerate(
        vectorbt_signals.columns
    ):
        rows = np.flatnonzero(matrix[:, position])
        if len(rows) != 1:
            raise ValueError(
                f"signal {signal_id!r} must mark exactly one row, got {len(rows)}"
            )
        records.append(
            {
                "signal_id": signal_id,
                "direction": direction,
                "start_time_ns": int(vectorbt_signals.index[rows[0]]),
                "availability_ordinal": int(ordinal),
            }
        )
    return records


def _require_ns_ints(name: str, values: list[object]) -> list[int]:
    """
    Return the values when each one is an exact int (UTC nanoseconds), otherwise raise.
    """
    for value in values:
        if type(value) is not int:
            raise TypeError(
                f"{name} must be an int (UTC nanoseconds), got {type(value).__name__}"
            )
    return values
