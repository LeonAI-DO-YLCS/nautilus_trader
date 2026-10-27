"""
Saved research runs: manifest JSON, integer Parquet tables and a summary text.

Money is integer cents and time is integer UTC nanoseconds. Both are stored as int64.
"""

import json
import numbers
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_SCHEMAS = {
    "signals": pa.schema(
        [
            ("signal_id", pa.string()),
            ("direction", pa.string()),
            ("requested_start_timestamp_ns", pa.int64()),
            ("available_timestamp_ns", pa.int64()),
        ]
    ),
    "contracts": pa.schema(
        [
            ("signal_id", pa.string()),
            ("direction", pa.string()),
            ("price_reference", pa.string()),
            ("status", pa.string()),
            ("reason", pa.string()),
            ("outcome", pa.string()),
            ("entry_ordinal", pa.int64()),
            ("exit_ordinal", pa.int64()),
            ("entry_timestamp_ns", pa.int64()),
            ("expiry_timestamp_ns", pa.int64()),
            ("exit_timestamp_ns", pa.int64()),
            ("entry_spot", pa.int64()),
            ("exit_spot", pa.int64()),
            ("stake_cents", pa.int64()),
            ("gross_credit_cents", pa.int64()),
            ("net_cents", pa.int64()),
        ]
    ),
    "cash": pa.schema(
        [
            ("event", pa.string()),
            ("signal_id", pa.string()),
            ("before_available_cents", pa.int64()),
            ("before_locked_cents", pa.int64()),
            ("available_cents", pa.int64()),
            ("locked_cents", pa.int64()),
        ]
    ),
}


# Shape of a saved manifest. A dict lists the required keys of a group. A one-item list
# is the shape of every item in a list, and the list must not be empty. int means an
# exact JSON integer, so floats and bools are rejected. _COUNT is an exact integer that
# is not negative. str means exact text. object means any value except null. Any other
# value, such as 1 or "MT5 simulation", must equal that value exactly.
_COUNT = object()
_MANIFEST_SHAPE = {
    "schema_version": 1,
    "label": "MT5 simulation",
    "engine": str,
    "purpose": str,
    "code_hash": str,
    "rule_hash": str,
    "lock_hash": str,
    "strategy": {
        "source": object,
        "hash": object,
        "parameters": object,
        "seed": object,
    },
    "scenario": str,
    "windows": [
        {
            "start_timestamp_ns": int,
            "end_timestamp_ns": int,
            "utc_status": str,
            "utc_offset_seconds": int,
        }
    ],
    "utc_proof_references": object,
    "holdout": object,
    "sources": [
        {
            "path": str,
            "sha256": str,
            "first_row": int,
            "last_row": int,
            "price_scale": int,
        }
    ],
    "quote_provenance": object,
    "policies": {
        "symbol": str,
        "directions": {"CALL": str, "PUT": str},
        "expiry": str,
        "net_payout_basis_points": int,
        "tie_rule": str,
        "rounding": str,
        "minimum_stake_cents": int,
        "stake_basis_points_of_available_cash": int,
        "cash_rule": str,
        "max_open_contracts": int,
        "stale_limit_seconds": int,
        "missing_side_rule": str,
        "order": str,
    },
    "information_availability": object,
    "selected_prices": object,
    "quote_diagnostics": {
        "skipped_rows": _COUNT,
        "missing_bid": _COUNT,
        "missing_ask": _COUNT,
    },
    "counts": {
        "signals": _COUNT,
        "contracts": _COUNT,
        "skipped_signals": _COUNT,
        "unresolved_contracts": _COUNT,
    },
    "completeness": {"status": str},
    "gate": {"status": str, "evidence": object},
}


def write_run(record: Mapping[str, Any], output_dir: str | Path) -> Path:
    """
    Write a run to a new child folder of output_dir and return that folder.

    The record holds the manifest dict and the signals, contracts and cash rows.
    Every call creates a new folder and fails if that folder already exists. The files
    go into a hidden partial folder that is renamed once complete, so a failed write
    never leaves a folder named run-*.
    """
    tables = {name: _table(record[name], schema) for name, schema in _SCHEMAS.items()}
    # The summary uses the stored int64 values, so it always matches the Parquet file.
    cash = tables["cash"]
    summary = _summary(
        record["manifest"],
        available_cents=cash["available_cents"][-1].as_py(),
        locked_cents=cash["locked_cents"][-1].as_py(),
    )
    run_dir = Path(output_dir) / f"run-{uuid.uuid4().hex}"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    partial_dir = run_dir.with_name(f".{run_dir.name}.partial")
    partial_dir.mkdir(parents=True)
    manifest = json.dumps(
        record["manifest"], sort_keys=True, indent=2, ensure_ascii=False
    )
    (partial_dir / "manifest.json").write_bytes((manifest + "\n").encode("utf-8"))
    for name, table in tables.items():
        pq.write_table(table, partial_dir / f"{name}.parquet")
    (partial_dir / "summary.txt").write_bytes(summary.encode("utf-8"))
    partial_dir.rename(run_dir)
    return run_dir


def read_run(path: str | Path) -> dict[str, Any]:
    """
    Read a saved run folder: manifest dict, three pandas DataFrames and summary text.

    Raise ValueError when the manifest has a missing, null, wrongly typed or out-of-range
    field, or when a count differs from the rows saved for it. A missing file still raises
    FileNotFoundError.
    """
    run_dir = Path(path)
    run: dict[str, Any] = {}
    run["manifest"] = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    _check_manifest(run["manifest"], _MANIFEST_SHAPE, "manifest")
    for name in _SCHEMAS:
        run[name] = pq.read_table(run_dir / f"{name}.parquet").to_pandas()
    _check_counts(run)
    run["summary"] = (run_dir / "summary.txt").read_text(encoding="utf-8")
    return run


def _check_counts(run: dict[str, Any]) -> None:
    """
    Raise ValueError unless each manifest count matches its saved table.

    counts.contracts counts the contracts rows that are not skipped, because a skipped
    signal also has a contracts row.
    """
    counts = run["manifest"]["counts"]
    signal_rows = len(run["signals"])
    if counts["signals"] != signal_rows:
        raise ValueError(
            f"manifest.counts.signals is {counts['signals']}, but signals.parquet "
            f"has {signal_rows} rows"
        )
    opened_rows = int((run["contracts"]["status"] != "skipped").sum())
    if counts["contracts"] != opened_rows:
        raise ValueError(
            f"manifest.counts.contracts is {counts['contracts']}, but contracts.parquet "
            f"has {opened_rows} rows that are not skipped"
        )


def _check_manifest(value: Any, shape: Any, path: str) -> None:
    """
    Raise ValueError unless value matches shape. The message names the field at fault.
    """
    if isinstance(shape, dict):
        if type(value) is not dict:
            raise ValueError(f"{path} must be an object, got {type(value).__name__}")
        for key, child_shape in shape.items():
            child_path = f"{path}.{key}"
            if key not in value:
                raise ValueError(f"missing required field {child_path}")
            _check_manifest(value[key], child_shape, child_path)
    elif isinstance(shape, list):
        if type(value) is not list:
            raise ValueError(f"{path} must be a list, got {type(value).__name__}")
        if not value:
            raise ValueError(f"{path} must not be empty")
        for index, item in enumerate(value):
            _check_manifest(item, shape[0], f"{path}[{index}]")
    elif shape is object:
        if value is None:
            raise ValueError(f"{path} must not be null")
    elif shape is str:
        if type(value) is not str:
            raise ValueError(f"{path} must be text, got {type(value).__name__}")
    elif shape is int:
        if type(value) is not int:
            raise ValueError(f"{path} must be an integer, got {type(value).__name__}")
    elif shape is _COUNT:
        _check_manifest(value, int, path)
        if value < 0:
            raise ValueError(f"{path} must not be negative, got {value}")
    elif type(value) is not type(shape) or value != shape:
        raise ValueError(f"{path} must be {shape!r}, got {value!r}")


def _table(rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    """
    Build a table with exactly the schema's columns, so no key is dropped or left null.

    An integer column takes only exact integers, because Arrow would otherwise
    truncate a float such as 1000000.9 to 1000000 without any error.
    """
    for row in rows:
        if set(row) != set(schema.names):
            raise ValueError(f"row columns {sorted(row)} must be {schema.names}")
        for field in schema:
            value = row[field.name]
            is_integer_column = pa.types.is_integer(field.type)
            if is_integer_column and value is not None and not _is_exact_int(value):
                raise TypeError(
                    f"{field.name} must be an integer, got {type(value).__name__}"
                )
    return pa.Table.from_pylist(rows, schema=schema)


def _is_exact_int(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _summary(
    manifest: Mapping[str, Any], *, available_cents: int, locked_cents: int
) -> str:
    """
    Return one "Field: value" line for each locked summary field.
    """
    fields = (
        ("Label", manifest["label"]),
        ("Scenario", manifest["scenario"]),
        ("Final available", _money(available_cents)),
        ("Locked", f"{locked_cents:,} cents"),
        ("Skipped signals", manifest["counts"]["skipped_signals"]),
        ("Unresolved contracts", manifest["counts"]["unresolved_contracts"]),
        ("Status", manifest["completeness"]["status"]),
    )
    return "".join(f"{name}: {value}\n" for name, value in fields)


def _money(cents: int) -> str:
    whole, part = divmod(abs(cents), 100)
    sign = "-" if cents < 0 else ""
    return f"{cents:,} cents ({sign}{whole:,}.{part:02d} USD)"
