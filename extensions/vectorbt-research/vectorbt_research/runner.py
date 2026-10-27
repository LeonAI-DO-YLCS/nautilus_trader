"""
Story 1 runner: quote rows and frozen signals in, one saved MT5 simulation run out.

Quote text is parsed to exact integers, the frozen signals go through the vectorbt signal
matrix, the ledger runs once for the chosen price scenario, and the manifest and the three
tables are saved with `write_run`. Money is integer cents and time is integer UTC
nanoseconds. There are no floats.

`run_story_1` takes quote rows from the caller. `run_story_1_window` reads a real window
from the yearly dataset, checks it against the exporter CSV, and reads the continuation.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import deriv_mt5.rules
import deriv_mt5.ticks
import pyarrow.parquet as pq

from vectorbt_research.charts import write_chart
from vectorbt_research.dataset import check_window_against_csv, read_quote_window
from vectorbt_research.engine import LedgerResult, run_ledger
from vectorbt_research.input import ParsedQuote, QuoteDiagnostics, parse_quote_rows
from vectorbt_research.input import _utc_nanoseconds as _utc_ns
from vectorbt_research.reports import write_run
from vectorbt_research.signals import (
    frozen_signals_to_vectorbt,
    signals_from_vectorbt,
)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_PROJECT_DIR = Path(__file__).resolve().parent.parent
_DIRECTLY_PROVEN_UTC_DATES = ("2025-10-27", "2026-08-18")
_NOT_APPLICABLE = "not applicable"
_OPENED_STATUSES = ("settled", "unresolved")
# Initial continuation read size; gaps extend this bound, never end the data.
_CONTINUATION_STEP_NS = 2 * 60 * 1_000_000_000

_POLICIES = {
    "symbol": "R_75",
    "directions": {"CALL": "Rise", "PUT": "Fall"},
    "expiry": "start plus 60 seconds",
    "net_payout_basis_points": 9520,
    "tie_rule": "equal exit price loses",
    "rounding": "cent floor",
    "minimum_stake_cents": 35,
    "stake_basis_points_of_available_cash": 5,
    "cash_rule": "available cash excludes locked stakes",
    "max_open_contracts": 5,
    "stale_limit_seconds": 10,
    "missing_side_rule": "row skipped and counted",
    "order": "timestamp, then source ordinal, then signal order",
}


def run_story_1(
    *,
    quote_rows: Sequence[Mapping[str, object]],
    frozen_signals: Sequence[Mapping[str, object]],
    scenario: str,
    output_dir: str | Path,
    starting_cash_cents: int = 1_000_000,
) -> Path:
    """
    Run one scenario (bid, mid or ask) and return the new saved run folder.

    Each quote row holds `timestamp` (UTC text), `ordinal`, `bid` and `ask` (price text,
    or None when missing). Each frozen signal holds `signal_id`, `direction` (CALL or
    PUT) and `start_time` (UTC text). A hand-written signal is known from its start time.
    """
    return _run(
        quote_rows=quote_rows,
        frozen_signals=frozen_signals,
        scenario=scenario,
        output_dir=output_dir,
        starting_cash_cents=starting_cash_cents,
    )


def run_story_1_window(
    *,
    dataset_folder: str | Path,
    window_start_ns: int,
    window_end_ns: int,
    frozen_signals: Sequence[Mapping[str, object]],
    scenario: str,
    output_dir: str | Path,
    starting_cash_cents: int = 1_000_000,
) -> Path:
    """
    Run one scenario (bid, mid or ask) on a real quote window and return the saved folder.

    The window is [window_start_ns, window_end_ns) in UTC nanoseconds and must lie in one
    yearly file of the dataset folder. It is cross-checked against the exporter CSV first,
    and a mismatch raises ValueError before anything is saved. Continuation ends at the
    first valid quote strictly after the last open expiry, or the true yearly-file end.
    No contract enters at or after window_end_ns. Cross-year continuation raises a clear
    error. The output folder must not be inside the dataset.
    """
    if window_end_ns <= window_start_ns:
        raise ValueError("window_end_ns must be after window_start_ns")
    dataset = Path(dataset_folder)
    output = Path(output_dir)
    if output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("output folder must not be inside the dataset folder")
    year = _utc_year(window_start_ns)
    if _utc_year(window_end_ns - 1) != year:
        raise ValueError("the window and its continuation must lie in one yearly file")
    parquet_path = dataset / f"Volatility75Index_{year}.parquet"
    csv_path = dataset / f"Volatility75Index_{year}.csv"
    window_rows = check_window_against_csv(
        parquet_path, csv_path, window_start_ns, window_end_ns
    )
    quote_rows = read_quote_window(parquet_path, window_start_ns, window_end_ns)
    quotes, _ = parse_quote_rows(quote_rows)
    ledger = _evaluate_ledger(
        quotes,
        _signal_records(frozen_signals),
        scenario,
        starting_cash_cents,
        window_end_ns,
    )
    open_expiries = [
        contract["expiry_timestamp_ns"]
        for contract in ledger.contracts
        if contract["status"] == "unresolved"
    ]
    read_end_ns = window_end_ns
    if open_expiries and (not quotes or quotes[-1].timestamp_ns <= max(open_expiries)):
        quote_rows, read_end_ns = _read_continuation(
            parquet_path, quote_rows, window_end_ns, max(open_expiries), year
        )
    ordinals = [row["ordinal"] for row in quote_rows]
    return _run(
        quote_rows=quote_rows,
        frozen_signals=frozen_signals,
        scenario=scenario,
        output_dir=output,
        starting_cash_cents=starting_cash_cents,
        window_end_ns=window_end_ns,
        window=(window_start_ns, window_end_ns),
        source={
            "path": str(parquet_path),
            "sha256": _rows_digest(quote_rows),
            "first_row": min(ordinals, default=0),
            "last_row": max(ordinals, default=0),
            "price_scale": 100,
        },
        provenance={
            "csv_check": "matched",
            "parquet_file": parquet_path.name,
            "csv_file": csv_path.name,
            "window_rows": window_rows,
            "continuation_rows": len(quote_rows) - window_rows,
            "continuation_end_timestamp_ns": read_end_ns,
        },
    )


def _run(
    *,
    quote_rows: Sequence[Mapping[str, object]],
    frozen_signals: Sequence[Mapping[str, object]],
    scenario: str,
    output_dir: str | Path,
    starting_cash_cents: int,
    window_end_ns: int | None = None,
    window: tuple[int, int] | None = None,
    source: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
) -> Path:
    """
    Run the shared pipeline and save the run. Real data passes `window`, `source` and
    `provenance`; quote rows given by the caller leave them None.
    """
    quotes, diagnostics = parse_quote_rows(quote_rows)
    signals = _signal_records(frozen_signals)
    quote_times = [quote.timestamp_ns for quote in quotes]
    ledger = _evaluate_ledger(
        quotes, signals, scenario, starting_cash_cents, window_end_ns
    )
    record = {
        "manifest": _manifest(
            scenario,
            quote_rows,
            quotes,
            diagnostics,
            signals,
            quote_times,
            ledger,
            window=window,
            source=source,
            provenance=provenance,
        ),
        "signals": [
            {
                "signal_id": signal["signal_id"],
                "direction": signal["direction"],
                "requested_start_timestamp_ns": signal["start_time_ns"],
                "available_timestamp_ns": signal["start_time_ns"],
            }
            for signal in signals
        ],
        "contracts": ledger.contracts,
        "cash": ledger.cash,
    }
    run_dir = write_run(record, output_dir)
    write_chart(run_dir, quotes)
    return run_dir


def _read_continuation(
    parquet_path: Path,
    rows: list[dict[str, object]],
    window_end_ns: int,
    last_expiry_ns: int,
    year: int,
) -> tuple[list[dict[str, object]], int]:
    """Read through gaps until expiry is bounded by a later valid quote or true EOF."""
    year_end_ns = _utc_ns(f"{year + 1}-01-01T00:00:00Z")
    if last_expiry_ns >= year_end_ns:
        raise ValueError("the window and its continuation must lie in one yearly file")
    with pq.ParquetFile(parquet_path) as parquet:
        total_rows = parquet.metadata.num_rows
    read_start_ns = window_end_ns
    step_ns = _CONTINUATION_STEP_NS
    while read_start_ns < year_end_ns:
        if rows and rows[-1]["ordinal"] == total_rows - 1:
            return rows, read_start_ns
        read_end_ns = min(read_start_ns + step_ns, year_end_ns)
        continuation = read_quote_window(parquet_path, read_start_ns, read_end_ns)
        quotes, _ = parse_quote_rows(continuation)
        later = next((q for q in quotes if q.timestamp_ns > last_expiry_ns), None)
        if later is not None:
            # Keep the first later quote only; expiry's complete group is already read.
            rows.extend(row for row in continuation if row["ordinal"] <= later.ordinal)
            return rows, later.timestamp_ns + 1
        rows.extend(continuation)
        read_start_ns = read_end_ns
        step_ns *= 2
    if rows and rows[-1]["ordinal"] != total_rows - 1:
        raise ValueError("the window and its continuation must lie in one yearly file")
    return rows, read_start_ns


def _evaluate_ledger(
    quotes: list[ParsedQuote],
    signals: list[dict[str, object]],
    scenario: str,
    starting_cash_cents: int,
    window_end_ns: int | None,
) -> LedgerResult:
    """Use the same signal matrix and exact ledger for window inspection and saving."""
    ticks = [
        (
            quote.timestamp_ns,
            quote.ordinal,
            quote.bid_half_cents,
            quote.mid_half_cents,
            quote.ask_half_cents,
        )
        for quote in quotes
    ]
    quote_times = [quote.timestamp_ns for quote in quotes]
    # The matrix has one row per distinct time: a signal may start between two quotes.
    matrix_times = sorted(
        set(quote_times) | {signal["start_time_ns"] for signal in signals}
    )
    matrix = frozen_signals_to_vectorbt(signals, matrix_times)
    return run_ledger(
        ticks,
        signals_from_vectorbt(matrix),
        scenario=scenario,
        starting_cash_cents=starting_cash_cents,
        window_end_ns=window_end_ns,
    )


def _signal_records(
    frozen_signals: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """
    Return the frozen signals in list order with start times as UTC nanoseconds.
    """
    ids = [signal["signal_id"] for signal in frozen_signals]
    if len(set(ids)) != len(ids):
        raise ValueError("signal ids must be unique")
    return [
        {
            "signal_id": signal["signal_id"],
            "direction": signal["direction"],
            "start_time_ns": _utc_ns(signal["start_time"]),
            "availability_ordinal": 0,
        }
        for signal in frozen_signals
    ]


def _manifest(
    scenario: str,
    quote_rows: Sequence[Mapping[str, object]],
    quotes: list[ParsedQuote],
    diagnostics: QuoteDiagnostics,
    signals: list[dict[str, object]],
    quote_times: list[int],
    ledger: LedgerResult,
    *,
    window: tuple[int, int] | None = None,
    source: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Build the run manifest. `spot_scale` 200 means the saved spots are price x 200.

    Without `window`, the run spans its quote rows, and the source is the in-memory rows.
    """
    times = quote_times + [signal["start_time_ns"] for signal in signals]
    if not times:
        raise ValueError("a run needs quote rows or signals")
    start_ns = min(times)
    if window is None:
        window = (start_ns, max(quote_times, default=start_ns))
    window_start_ns, window_end_ns = window
    if _utc_date(window_start_ns) in _DIRECTLY_PROVEN_UTC_DATES:
        utc_status = "directly proven"
        utc_proof = (
            f"DR-2 D8: owner-confirmed UTC sample date {_utc_date(window_start_ns)}"
        )
    else:
        utc_status = "UTC, same exporter, not directly proven"
        utc_proof = "DR-2 D8: same exporter as the proven sample dates"
    ordinals = [quote.ordinal for quote in quotes]
    opened = sum(
        contract["status"] in _OPENED_STATUSES for contract in ledger.contracts
    )
    return {
        "schema_version": 1,
        "label": "MT5 simulation",
        "engine": "vectorbt_research",
        "purpose": "story 1 end-to-end run",
        "code_hash": _digest(sorted(Path(__file__).parent.glob("*.py"))),
        "rule_hash": _digest(
            [Path(deriv_mt5.rules.__file__), Path(deriv_mt5.ticks.__file__)]
        ),
        "lock_hash": _digest([_PROJECT_DIR / "uv.lock"]),
        "strategy": {
            "source": _NOT_APPLICABLE,
            "hash": _NOT_APPLICABLE,
            "parameters": _NOT_APPLICABLE,
            "seed": _NOT_APPLICABLE,
        },
        "scenario": scenario,
        "windows": [
            {
                "start_timestamp_ns": window_start_ns,
                "end_timestamp_ns": window_end_ns,
                "utc_status": utc_status,
                "utc_offset_seconds": 0,
            }
        ],
        "utc_proof_references": utc_proof,
        "holdout": _NOT_APPLICABLE,
        "sources": [
            source
            or {
                "path": "in-memory quote rows",
                "sha256": _rows_digest(quote_rows),
                "first_row": min(ordinals, default=0),
                "last_row": max(ordinals, default=0),
                "price_scale": 100,
            }
        ],
        "quote_provenance": provenance or "quote rows given by the caller",
        "policies": _POLICIES,
        "information_availability": "signal visible from its start time only",
        "selected_prices": (
            "entry first tick at or after start; exit last tick at or before expiry"
        ),
        "spot_scale": 200,
        "quote_diagnostics": {
            "skipped_rows": diagnostics.skipped_rows,
            "missing_bid": diagnostics.missing_bid,
            "missing_ask": diagnostics.missing_ask,
        },
        "counts": {
            "signals": len(signals),
            "contracts": opened,
            "skipped_signals": ledger.skipped_count,
            "unresolved_contracts": ledger.unresolved_count,
        },
        "completeness": (
            {"status": "incomplete", "reason": ledger.incomplete_reason}
            if ledger.incomplete
            else {"status": "complete"}
        ),
        "gate": {"status": "not evaluated", "evidence": _NOT_APPLICABLE},
    }


def _utc_date(timestamp_ns: int) -> str:
    return (
        (_EPOCH + timedelta(seconds=timestamp_ns // 1_000_000_000)).date().isoformat()
    )


def _utc_year(timestamp_ns: int) -> int:
    return (_EPOCH + timedelta(seconds=timestamp_ns // 1_000_000_000)).year


def _rows_digest(rows: Sequence[Mapping[str, object]]) -> str:
    """
    Return the SHA-256 of the quote rows as canonical JSON.
    """
    return hashlib.sha256(
        json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _digest(paths: Sequence[Path]) -> str:
    """
    Return the SHA-256 of the files' names and bytes, in the given order.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
