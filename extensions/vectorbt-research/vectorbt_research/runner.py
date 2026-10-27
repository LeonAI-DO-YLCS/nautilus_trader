"""
Story 1 runner: quote rows and frozen signals in, one saved MT5 simulation run out.

Quote text is parsed to exact integers, the frozen signals go through the vectorbt signal
matrix, the ledger runs once for the chosen price scenario, and the manifest and the three
tables are saved with `write_run`. Money is integer cents and time is integer UTC
nanoseconds. There are no floats.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import deriv_mt5.rules
import deriv_mt5.ticks

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
    quotes, diagnostics = parse_quote_rows(quote_rows)
    signals = _signal_records(frozen_signals)
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
    # TODO(part 2): pass window_end_ns (first nanosecond after the window) and
    # compiled=True once the engine supports them. The caller has no window yet.
    ledger = run_ledger(
        ticks,
        signals_from_vectorbt(matrix),
        scenario=scenario,
        starting_cash_cents=starting_cash_cents,
    )
    record = {
        "manifest": _manifest(
            scenario, quote_rows, quotes, diagnostics, signals, quote_times, ledger
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
    return write_run(record, output_dir)


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
) -> dict[str, object]:
    """
    Build the run manifest. `spot_scale` 200 means the saved spots are price x 200.
    """
    times = quote_times + [signal["start_time_ns"] for signal in signals]
    if not times:
        raise ValueError("a run needs quote rows or signals")
    start_ns = min(times)
    if _utc_date(start_ns) in _DIRECTLY_PROVEN_UTC_DATES:
        utc_status = "directly proven"
        utc_proof = f"DR-2 D8: owner-confirmed UTC sample date {_utc_date(start_ns)}"
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
                "start_timestamp_ns": start_ns,
                "end_timestamp_ns": max(quote_times, default=start_ns),
                "utc_status": utc_status,
                "utc_offset_seconds": 0,
            }
        ],
        "utc_proof_references": utc_proof,
        "holdout": _NOT_APPLICABLE,
        "sources": [
            {
                "path": "in-memory quote rows",
                "sha256": hashlib.sha256(
                    json.dumps(
                        list(quote_rows), sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "first_row": min(ordinals, default=0),
                "last_row": max(ordinals, default=0),
                "price_scale": 100,
            }
        ],
        "quote_provenance": "quote rows given by the caller",
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


def _digest(paths: Sequence[Path]) -> str:
    """
    Return the SHA-256 of the files' names and bytes, in the given order.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
