"""
Exact parsing of quote rows into integer cents, half cents and UTC nanoseconds.

Prices are split as text and never converted through float.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_SECOND = timedelta(seconds=1)
_PRICE = re.compile(r"([0-9]+)(?:\.([0-9]{1,2}))?")
_TIMESTAMP = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?Z",
)


@dataclass(frozen=True)
class ParsedQuote:
    """
    A quote with exact integer prices and a UTC timestamp in nanoseconds.
    """

    ordinal: int
    timestamp_ns: int
    bid_cents: int
    ask_cents: int
    bid_half_cents: int
    mid_half_cents: int
    ask_half_cents: int


@dataclass(frozen=True)
class QuoteDiagnostics:
    """
    Counts of rows skipped because their bid or ask was missing.
    """

    skipped_rows: int
    missing_bid: int
    missing_ask: int


def parse_quote_row(row: Mapping[str, object]) -> ParsedQuote:
    """
    Parse one quote row. The bid and the ask must both be present.
    """
    bid_cents = _parse_price("bid", row["bid"])
    ask_cents = _parse_price("ask", row["ask"])
    if bid_cents is None or ask_cents is None:
        raise ValueError(
            "quote row has a missing bid or ask; use parse_quote_rows to skip it"
        )
    return _build_quote(row, bid_cents, ask_cents)


def parse_quote_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[ParsedQuote], QuoteDiagnostics]:
    """
    Parse quote rows. Rows with a missing bid or ask are skipped and counted.
    """
    parsed: list[ParsedQuote] = []
    skipped_rows = 0
    missing_bid = 0
    missing_ask = 0
    for row in rows:
        bid_cents = _parse_price("bid", row["bid"])
        ask_cents = _parse_price("ask", row["ask"])
        if bid_cents is None or ask_cents is None:
            # A skipped row must still have a valid ordinal and timestamp
            _row_identity(row)
            skipped_rows += 1
            missing_bid += int(bid_cents is None)
            missing_ask += int(ask_cents is None)
        else:
            parsed.append(_build_quote(row, bid_cents, ask_cents))
    return parsed, QuoteDiagnostics(skipped_rows, missing_bid, missing_ask)


def _parse_price(field: str, text: object) -> int | None:
    """
    Return the price in cents, or None when the text is missing (None or empty).

    Accepts 0 to 2 decimals (for example 12345.6 is 1234560 cents). Rejects more decimals,
    signs, spaces, exponents and any other form, so no value is ever rounded.
    """
    if text is None or text == "":
        return None
    if type(text) is not str:
        raise TypeError(f"{field} must be text, got {type(text).__name__}")
    match = _PRICE.fullmatch(text)
    if match is None:
        raise ValueError(
            f"{field} must be a price with at most 2 decimals, got {text!r}"
        )
    whole, fraction = match.groups()
    return int(whole) * 100 + int((fraction or "").ljust(2, "0"))


def _row_identity(row: Mapping[str, object]) -> tuple[int, int]:
    """
    Return the row's ordinal and UTC nanoseconds, or raise when either is malformed.
    """
    ordinal = row["ordinal"]
    if type(ordinal) is not int:
        raise TypeError(f"ordinal must be an int, got {type(ordinal).__name__}")
    return ordinal, _utc_nanoseconds(row["timestamp"])


def _build_quote(
    row: Mapping[str, object], bid_cents: int, ask_cents: int
) -> ParsedQuote:
    ordinal, timestamp_ns = _row_identity(row)
    return ParsedQuote(
        ordinal=ordinal,
        timestamp_ns=timestamp_ns,
        bid_cents=bid_cents,
        ask_cents=ask_cents,
        bid_half_cents=2 * bid_cents,
        mid_half_cents=bid_cents + ask_cents,
        ask_half_cents=2 * ask_cents,
    )


def _utc_nanoseconds(text: object) -> int:
    """
    Convert UTC text such as 2026-08-18T03:20:00Z (fraction up to 9 digits) exactly.
    """
    if type(text) is not str:
        raise TypeError(f"timestamp must be text, got {type(text).__name__}")
    match = _TIMESTAMP.fullmatch(text)
    if match is None:
        raise ValueError(f"timestamp must be UTC text ending in Z, got {text!r}")
    *clock, fraction = match.groups()
    moment = datetime(*map(int, clock), tzinfo=UTC)
    whole_seconds = (moment - _EPOCH) // _ONE_SECOND
    return whole_seconds * 1_000_000_000 + int((fraction or "").ljust(9, "0"))
