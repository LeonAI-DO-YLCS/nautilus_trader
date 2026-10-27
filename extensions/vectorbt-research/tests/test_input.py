import dataclasses

from vectorbt_research.input import parse_quote_row, parse_quote_rows


def test_quote_text_becomes_exact_integers_and_utc_nanoseconds_no_float():
    """quote text becomes exact integers and UTC nanoseconds, no float"""
    parsed = parse_quote_row(
        {
            "timestamp": "2026-08-18T03:20:00Z",
            "ordinal": 7,
            "bid": "12345.67",
            "ask": "12345.68",
        }
    )

    assert [field.name for field in dataclasses.fields(parsed)] == [
        "ordinal",
        "timestamp_ns",
        "bid_cents",
        "ask_cents",
        "bid_half_cents",
        "mid_half_cents",
        "ask_half_cents",
    ]
    assert parsed.ordinal == 7
    assert parsed.timestamp_ns == 1787023200000000000
    assert parsed.bid_cents == 1234567
    assert parsed.ask_cents == 1234568
    assert parsed.bid_half_cents == 2469134
    assert parsed.mid_half_cents == 2469135
    assert parsed.ask_half_cents == 2469136
    for field in dataclasses.fields(parsed):
        assert type(getattr(parsed, field.name)) is int, field.name


def test_rows_with_an_empty_bid_or_ask_are_skipped_and_counted():
    """rows with an empty bid or ask are skipped and counted"""
    parsed, diagnostics = parse_quote_rows(
        [
            {
                "timestamp": "2026-08-18T03:20:01Z",
                "ordinal": 60,
                "bid": None,
                "ask": "12345.68",
            },
            {
                "timestamp": "2026-08-18T03:20:02Z",
                "ordinal": 61,
                "bid": "12345.67",
                "ask": None,
            },
            {
                "timestamp": "2026-08-18T03:20:03Z",
                "ordinal": 62,
                "bid": None,
                "ask": None,
            },
        ]
    )

    assert parsed == []
    assert [field.name for field in dataclasses.fields(diagnostics)] == [
        "skipped_rows",
        "missing_bid",
        "missing_ask",
    ]
    assert diagnostics.skipped_rows == 3
    assert diagnostics.missing_bid == 2
    assert diagnostics.missing_ask == 2
    for field in dataclasses.fields(diagnostics):
        assert type(getattr(diagnostics, field.name)) is int, field.name
