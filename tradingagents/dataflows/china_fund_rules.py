"""Mainland China off-exchange mutual fund symbol rules and prompt context."""

from __future__ import annotations

from collections.abc import Mapping
import re

_EXPLICIT_FUND_RE = re.compile(
    r"^(?:FUND:(?P<prefix_code>\d{6})|(?P<suffix_code>\d{6})\.FUND)$",
    re.IGNORECASE,
)


def _parse_code(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise ValueError(f"{symbol!r} is not a supported China mutual fund symbol")
    normalized = symbol.strip().upper()
    match = _EXPLICIT_FUND_RE.fullmatch(normalized)
    if match:
        return match.group("prefix_code") or match.group("suffix_code")

    # Bare codes are intentionally conservative to avoid stealing A-share
    # stocks like 000001/002475 and exchange-traded funds like 510300.SS.
    if re.fullmatch(r"00[4-9]\d{3}", normalized):
        return normalized

    raise ValueError(f"{symbol!r} is not a supported China mutual fund symbol")


def is_china_fund_symbol(symbol: str) -> bool:
    try:
        _parse_code(symbol)
    except ValueError:
        return False
    return True


def extract_china_fund_code(symbol: str) -> str:
    return _parse_code(symbol)


def normalize_china_fund_symbol(symbol: str) -> str:
    return f"{_parse_code(symbol)}.FUND"


def build_china_fund_rule_context(
    symbol: str,
    trade_date: str | None = None,
    identity: Mapping[str, str] | None = None,
) -> str:
    normalized = normalize_china_fund_symbol(symbol)
    name = (identity or {}).get("fund_name") or (identity or {}).get("name")
    name_part = f" Fund/name context: {name}." if name else ""
    date_part = f" Analysis date: {trade_date}." if trade_date else ""
    return (
        f"China mutual fund constraints for `{normalized}`."
        f"{name_part}{date_part} Treat it as an off-exchange public fund, "
        "not an intraday exchange-traded stock. Use end-of-day NAV/unit-NAV data "
        "rather than intraday OHLCV assumptions. Do not apply A-share stock rules "
        "such as board lots, limit-up/limit-down executability, dragon-tiger lists, "
        "or stock exchange auction sessions. Focus on fund type, manager, fees, "
        "subscription/redemption restrictions, holdings, asset allocation, NAV "
        "drawdown, benchmark-relative performance, and concentration risk."
    )
