"""Symbol and trading context for mainland China exchange-traded funds."""

from __future__ import annotations

from collections.abc import Mapping
import re


_ETF_RE = re.compile(r"^(?P<code>\d{6})(?P<suffix>\.(SS|SH|SZ))?$", re.IGNORECASE)


def _parse_code(symbol: str) -> tuple[str, str | None]:
    if not isinstance(symbol, str):
        raise ValueError(f"{symbol!r} is not a supported China ETF symbol")
    match = _ETF_RE.fullmatch(symbol.strip().upper())
    if not match:
        raise ValueError(f"{symbol!r} is not a supported China ETF symbol")
    code, suffix = match.group("code"), match.group("suffix")
    if code.startswith(("51", "56", "58")):
        expected = ".SS"
    elif code.startswith("15"):
        expected = ".SZ"
    else:
        raise ValueError(f"{symbol!r} is not a supported China ETF symbol")
    if suffix and ({".SH": ".SS"}.get(suffix, suffix) != expected):
        raise ValueError(f"{symbol!r} has an ETF code/exchange mismatch")
    return code, expected


def is_china_etf_symbol(symbol: str) -> bool:
    try:
        _parse_code(symbol)
    except ValueError:
        return False
    return True


def normalize_china_etf_symbol(symbol: str) -> str:
    code, suffix = _parse_code(symbol)
    return f"{code}{suffix}"


def extract_china_etf_code(symbol: str) -> str:
    return _parse_code(symbol)[0]


def build_china_etf_rule_context(
    symbol: str,
    identity: Mapping[str, str] | None = None,
) -> str:
    canonical = normalize_china_etf_symbol(symbol)
    name = (identity or {}).get("company_name") or (identity or {}).get("name")
    name_text = f" Resolved fund name: {name}." if name else ""
    return (
        f"China exchange-traded fund context for `{canonical}`.{name_text} "
        "Treat this as an exchange-traded fund, not an operating company or an off-exchange mutual fund. "
        "Use exchange OHLCV for execution and forward-adjusted OHLCV for technical continuity. "
        "Analyze index exposure, holdings, liquidity, premium/discount, tracking error, fund size, and share changes; "
        "company financial statements are not applicable. Standard buy orders are generally in 100-unit lots. "
        "Settlement can be T+0 or T+1 depending on the underlying asset class, so do not infer it from the ticker alone."
    )


__all__ = [
    "build_china_etf_rule_context",
    "extract_china_etf_code",
    "is_china_etf_symbol",
    "normalize_china_etf_symbol",
]
