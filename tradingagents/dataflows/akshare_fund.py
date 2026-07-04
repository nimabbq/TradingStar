"""AkShare data vendor for mainland China off-exchange mutual funds."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .akshare import _akshare, _call_endpoint, _format_frame, _non_empty_frame
from .china_fund_rules import extract_china_fund_code, normalize_china_fund_symbol
from .errors import NoMarketDataError

_SUPPORTED_INDICATORS = {
    "close_50_sma": "50 SMA based on unit NAV.",
    "close_200_sma": "200 SMA based on unit NAV.",
    "close_10_ema": "10 EMA based on unit NAV.",
    "macd": "MACD momentum line based on unit NAV.",
    "macds": "MACD signal line based on unit NAV.",
    "macdh": "MACD histogram based on unit NAV.",
    "rsi": "RSI momentum oscillator based on unit NAV.",
    "boll": "Bollinger middle band based on unit NAV.",
    "boll_ub": "Bollinger upper band based on unit NAV.",
    "boll_lb": "Bollinger lower band based on unit NAV.",
    "atr": "ATR is approximate for NAV data because funds have no intraday high/low.",
    "vwma": "VWMA is approximate for NAV data because off-exchange funds have no volume.",
    "mfi": "MFI is approximate for NAV data because off-exchange funds have no volume.",
}


def _date_range_filter(
    frame: pd.DataFrame,
    date_column: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    mask = dates.notna()
    if start_date:
        start = pd.to_datetime(start_date, errors="coerce")
        if not pd.isna(start):
            mask = mask & (dates >= start)
    if end_date:
        end = pd.to_datetime(end_date, errors="coerce")
        if not pd.isna(end):
            mask = mask & (dates <= end)
    return frame[mask].copy()


def _fund_nav_frame(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[str, pd.DataFrame]:
    canonical = normalize_china_fund_symbol(symbol)
    code = extract_china_fund_code(symbol)
    frame = _akshare().fund_open_fund_info_em(
        symbol=code,
        indicator="\u5355\u4f4d\u51c0\u503c\u8d70\u52bf",
        period="\u5168\u90e8",
    )
    frame = _non_empty_frame(frame, symbol, canonical, "no fund NAV rows")
    frame = frame.rename(
        columns={
            "\u51c0\u503c\u65e5\u671f": "Date",
            "\u5355\u4f4d\u51c0\u503c": "Unit NAV",
            "\u7d2f\u8ba1\u51c0\u503c": "Accumulated NAV",
            "\u65e5\u589e\u957f\u7387": "Daily Return %",
        }
    )
    if "Date" not in frame.columns or "Unit NAV" not in frame.columns:
        raise NoMarketDataError(symbol, canonical, "AkShare returned no fund NAV/date columns")
    frame = _date_range_filter(frame, "Date", start_date, end_date)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])
    frame["Unit NAV"] = pd.to_numeric(frame["Unit NAV"], errors="coerce")
    frame = frame.dropna(subset=["Unit NAV"])
    if "Accumulated NAV" in frame.columns:
        frame["Accumulated NAV"] = pd.to_numeric(frame["Accumulated NAV"], errors="coerce")
    if "Daily Return %" in frame.columns:
        frame["Daily Return %"] = pd.to_numeric(frame["Daily Return %"], errors="coerce")

    # Give stockstats an OHLCV-shaped frame while keeping the true NAV columns.
    frame["Open"] = frame["Unit NAV"]
    frame["High"] = frame["Unit NAV"]
    frame["Low"] = frame["Unit NAV"]
    frame["Close"] = frame["Unit NAV"]
    frame["Volume"] = 0.0
    frame = frame.sort_values("Date")
    if frame.empty:
        detail = f"no fund NAV rows between {start_date} and {end_date}"
        raise NoMarketDataError(symbol, canonical, detail)
    return canonical, frame


def get_stock(
    symbol: Annotated[str, "China mutual fund code, e.g. 005827 or 005827.FUND"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    canonical, frame = _fund_nav_frame(symbol, start_date, end_date)
    out = frame.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return _format_frame(
        f"# AkShare fund NAV data for {canonical} from {start_date} to {end_date}",
        out,
    )


def get_indicator(
    symbol: Annotated[str, "China mutual fund code"],
    indicator: Annotated[str, "technical indicator to calculate from unit NAV"],
    curr_date: Annotated[str, "The current analysis date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    if indicator not in _SUPPORTED_INDICATORS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(_SUPPORTED_INDICATORS)}"
        )

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=max(int(look_back_days) + 260, 365))
    canonical, frame = _fund_nav_frame(symbol, start_dt.strftime("%Y-%m-%d"), curr_date)
    stats_frame = frame[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = wrap(stats_frame.rename(columns=str.lower).rename(columns={"date": "Date"}))
    df[indicator]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    before = end_dt - relativedelta(days=look_back_days)
    values = []
    current = end_dt
    while current >= before:
        date_str = current.strftime("%Y-%m-%d")
        matching = df[df["Date"] == date_str]
        if matching.empty:
            value = "N/A: No NAV row for this date"
        else:
            raw_value = matching[indicator].iloc[0]
            value = "N/A" if pd.isna(raw_value) else str(raw_value)
        values.append(f"{date_str}: {value}")
        current -= relativedelta(days=1)

    return (
        f"## AkShare fund {indicator} values for {canonical} from "
        f"{before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + "\n".join(values)
        + "\n\n"
        + _SUPPORTED_INDICATORS[indicator]
    )


def _year_candidates(curr_date: str | None) -> list[str]:
    parsed = pd.to_datetime(curr_date, errors="coerce") if curr_date else pd.Timestamp.today()
    if pd.isna(parsed):
        parsed = pd.Timestamp.today()
    year = int(parsed.year)
    return [str(year - offset) for offset in range(0, 4)]


def _append_first_non_empty_section(
    sections: list[str],
    title: str,
    endpoint_name: str,
    code: str,
    years: list[str],
) -> None:
    endpoint = getattr(_akshare(), endpoint_name, None)
    if endpoint is None:
        return
    for year in years:
        try:
            frame = _non_empty_frame(
                _call_endpoint(endpoint, symbol=code, date=year),
                code,
                f"{code}.FUND",
                f"no {title} rows for {year}",
            )
        except Exception:
            continue
        sections.append(f"## {title} ({year}, {endpoint_name})\n" + frame.to_csv(index=False))
        return


def get_fundamentals(
    ticker: Annotated[str, "China mutual fund code"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    canonical = normalize_china_fund_symbol(ticker)
    code = extract_china_fund_code(ticker)
    ak = _akshare()
    sections: list[str] = []

    for endpoint_name, title in (
        ("fund_individual_basic_info_xq", "Fund profile"),
        ("fund_individual_detail_info_xq", "Fees and trading rules"),
    ):
        endpoint = getattr(ak, endpoint_name, None)
        if endpoint is None:
            continue
        try:
            frame = _non_empty_frame(
                _call_endpoint(endpoint, symbol=code),
                ticker,
                canonical,
                f"{endpoint_name} returned no rows",
            )
        except Exception:
            continue
        sections.append(f"## {title} ({endpoint_name})\n" + frame.to_csv(index=False))

    years = _year_candidates(curr_date)
    _append_first_non_empty_section(sections, "Stock holdings", "fund_portfolio_hold_em", code, years)
    _append_first_non_empty_section(
        sections,
        "Bond holdings",
        "fund_portfolio_bond_hold_em",
        code,
        years,
    )
    _append_first_non_empty_section(
        sections,
        "Industry allocation",
        "fund_portfolio_industry_allocation_em",
        code,
        years,
    )

    if not sections:
        raise NoMarketDataError(ticker, canonical, "no fund fundamentals or holdings")

    header = f"# AkShare fund fundamentals for {canonical}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + "\n".join(sections)


def get_statement_not_applicable(
    ticker: Annotated[str, "China mutual fund code"],
    *args,
    **kwargs,
) -> str:
    canonical = normalize_china_fund_symbol(ticker)
    return (
        f"DATA_UNAVAILABLE: {canonical} is a China mutual fund. Company-style "
        "balance sheet, cash-flow, and income-statement tools are not applicable; "
        "use get_fundamentals for fund profile, fees, holdings, and allocation."
    )


def get_news(
    ticker: Annotated[str, "China mutual fund code"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    canonical = normalize_china_fund_symbol(ticker)
    return (
        f"DATA_UNAVAILABLE: AkShare does not expose a reliable fund-specific news "
        f"feed for {canonical} between {start_date} and {end_date}. Use fund NAV, "
        "holdings, manager, fees, and broad market context instead; do not fabricate news."
    )
