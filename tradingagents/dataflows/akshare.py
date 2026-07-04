"""AkShare data vendor for mainland China A-shares.

The AkShare package is intentionally imported lazily so the base project still
works without the optional China-market dependency installed.
"""

from __future__ import annotations

import importlib
from datetime import datetime
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .a_share_rules import is_a_share_symbol, normalize_a_share_symbol
from .errors import NoMarketDataError, VendorNotConfiguredError

_OHLCV_COLUMNS = {
    "\u65e5\u671f": "Date",
    "\u5f00\u76d8": "Open",
    "\u6536\u76d8": "Close",
    "\u6700\u9ad8": "High",
    "\u6700\u4f4e": "Low",
    "\u6210\u4ea4\u91cf": "Volume",
    "\u6210\u4ea4\u989d": "Amount",
    "\u632f\u5e45": "Amplitude",
    "\u6da8\u8dcc\u5e45": "Pct Change",
    "\u6da8\u8dcc\u989d": "Change",
    "\u6362\u624b\u7387": "Turnover",
    "date": "Date",
    "open": "Open",
    "close": "Close",
    "high": "High",
    "low": "Low",
    "volume": "Volume",
    "amount": "Amount",
    "turnover": "Turnover",
}

_SUPPORTED_INDICATORS = {
    "close_50_sma": "50 SMA trend benchmark.",
    "close_200_sma": "200 SMA long-term trend benchmark.",
    "close_10_ema": "10 EMA short-term momentum benchmark.",
    "macd": "MACD momentum line.",
    "macds": "MACD signal line.",
    "macdh": "MACD histogram.",
    "rsi": "RSI momentum oscillator.",
    "boll": "Bollinger middle band.",
    "boll_ub": "Bollinger upper band.",
    "boll_lb": "Bollinger lower band.",
    "atr": "Average true range volatility.",
    "vwma": "Volume-weighted moving average.",
    "mfi": "Money flow index.",
}


def _akshare():
    try:
        ak = importlib.import_module("akshare")
    except ImportError as exc:
        raise VendorNotConfiguredError(
            "akshare is not installed. Install it with `pip install akshare` "
            "or `pip install tradingagents[a-share]`."
        ) from exc
    if ak is None:
        raise VendorNotConfiguredError(
            "akshare is not installed. Install it with `pip install akshare` "
            "or `pip install tradingagents[a-share]`."
        )
    return ak


def _canonical_and_code(symbol: str) -> tuple[str, str]:
    if not is_a_share_symbol(symbol):
        raise NoMarketDataError(symbol, symbol, "not a supported A-share symbol")
    canonical = normalize_a_share_symbol(symbol)
    return canonical, canonical.split(".", 1)[0]


def _em_symbol(canonical: str) -> str:
    code, suffix = canonical.split(".", 1)
    prefix = {"SS": "SH", "SZ": "SZ", "BJ": "BJ"}[suffix]
    return f"{prefix}{code}"


def _market_symbol(canonical: str) -> str:
    code, suffix = canonical.split(".", 1)
    prefix = {"SS": "sh", "SZ": "sz", "BJ": "bj"}[suffix]
    return f"{prefix}{code}"


def _yyyymmdd(date_value: str) -> str:
    return datetime.strptime(date_value, "%Y-%m-%d").strftime("%Y%m%d")


def _non_empty_frame(data, symbol: str, canonical: str, detail: str) -> pd.DataFrame:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if frame.empty:
        raise NoMarketDataError(symbol, canonical, detail)
    return frame.copy()


def _format_frame(title: str, frame: pd.DataFrame, *, index: bool = False) -> str:
    header = f"{title}\n"
    header += f"# Total records: {len(frame)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + frame.to_csv(index=index)


def _call_endpoint(endpoint, **kwargs):
    try:
        return endpoint(**kwargs, timeout=10)
    except TypeError:
        return endpoint(**kwargs)


def _stock_history_frame(symbol: str, start_date: str, end_date: str) -> tuple[str, pd.DataFrame]:
    canonical, code = _canonical_and_code(symbol)
    ak = _akshare()
    start_yyyymmdd = _yyyymmdd(start_date)
    end_yyyymmdd = _yyyymmdd(end_date)
    market_symbol = _market_symbol(canonical)

    def tencent_hist():
        return ak.stock_zh_a_hist_tx(
            symbol=market_symbol,
            start_date=start_yyyymmdd,
            end_date=end_yyyymmdd,
            adjust="",
            timeout=10,
        )

    def sina_daily():
        return ak.stock_zh_a_daily(
            symbol=market_symbol,
            start_date=start_yyyymmdd,
            end_date=end_yyyymmdd,
            adjust="",
        )

    def eastmoney_hist():
        try:
            return ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_yyyymmdd,
                end_date=end_yyyymmdd,
                adjust="",
                timeout=10,
            )
        except TypeError:
            return ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_yyyymmdd,
                end_date=end_yyyymmdd,
                adjust="",
            )

    last_error: Exception | None = None
    last_no_data: NoMarketDataError | None = None
    frame = None
    for endpoint_name, fetch in (
        ("stock_zh_a_daily", sina_daily),
        ("stock_zh_a_hist_tx", tencent_hist),
        ("stock_zh_a_hist", eastmoney_hist),
    ):
        if not hasattr(ak, endpoint_name):
            continue
        try:
            candidate = _non_empty_frame(
                fetch(),
                symbol,
                canonical,
                f"{endpoint_name} returned no rows between {start_date} and {end_date}",
            )
        except NoMarketDataError as exc:
            last_no_data = exc
            continue
        except Exception as exc:
            last_error = exc
            continue
        frame = candidate
        break

    if frame is None:
        if last_no_data is not None and last_error is None:
            raise last_no_data
        if last_error is not None:
            raise last_error
        raise NoMarketDataError(symbol, canonical, f"no rows between {start_date} and {end_date}")

    frame = frame.rename(columns=_OHLCV_COLUMNS)
    if "Date" not in frame.columns:
        raise NoMarketDataError(symbol, canonical, "AkShare returned no date column")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["Date"])
    for col in ("Open", "High", "Low", "Close", "Amount", "Pct Change", "Change", "Turnover"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").round(4)
    if "Volume" in frame.columns:
        frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    if frame.empty:
        raise NoMarketDataError(symbol, canonical, "AkShare returned no usable OHLCV rows")
    return canonical, frame


def get_stock(
    symbol: Annotated[str, "A-share ticker symbol, e.g. 600519 or 600519.SS"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    canonical, frame = _stock_history_frame(symbol, start_date, end_date)
    return _format_frame(f"# AkShare stock data for {canonical} from {start_date} to {end_date}", frame)


def get_indicator(
    symbol: Annotated[str, "A-share ticker symbol"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "The current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    if indicator not in _SUPPORTED_INDICATORS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(_SUPPORTED_INDICATORS)}"
        )

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=max(int(look_back_days) + 260, 365))
    canonical, frame = _stock_history_frame(symbol, start_dt.strftime("%Y-%m-%d"), curr_date)
    stats_frame = frame.rename(columns=str.lower)
    stats_frame = stats_frame.rename(columns={"date": "Date"})
    stats_frame["Date"] = pd.to_datetime(stats_frame["Date"], errors="coerce")
    stats_frame = stats_frame.dropna(subset=["Date"])
    df = wrap(stats_frame)
    df[indicator]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    before = end_dt - relativedelta(days=look_back_days)
    values = []
    current = end_dt
    while current >= before:
        date_str = current.strftime("%Y-%m-%d")
        matching = df[df["Date"] == date_str]
        if matching.empty:
            value = "N/A: Not a trading day or no AkShare row"
        else:
            raw_value = matching[indicator].iloc[0]
            value = "N/A" if pd.isna(raw_value) else str(raw_value)
        values.append(f"{date_str}: {value}")
        current -= relativedelta(days=1)

    return (
        f"## AkShare {indicator} values for {canonical} from "
        f"{before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + "\n".join(values)
        + "\n\n"
        + _SUPPORTED_INDICATORS[indicator]
    )


def get_fundamentals(
    ticker: Annotated[str, "A-share ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    canonical, code = _canonical_and_code(ticker)
    ak = _akshare()
    sections: list[str] = []
    errors: list[Exception] = []

    for endpoint_name, kwargs in (
        ("stock_individual_info_em", {"symbol": code}),
        ("stock_profile_cninfo", {"symbol": code}),
        ("stock_info_a_code_name", {}),
    ):
        endpoint = getattr(ak, endpoint_name, None)
        if endpoint is None:
            continue
        try:
            info = _non_empty_frame(
                _call_endpoint(endpoint, **kwargs),
                ticker,
                canonical,
                f"{endpoint_name} returned no company profile",
            )
            if endpoint_name == "stock_info_a_code_name":
                info = _filter_code_name_frame(info, code)
            sections.append(f"## Company profile ({endpoint_name})\n" + info.to_csv(index=False))
            break
        except Exception as exc:
            errors.append(exc)
            continue

    start_year = _start_year_for_financial_indicator(curr_date)
    for endpoint_name, kwargs in (
        ("stock_financial_abstract", {"symbol": code}),
        ("stock_financial_abstract_ths", {"symbol": code}),
        ("stock_financial_analysis_indicator", {"symbol": code, "start_year": start_year}),
    ):
        endpoint = getattr(ak, endpoint_name, None)
        if endpoint is None:
            continue
        try:
            abstract = _non_empty_frame(
                _call_endpoint(endpoint, **kwargs),
                ticker,
                canonical,
                f"{endpoint_name} returned no financial abstract",
            )
            abstract = _filter_frame_by_date(abstract, curr_date)
            sections.append(f"## Financial abstract ({endpoint_name})\n" + abstract.to_csv(index=False))
            break
        except Exception as exc:
            errors.append(exc)
            continue

    if not sections:
        if errors:
            raise errors[0]
        raise VendorNotConfiguredError("akshare has no supported fundamentals endpoints")

    header = f"# AkShare fundamentals for {canonical}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + "\n".join(sections)


def _filter_code_name_frame(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if "code" in frame.columns:
        filtered = frame[frame["code"].astype(str).str.zfill(6) == code]
        return filtered if not filtered.empty else frame
    return frame


def _start_year_for_financial_indicator(curr_date: str | None) -> str:
    if not curr_date:
        return "1900"
    parsed = pd.to_datetime(curr_date, errors="coerce")
    if pd.isna(parsed):
        return "1900"
    return str(max(1900, int(parsed.year) - 3))


def _filter_frame_by_date(frame: pd.DataFrame, curr_date: str | None) -> pd.DataFrame:
    if not curr_date or frame.empty:
        return frame
    cutoff = pd.to_datetime(curr_date, errors="coerce")
    if pd.isna(cutoff):
        return frame
    for column in ("\u62a5\u544a\u671f", "\u516c\u544a\u65e5\u671f", "\u65e5\u671f"):
        if column in frame.columns:
            dates = pd.to_datetime(frame[column], errors="coerce")
            return frame[dates.isna() | (dates <= cutoff)]
    keep_columns = []
    saw_period_column = False
    for column in frame.columns:
        parsed = pd.to_datetime(str(column), format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            keep_columns.append(column)
            continue
        saw_period_column = True
        if parsed <= cutoff:
            keep_columns.append(column)
    if saw_period_column:
        return frame.loc[:, keep_columns]
    return frame


def _statement(
    ticker: str,
    statement_name: str,
    endpoint_name: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    canonical, code = _canonical_and_code(ticker)
    endpoint = getattr(_akshare(), endpoint_name, None)
    if endpoint is None:
        raise VendorNotConfiguredError(f"akshare endpoint {endpoint_name} is unavailable")

    last_error: Exception | None = None
    for candidate in (_em_symbol(canonical), code):
        try:
            frame = endpoint(symbol=candidate)
            frame = _non_empty_frame(frame, ticker, canonical, f"no {statement_name} data")
            frame = _filter_frame_by_date(frame, curr_date)
            if not frame.empty:
                title = f"# AkShare {statement_name} data for {canonical} ({freq})"
                return _format_frame(title, frame)
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise NoMarketDataError(ticker, canonical, f"no {statement_name} data")


def get_balance_sheet(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    return _statement(ticker, "balance sheet", "stock_balance_sheet_by_report_em", freq, curr_date)


def get_cashflow(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    return _statement(ticker, "cash flow", "stock_cash_flow_sheet_by_report_em", freq, curr_date)


def get_income_statement(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    return _statement(ticker, "income statement", "stock_profit_sheet_by_report_em", freq, curr_date)


def get_news(
    ticker: Annotated[str, "A-share ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    canonical, code = _canonical_and_code(ticker)
    endpoint = getattr(_akshare(), "stock_news_em", None)
    if endpoint is None:
        raise VendorNotConfiguredError("akshare endpoint stock_news_em is unavailable")
    frame = _non_empty_frame(endpoint(symbol=code), ticker, canonical, "no stock news")
    frame = _filter_news_by_date(frame, start_date, end_date)
    if frame.empty:
        raise NoMarketDataError(ticker, canonical, f"no news between {start_date} and {end_date}")
    return _format_frame(f"# AkShare news for {canonical} from {start_date} to {end_date}", frame)


def _filter_news_by_date(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    for column in ("\u53d1\u5e03\u65f6\u95f4", "\u65e5\u671f", "date"):
        if column not in frame.columns:
            continue
        dates = pd.to_datetime(frame[column], errors="coerce")
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce") + pd.Timedelta(days=1)
        if pd.isna(start) or pd.isna(end):
            return frame
        return frame[dates.isna() | ((dates >= start) & (dates < end))]
    return frame
