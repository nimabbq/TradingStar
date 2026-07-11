"""AkShare-backed macroeconomic series for mainland China analysis."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .akshare import _akshare, _format_frame
from .errors import NoMarketDataError, VendorNotConfiguredError


CHINA_MACRO_SERIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "cpi": ("CPI", ("macro_china_cpi", "macro_china_cpi_yearly")),
    "ppi": ("PPI", ("macro_china_ppi", "macro_china_ppi_yearly")),
    "pmi": ("Manufacturing PMI", ("macro_china_pmi", "macro_china_pmi_yearly")),
    "non_manufacturing_pmi": ("Non-manufacturing PMI", ("macro_china_non_man_pmi",)),
    "lpr": ("Loan Prime Rate", ("macro_china_lpr",)),
    "money_supply": ("Money supply", ("macro_china_money_supply", "macro_china_supply_of_money")),
    "m2": ("M2 money supply", ("macro_china_money_supply", "macro_china_m2_yearly")),
    "social_financing": ("Aggregate financing to the real economy", ("macro_china_shrzgm",)),
    "new_credit": ("New RMB loans", ("macro_china_new_financial_credit",)),
    "rrr": ("Reserve requirement ratio", ("macro_china_reserve_requirement_ratio",)),
    "gdp": ("GDP", ("macro_china_gdp", "macro_china_gdp_yearly")),
    "unemployment": ("Urban unemployment", ("macro_china_urban_unemployment",)),
    "fx_reserves": ("Foreign-exchange reserves", ("macro_china_fx_reserves_yearly", "macro_china_fx_gold")),
}

_ALIASES = {
    "manufacturing_pmi": "pmi",
    "non_man_pmi": "non_manufacturing_pmi",
    "social_finance": "social_financing",
    "total_social_financing": "social_financing",
    "required_reserve_ratio": "rrr",
    "urban_unemployment": "unemployment",
}

_MONTHLY_RELEASE_LAG_KEYS = {
    "cpi",
    "ppi",
    "pmi",
    "non_manufacturing_pmi",
    "money_supply",
    "m2",
    "social_financing",
    "new_credit",
    "unemployment",
    "fx_reserves",
}

_DATE_COLUMNS = (
    "\u65e5\u671f",
    "\u6708\u4efd",
    "\u7edf\u8ba1\u65f6\u95f4",
    "\u62a5\u544a\u65e5\u671f",
    "\u65f6\u95f4",
    "date",
    "Date",
)


def _resolve_indicator(indicator: str) -> str | None:
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    return key if key in CHINA_MACRO_SERIES else None


def _parse_macro_dates(values: pd.Series) -> pd.Series:
    normalized = (
        values.astype(str)
        .str.strip()
        .str.replace("\u5e74", "-", regex=False)
        .str.replace("\u6708\u4efd", "", regex=False)
        .str.replace("\u6708", "", regex=False)
        .str.replace("\u65e5", "", regex=False)
    )
    normalized = normalized.str.replace(r"^(\d{4})-(\d{1,2})$", r"\1-\2-01", regex=True)
    return pd.to_datetime(normalized, errors="coerce")


def _filter_macro_frame(
    frame: pd.DataFrame,
    indicator_key: str,
    curr_date: str,
    look_back_days: int | None,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(curr_date, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError(f"Invalid China macro analysis date: {curr_date!r}")
    date_column = next((column for column in _DATE_COLUMNS if column in frame.columns), None)
    if date_column is None:
        return frame.tail(40).copy()
    dates = _parse_macro_dates(frame[date_column])
    if indicator_key in _MONTHLY_RELEASE_LAG_KEYS:
        visible_dates = dates + pd.offsets.MonthEnd(0) + pd.Timedelta(days=20)
    elif indicator_key == "gdp":
        visible_dates = dates + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=35)
    else:
        visible_dates = dates
    mask = dates.notna() & visible_dates.notna() & (visible_dates <= cutoff)
    if look_back_days is not None:
        mask &= dates >= cutoff - timedelta(days=max(1, int(look_back_days)))
    filtered = frame[mask].copy()
    filtered["AsOfDate"] = dates[mask].dt.strftime("%Y-%m-%d")
    filtered["ConservativeVisibleFrom"] = visible_dates[mask].dt.strftime("%Y-%m-%d")
    return filtered.sort_values("AsOfDate").tail(40)


def get_china_macro_data(
    indicator: str,
    curr_date: str,
    look_back_days: int | None = None,
) -> str:
    key = _resolve_indicator(indicator)
    if key is None:
        supported = ", ".join(sorted(CHINA_MACRO_SERIES))
        return (
            f"DATA_UNAVAILABLE: unknown China macro indicator {indicator!r}. "
            f"Supported China macro indicators: {supported}."
        )

    title, endpoint_names = CHINA_MACRO_SERIES[key]
    ak = _akshare()
    saw_endpoint = False
    last_error: Exception | None = None
    for endpoint_name in endpoint_names:
        endpoint = getattr(ak, endpoint_name, None)
        if endpoint is None:
            continue
        saw_endpoint = True
        try:
            frame = endpoint()
            frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            if frame.empty:
                continue
            frame = _filter_macro_frame(frame, key, curr_date, look_back_days)
            if frame.empty:
                continue
            return _format_frame(
                f"# China macro: {title}\n"
                f"# Indicator key: {key}\n# Source endpoint: {endpoint_name}\n"
                f"# Point-in-time cutoff: {curr_date}\n"
                "# Period rows use a conservative publication lag when no release timestamp is supplied",
                frame,
            )
        except Exception as exc:  # noqa: BLE001 - continue through documented fallbacks
            last_error = exc
            continue
    if not saw_endpoint:
        raise VendorNotConfiguredError(
            f"AkShare exposes none of the China macro endpoints for {key}: {endpoint_names}"
        )
    if last_error is not None:
        raise last_error
    raise NoMarketDataError(indicator, detail=f"no China macro rows on or before {curr_date}")


__all__ = ["CHINA_MACRO_SERIES", "get_china_macro_data"]
