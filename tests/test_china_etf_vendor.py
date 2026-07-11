import sys

import pandas as pd
import pytest

from cli.models import AssetType
from cli.utils import detect_asset_type, normalize_ticker_symbol
from tradingagents.dataflows import interface
from tradingagents.dataflows.china_etf_rules import (
    build_china_etf_rule_context,
    is_china_etf_symbol,
    normalize_china_etf_symbol,
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [("510300", "510300.SS"), ("510300.SH", "510300.SS"), ("159915", "159915.SZ")],
)
def test_china_etf_symbol_normalization(raw, canonical):
    assert is_china_etf_symbol(raw)
    assert normalize_china_etf_symbol(raw) == canonical
    assert normalize_ticker_symbol(raw) == canonical
    assert detect_asset_type(raw) is AssetType.ETF


def test_china_etf_context_does_not_apply_company_rules():
    context = build_china_etf_rule_context("510300")
    assert "exchange-traded fund" in context
    assert "tracking error" in context
    assert "company financial statements" in context


class FakeETF:
    def fund_etf_hist_em(self, symbol, period, start_date, end_date, adjust):
        self.hist_call = {
            "symbol": symbol,
            "period": period,
            "start_date": start_date,
            "end_date": end_date,
            "adjust": adjust,
        }
        return pd.DataFrame(
            [{"日期": "2026-01-05", "开盘": 4.0, "收盘": 4.1, "最高": 4.2, "最低": 3.9, "成交量": 10000}]
        )

    def fund_etf_spot_em(self):
        return pd.DataFrame([{"代码": "510300", "名称": "沪深300ETF", "最新价": 4.1}])


@pytest.mark.unit
def test_etf_default_chain_uses_akshare(monkeypatch):
    fake = FakeETF()
    monkeypatch.setitem(sys.modules, "akshare", fake)

    result = interface.route_to_vendor("get_stock_data", "510300", "2026-01-01", "2026-01-10")

    assert fake.hist_call["adjust"] == ""
    assert "AkShare ETF stock data for 510300.SS" in result
    assert "Price basis: unadjusted" in result


@pytest.mark.unit
def test_etf_history_falls_back_to_sina_when_eastmoney_proxy_fails(monkeypatch):
    from tradingagents.dataflows.akshare_etf import get_ohlcv_frame

    class SinaFallback:
        def fund_etf_hist_em(self, **kwargs):
            raise RuntimeError("eastmoney proxy failed")

        def fund_etf_hist_sina(self, symbol):
            self.sina_symbol = symbol
            return pd.DataFrame(
                [{"date": "2026-01-05", "open": 4.0, "close": 4.1, "high": 4.2, "low": 3.9, "volume": 10000}]
            )

    fake = SinaFallback()
    monkeypatch.setitem(sys.modules, "akshare", fake)

    frame = get_ohlcv_frame("510300", "2026-01-01", "2026-01-10", adjust="qfq")

    assert fake.sina_symbol == "sh510300"
    assert frame.iloc[-1]["Close"] == 4.1
    assert "unadjusted" in frame.attrs["price_basis"]
