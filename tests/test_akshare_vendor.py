import sys

import pandas as pd
import pytest

from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import VendorNotConfiguredError


class FakeAkShare:
    def stock_zh_a_hist(self, symbol, period, start_date, end_date, adjust):
        self.hist_call = {
            "symbol": symbol,
            "period": period,
            "start_date": start_date,
            "end_date": end_date,
            "adjust": adjust,
        }
        return pd.DataFrame(
            [
                {
                    "\u65e5\u671f": "2026-01-05",
                    "\u5f00\u76d8": 100.0,
                    "\u6536\u76d8": 105.5,
                    "\u6700\u9ad8": 106.0,
                    "\u6700\u4f4e": 99.0,
                    "\u6210\u4ea4\u91cf": 123456,
                    "\u6210\u4ea4\u989d": 987654321.0,
                    "\u6da8\u8dcc\u5e45": 5.5,
                    "\u6362\u624b\u7387": 1.2,
                }
            ]
        )

    def stock_individual_info_em(self, symbol):
        return pd.DataFrame(
            [
                {"item": "\u80a1\u7968\u7b80\u79f0", "value": "\u8d35\u5dde\u8305\u53f0"},
                {"item": "\u884c\u4e1a", "value": "\u9152\u7c7b\u884c\u4e1a"},
                {"item": "\u4e0a\u5e02\u65f6\u95f4", "value": "20010827"},
            ]
        )

    def stock_financial_abstract(self, symbol):
        return pd.DataFrame(
            [
                {
                    "\u62a5\u544a\u671f": "2025-09-30",
                    "\u51c0\u5229\u6da6": 1000,
                    "\u8425\u4e1a\u6536\u5165": 2000,
                }
            ]
        )

    def stock_news_em(self, symbol):
        return pd.DataFrame(
            [
                {
                    "\u5173\u952e\u8bcd": "600519",
                    "\u65b0\u95fb\u6807\u9898": "\u8d35\u5dde\u8305\u53f0\u53d1\u5e03\u516c\u544a",
                    "\u65b0\u95fb\u5185\u5bb9": "\u516c\u53f8\u7ecf\u8425\u7a33\u5065",
                    "\u53d1\u5e03\u65f6\u95f4": "2026-01-05 09:00:00",
                    "\u6587\u7ae0\u6765\u6e90": "\u4e1c\u65b9\u8d22\u5bcc",
                    "\u65b0\u95fb\u94fe\u63a5": "https://example.com/news",
                }
            ]
        )


class FakeAkShareWithBrokenEastmoney(FakeAkShare):
    def stock_zh_a_hist(self, symbol, period, start_date, end_date, adjust):
        raise RuntimeError("eastmoney proxy failed")

    def stock_zh_a_hist_tx(self, symbol, start_date, end_date, adjust, timeout=None):
        self.tx_hist_call = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "adjust": adjust,
            "timeout": timeout,
        }
        return pd.DataFrame(
            [
                {
                    "date": "2026-01-05",
                    "open": 100.0,
                    "close": 105.5,
                    "high": 106.0,
                    "low": 99.0,
                    "amount": 32156.0,
                }
            ]
        )


class FakeAkShareWithBrokenProfile(FakeAkShare):
    def stock_individual_info_em(self, symbol):
        raise RuntimeError("eastmoney profile failed")

    def stock_profile_cninfo(self, symbol):
        self.profile_call = {"symbol": symbol}
        return pd.DataFrame(
            [
                {
                    "\u516c\u53f8\u540d\u79f0": "\u8d35\u5dde\u8305\u53f0\u9152\u80a1\u4efd\u6709\u9650\u516c\u53f8",
                    "A\u80a1\u4ee3\u7801": symbol,
                    "A\u80a1\u7b80\u79f0": "\u8d35\u5dde\u8305\u53f0",
                }
            ]
        )


@pytest.fixture()
def fake_akshare(monkeypatch):
    fake = FakeAkShare()
    monkeypatch.setitem(sys.modules, "akshare", fake)
    return fake


@pytest.mark.unit
def test_akshare_registered_as_vendor():
    assert "akshare" in interface.VENDOR_LIST
    assert "akshare" in interface.VENDOR_METHODS["get_stock_data"]
    assert "akshare" in interface.VENDOR_METHODS["get_news"]


@pytest.mark.unit
def test_a_share_default_chain_prefers_akshare(fake_akshare):
    result = interface.route_to_vendor("get_stock_data", "600519", "2026-01-01", "2026-01-10")

    assert fake_akshare.hist_call == {
        "symbol": "600519",
        "period": "daily",
        "start_date": "20260101",
        "end_date": "20260110",
        "adjust": "",
    }
    assert "# AkShare stock data for 600519.SS" in result
    assert "2026-01-05" in result
    assert "105.5" in result


@pytest.mark.unit
def test_stock_data_uses_akshare_tencent_fallback_when_eastmoney_fails(monkeypatch):
    fake = FakeAkShareWithBrokenEastmoney()
    monkeypatch.setitem(sys.modules, "akshare", fake)

    result = interface.route_to_vendor("get_stock_data", "600519", "2026-01-01", "2026-01-10")

    assert fake.tx_hist_call == {
        "symbol": "sh600519",
        "start_date": "20260101",
        "end_date": "20260110",
        "adjust": "",
        "timeout": 10,
    }
    assert "# AkShare stock data for 600519.SS" in result
    assert "105.5" in result


@pytest.mark.unit
def test_non_a_share_default_chain_keeps_yfinance(monkeypatch):
    calls = []

    def yfinance(symbol, *args):
        calls.append(symbol)
        return "YF"

    def akshare(symbol, *args):
        raise AssertionError("akshare should not be called for non-A-share default routing")

    with monkeypatch.context() as m:
        m.setitem(interface.VENDOR_METHODS["get_stock_data"], "yfinance", yfinance)
        m.setitem(interface.VENDOR_METHODS["get_stock_data"], "akshare", akshare)
        result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")

    assert result == "YF"
    assert calls == ["AAPL"]


@pytest.mark.unit
def test_a_share_default_chain_falls_back_when_akshare_errors(monkeypatch):
    calls = []

    def akshare(symbol, *args):
        calls.append(("akshare", symbol))
        raise RuntimeError("proxy failed")

    def yfinance(symbol, *args):
        calls.append(("yfinance", symbol))
        return "YF"

    with monkeypatch.context() as m:
        m.setitem(interface.VENDOR_METHODS["get_stock_data"], "akshare", akshare)
        m.setitem(interface.VENDOR_METHODS["get_stock_data"], "yfinance", yfinance)
        result = interface.route_to_vendor("get_stock_data", "600519", "2026-01-01", "2026-01-10")

    assert result == "YF"
    assert calls == [("akshare", "600519"), ("yfinance", "600519")]


@pytest.mark.unit
def test_explicit_tool_vendor_config_still_wins_for_a_share(monkeypatch, fake_akshare):
    calls = []

    def yfinance(symbol, *args):
        calls.append(symbol)
        return "YF"

    set_config({"tool_vendors": {"get_stock_data": "yfinance"}})
    with monkeypatch.context() as m:
        m.setitem(interface.VENDOR_METHODS["get_stock_data"], "yfinance", yfinance)
        result = interface.route_to_vendor("get_stock_data", "600519", "2026-01-01", "2026-01-10")

    assert result == "YF"
    assert calls == ["600519"]


@pytest.mark.unit
def test_missing_akshare_raises_not_configured(monkeypatch):
    monkeypatch.setitem(sys.modules, "akshare", None)
    from tradingagents.dataflows.akshare import get_stock

    with pytest.raises(VendorNotConfiguredError, match="akshare"):
        get_stock("600519", "2026-01-01", "2026-01-10")


@pytest.mark.unit
def test_fundamentals_format_fake_akshare(fake_akshare):
    from tradingagents.dataflows.akshare import get_fundamentals

    result = get_fundamentals("600519", "2026-01-10")

    assert "# AkShare fundamentals for 600519.SS" in result
    assert "\u8d35\u5dde\u8305\u53f0" in result
    assert "\u9152\u7c7b\u884c\u4e1a" in result
    assert "\u51c0\u5229\u6da6" in result


@pytest.mark.unit
def test_fundamentals_falls_back_to_cninfo_profile(monkeypatch):
    fake = FakeAkShareWithBrokenProfile()
    monkeypatch.setitem(sys.modules, "akshare", fake)
    from tradingagents.dataflows.akshare import get_fundamentals

    result = get_fundamentals("600519", "2026-01-10")

    assert fake.profile_call == {"symbol": "600519"}
    assert "# AkShare fundamentals for 600519.SS" in result
    assert "\u8d35\u5dde\u8305\u53f0\u9152\u80a1\u4efd\u6709\u9650\u516c\u53f8" in result
    assert "\u51c0\u5229\u6da6" in result


@pytest.mark.unit
def test_news_format_fake_akshare(fake_akshare):
    from tradingagents.dataflows.akshare import get_news

    result = get_news("600519", "2026-01-01", "2026-01-10")

    assert "# AkShare news for 600519.SS" in result
    assert "\u8d35\u5dde\u8305\u53f0\u53d1\u5e03\u516c\u544a" in result
    assert "\u4e1c\u65b9\u8d22\u5bcc" in result
    assert "https://example.com/news" in result
