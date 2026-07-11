import sys

import pandas as pd
import pytest

from tradingagents.agents.utils.macro_data_tools import get_china_macro_indicators
from tradingagents.dataflows import interface


class FakeChinaMacro:
    def macro_china_cpi(self):
        return pd.DataFrame(
            [
                {"月份": "2025年12月份", "全国-同比增长": 0.8},
                {"月份": "2026年01月份", "全国-同比增长": 1.0},
                {"月份": "2026年02月份", "全国-同比增长": 1.2},
            ]
        )


@pytest.mark.unit
def test_china_macro_registered_and_filters_future_rows(monkeypatch):
    monkeypatch.setitem(sys.modules, "akshare", FakeChinaMacro())

    result = get_china_macro_indicators.func("cpi", "2026-02-20", 365)

    assert "China macro: CPI" in result
    assert "2026年01月份" in result
    assert "2026年02月份" not in result
    assert "akshare_china_macro" in interface.VENDOR_METHODS["get_china_macro_indicators"]


@pytest.mark.unit
def test_china_macro_unknown_alias_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "akshare", FakeChinaMacro())

    result = get_china_macro_indicators.func("made_up_indicator", "2026-01-31", 365)

    assert result.startswith("DATA_UNAVAILABLE")
    assert "Supported China macro indicators" in result


@pytest.mark.unit
def test_china_macro_continues_to_documented_fallback(monkeypatch):
    class FallbackMacro:
        def macro_china_cpi(self):
            raise RuntimeError("primary blocked")

        def macro_china_cpi_yearly(self):
            return pd.DataFrame([{"月份": "2026年01月份", "同比增长": 1.0}])

    monkeypatch.setitem(sys.modules, "akshare", FallbackMacro())

    result = get_china_macro_indicators.func("cpi", "2026-02-20", 365)

    assert "macro_china_cpi_yearly" in result
    assert "2026年01月份" in result


@pytest.mark.unit
def test_china_macro_period_is_not_visible_before_conservative_release_date(monkeypatch):
    monkeypatch.setitem(sys.modules, "akshare", FakeChinaMacro())

    result = get_china_macro_indicators.func("cpi", "2026-01-31", 365)

    assert "2025年12月份" in result
    assert "2026年01月份" not in result
