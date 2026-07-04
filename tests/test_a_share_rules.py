import pytest

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.dataflows.a_share_rules import (
    AShareBoard,
    build_a_share_rule_context,
    classify_a_share_board,
    is_a_share_symbol,
    normalize_a_share_symbol,
)
from tradingagents.dataflows.symbol_utils import normalize_symbol


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("600519", "600519.SS"),
        ("601318", "601318.SS"),
        ("688981", "688981.SS"),
        ("000001", "000001.SZ"),
        ("002594", "002594.SZ"),
        ("300750", "300750.SZ"),
        ("430047", "430047.BJ"),
        ("839680", "839680.BJ"),
        ("600519.ss", "600519.SS"),
        ("000001.sz", "000001.SZ"),
    ],
)
def test_normalize_a_share_symbol(raw, expected):
    assert normalize_a_share_symbol(raw) == expected
    assert normalize_symbol(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "symbol,board",
    [
        ("600519", AShareBoard.SH_MAIN),
        ("605499.SS", AShareBoard.SH_MAIN),
        ("688981", AShareBoard.STAR),
        ("000001", AShareBoard.SZ_MAIN),
        ("002594.SZ", AShareBoard.SZ_MAIN),
        ("300750", AShareBoard.CHINEXT),
        ("430047", AShareBoard.BSE),
        ("839680.BJ", AShareBoard.BSE),
    ],
)
def test_classify_a_share_board(symbol, board):
    assert classify_a_share_board(symbol) is board
    assert is_a_share_symbol(symbol) is True


@pytest.mark.unit
@pytest.mark.parametrize("symbol", ["AAPL", "0700.HK", "BTC-USD", "GC=F", "not-a-symbol"])
def test_non_a_share_symbols_are_not_classified(symbol):
    assert is_a_share_symbol(symbol) is False
    with pytest.raises(ValueError, match="not a supported A-share symbol"):
        classify_a_share_board(symbol)


@pytest.mark.unit
def test_rule_context_contains_main_board_constraints():
    context = build_a_share_rule_context("600519", identity={"company_name": "Kweichow Moutai"})

    assert "Shanghai Main Board" in context
    assert "600519.SS" in context
    assert "Kweichow Moutai" in context
    assert "T+1" in context
    assert "100 shares" in context
    assert "10%" in context
    assert "limit-up" in context
    assert "midday break" in context


@pytest.mark.unit
def test_rule_context_contains_board_specific_limits():
    assert "20%" in build_a_share_rule_context("688981")
    assert "STAR Market" in build_a_share_rule_context("688981")
    assert "20%" in build_a_share_rule_context("300750")
    assert "ChiNext" in build_a_share_rule_context("300750")
    assert "30%" in build_a_share_rule_context("430047")
    assert "Beijing Stock Exchange" in build_a_share_rule_context("430047")


@pytest.mark.unit
def test_instrument_context_injects_a_share_rules_only_for_a_shares():
    a_share_context = build_instrument_context("600519", identity={"company_name": "Kweichow Moutai"})
    us_context = build_instrument_context("AAPL", identity={"company_name": "Apple Inc."})

    assert "A-share trading constraints" in a_share_context
    assert "T+1" in a_share_context
    assert "daily price limit" in a_share_context
    assert "A-share trading constraints" not in us_context
    assert "T+1" not in us_context
