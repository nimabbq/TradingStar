import pytest

from tradingagents.dataflows.a_share_execution import build_a_share_execution_plan


@pytest.mark.unit
def test_execution_plan_does_not_invent_quantity_without_account_inputs(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.a_share_execution._latest_raw_price",
        lambda ticker, date: (10.0, "2026-01-05"),
    )

    result = build_a_share_execution_plan("600519", "2026-01-05", "Buy", {})

    assert "Order quantity: unavailable" in result
    assert "configure TRADINGAGENTS_A_SHARE_ACCOUNT_CASH" in result


@pytest.mark.unit
def test_execution_plan_rounds_buy_to_board_lot_and_itemizes_fees(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.a_share_execution._latest_raw_price",
        lambda ticker, date: (10.0, "2026-01-05"),
    )
    config = {
        "a_share_account_cash": 100_000.0,
        "a_share_available_shares": 0,
        "a_share_commission_rate": 0.0003,
        "a_share_min_commission": 5.0,
        "a_share_transfer_fee_rate": 0.00001,
        "a_share_stamp_duty_rate": 0.0005,
    }

    result = build_a_share_execution_plan("600519", "2026-01-05", "Buy", config)

    assert "Target weight: 50%" in result
    assert "Order side: BUY" in result
    assert "Order quantity: 4900 shares" in result
    assert "Board lot: 100 shares" in result
    assert "Stamp duty: 0.00 CNY" in result


@pytest.mark.unit
def test_execution_plan_sell_uses_only_configured_sellable_shares(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.a_share_execution._latest_raw_price",
        lambda ticker, date: (10.0, "2026-01-05"),
    )
    config = {
        "a_share_account_cash": 0.0,
        "a_share_available_shares": 350,
        "a_share_commission_rate": 0.0003,
        "a_share_min_commission": 5.0,
        "a_share_transfer_fee_rate": 0.00001,
        "a_share_stamp_duty_rate": 0.0005,
    }

    result = build_a_share_execution_plan("600519", "2026-01-05", "Sell", config)

    assert "Order side: SELL" in result
    assert "Order quantity: 350 shares" in result
    assert "T+1" in result
