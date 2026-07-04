# China Mutual Fund Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mainland China off-exchange mutual fund support so users can analyze codes such as `005827`, `005827.FUND`, and `FUND:005827` through AkShare.

**Architecture:** Add a small fund symbol/rule module, a dedicated AkShare fund vendor, and route fund symbols to that vendor before global stock vendors. Keep fund semantics separate from A-share stock rules so mutual funds do not receive limit-up, board-lot, or dragon-tiger guidance.

**Tech Stack:** Python 3.12, AkShare, pandas, stockstats, LangChain tools, pytest.

---

### Task 1: Fund Symbol And Context

**Files:**
- Create: `tradingagents/dataflows/china_fund_rules.py`
- Modify: `tradingagents/dataflows/symbol_utils.py`
- Modify: `cli/models.py`
- Modify: `cli/utils.py`
- Test: `tests/test_china_fund_rules.py`
- Test: `tests/test_cli_symbol_handling.py`

- [ ] **Step 1: Write failing tests**

Test that `005827`, `005827.FUND`, and `FUND:005827` normalize to `005827.FUND`, and that `510300.SS` and `600519` are not classified as off-exchange funds.

- [ ] **Step 2: Run tests and verify red**

Run: `python -m pytest tests/test_china_fund_rules.py tests/test_cli_symbol_handling.py -q --basetemp C:\tmp\pytest-fund-rules-red`

Expected: import/attribute failures for missing fund rule module or missing `AssetType.FUND`.

- [ ] **Step 3: Implement symbol rules**

Create `china_fund_rules.py` with `is_china_fund_symbol`, `normalize_china_fund_symbol`, `extract_china_fund_code`, and `build_china_fund_rule_context`. Update `normalize_symbol()` before the crypto/forex fallback. Add `AssetType.FUND` and let `detect_asset_type()` return it for fund symbols.

- [ ] **Step 4: Run tests and verify green**

Run: `python -m pytest tests/test_china_fund_rules.py tests/test_cli_symbol_handling.py -q --basetemp C:\tmp\pytest-fund-rules`

Expected: all selected tests pass.

### Task 2: AkShare Fund Vendor

**Files:**
- Create: `tradingagents/dataflows/akshare_fund.py`
- Modify: `tradingagents/dataflows/interface.py`
- Modify: `tradingagents/default_config.py`
- Test: `tests/test_akshare_fund_vendor.py`

- [ ] **Step 1: Write failing tests**

Use fake AkShare endpoints for `fund_open_fund_info_em`, `fund_individual_basic_info_xq`, `fund_individual_detail_info_xq`, and portfolio endpoints. Assert routing `get_stock_data("005827", ...)`, `get_indicators("005827", "rsi", ...)`, and `get_fundamentals("005827", ...)` prefers the fund vendor.

- [ ] **Step 2: Run tests and verify red**

Run: `python -m pytest tests/test_akshare_fund_vendor.py -q --basetemp C:\tmp\pytest-akshare-fund-red`

Expected: missing module or missing vendor routing failures.

- [ ] **Step 3: Implement fund vendor and routing**

Implement AkShare fund NAV history, NAV-based indicators, fundamentals/fees/holdings formatting, and non-applicable statements returning `DATA_UNAVAILABLE`. Add `akshare_fund` to `VENDOR_METHODS` and `china_fund_data_vendors` defaults.

- [ ] **Step 4: Run tests and verify green**

Run: `python -m pytest tests/test_akshare_fund_vendor.py tests/test_vendor_routing.py tests/test_akshare_vendor.py -q --basetemp C:\tmp\pytest-akshare-fund`

Expected: all selected tests pass.

### Task 3: Agent Context And Real Network Smoke

**Files:**
- Modify: `tradingagents/agents/utils/agent_utils.py`
- Test: `tests/test_china_fund_rules.py`

- [ ] **Step 1: Add context tests**

Assert `build_instrument_context("005827.FUND", "fund")` tells agents this is a China mutual fund, not a company, and not an A-share stock.

- [ ] **Step 2: Implement context**

Append fund-specific guidance: NAV is end-of-day, no intraday stock price assumptions, no limit-up/limit-down, focus on fund manager, fees, holdings, drawdown, and redemption/subscription restrictions.

- [ ] **Step 3: Run tests and smoke checks**

Run selected pytest commands, then run one real AkShare smoke for `005827` NAV and fundamentals in the current network.

### Task 4: Final Verification And Push

**Files:**
- Commit all changed files.

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q --basetemp C:\tmp\pytest-tradingagents-fund-final`

Expected: all tests pass, with existing skips/warnings only.

- [ ] **Step 2: Commit and push**

Commit: `Add China mutual fund AkShare support`

Push: `git push`
