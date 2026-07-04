# AkShare A-Share Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AkShare-backed A-share data, A-share market rules, and China-market specialty tools to TradingAgents.

**Architecture:** Add deterministic A-share rules in `dataflows/a_share_rules.py`, AkShare vendor functions in `dataflows/akshare.py`, and explicit A-share specialty LangChain tools in `agents/utils/a_share_tools.py`. Register AkShare in the existing vendor router and inject A-share rule context into existing agent prompts through `build_instrument_context`.

**Tech Stack:** Python 3.12, pytest, pandas, stockstats, optional AkShare dependency, LangChain tools, existing TradingAgents vendor router.

---

### Task 1: A-Share Symbol And Rule Context

**Files:**
- Create: `tradingagents/dataflows/a_share_rules.py`
- Modify: `tradingagents/dataflows/symbol_utils.py`
- Modify: `tradingagents/agents/utils/agent_utils.py`
- Test: `tests/test_a_share_rules.py`

- [ ] **Step 1: Write failing tests**

Cover:
- `600519 -> 600519.SS`
- `300750 -> 300750.SZ`
- `688981 -> 688981.SS`
- `430047 -> 430047.BJ`
- board-specific context mentions T+1, lot size, price limit, and board name
- non-A-share symbols are left alone and get no A-share context

Run:

```powershell
C:\Users\35338\miniconda3\envs\tradingagents\python.exe -m pytest tests\test_a_share_rules.py -q --basetemp C:\tmp\pytest-ashare-rules
```

Expected before implementation: import failure for `tradingagents.dataflows.a_share_rules`.

- [ ] **Step 2: Implement minimal rules module**

Add `AShareBoard`, `normalize_a_share_symbol`, `is_a_share_symbol`, `classify_a_share_board`, and `build_a_share_rule_context`.

- [ ] **Step 3: Wire context injection**

Update `build_instrument_context` to append A-share rule context only for A-share symbols.

- [ ] **Step 4: Verify**

Run the same targeted test and confirm it passes.

### Task 2: AkShare Vendor Registration And Core Data

**Files:**
- Create: `tradingagents/dataflows/akshare.py`
- Modify: `tradingagents/dataflows/interface.py`
- Modify: `tradingagents/default_config.py`
- Modify: `pyproject.toml`
- Test: `tests/test_akshare_vendor.py`

- [ ] **Step 1: Write failing tests**

Cover:
- `akshare` appears in `VENDOR_LIST`
- `route_to_vendor("get_stock_data", "600519", ...)` uses fake AkShare when A-share defaults apply
- missing AkShare raises `VendorNotConfiguredError`
- fake `stock_zh_a_hist` data frame formats as CSV with AkShare source header
- fake `stock_news_em` formats news text

Run:

```powershell
C:\Users\35338\miniconda3\envs\tradingagents\python.exe -m pytest tests\test_akshare_vendor.py -q --basetemp C:\tmp\pytest-akshare-vendor
```

Expected before implementation: import/registration failures.

- [ ] **Step 2: Implement AkShare lazy import and helpers**

Add lazy `_ak()` import, symbol conversion helpers, date conversion, empty-frame guard, and safe frame formatting.

- [ ] **Step 3: Implement vendor functions**

Add `get_stock`, `get_indicator`, `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`, `get_news`, and a no-data `get_insider_transactions` placeholder for A-shares.

- [ ] **Step 4: Register vendor**

Add AkShare imports and methods in `interface.py`. Add `a_share_data_vendors` config and optional `a-share` dependency.

- [ ] **Step 5: Verify**

Run AkShare vendor tests and adjacent vendor routing tests.

### Task 3: A-Share Specialty Tools

**Files:**
- Create: `tradingagents/agents/utils/a_share_tools.py`
- Modify: `tradingagents/agents/utils/agent_utils.py`
- Modify: `tradingagents/graph/trading_graph.py`
- Test: `tests/test_a_share_tools.py`

- [ ] **Step 1: Write failing tests**

Cover:
- fake endpoint responses format for dragon-tiger, northbound flow, margin financing, limit pool, sector fund flow
- unavailable endpoints return `DATA_UNAVAILABLE`
- graph tool nodes include A-share tools for A-share runs and omit them for non-A-share runs

Run:

```powershell
C:\Users\35338\miniconda3\envs\tradingagents\python.exe -m pytest tests\test_a_share_tools.py -q --basetemp C:\tmp\pytest-ashare-tools
```

Expected before implementation: missing module/tool failures.

- [ ] **Step 2: Implement specialty tools**

Wrap AkShare specialty endpoints behind LangChain tools, format data frames consistently, and catch exceptions as `DATA_UNAVAILABLE`.

- [ ] **Step 3: Wire graph tool nodes**

Make `TradingAgentsGraph._create_tool_nodes` aware of selected asset/symbol context so A-share tools are only offered for A-share runs.

- [ ] **Step 4: Update analyst prompts**

Mention A-share specialty tools in market, news, and fundamentals analyst prompts when they are included.

- [ ] **Step 5: Verify**

Run specialty tool tests plus `tests/test_market_toolnode.py`.

### Task 4: Full Verification

**Files:**
- All touched files

- [ ] **Step 1: Run targeted tests**

```powershell
C:\Users\35338\miniconda3\envs\tradingagents\python.exe -m pytest tests\test_a_share_rules.py tests\test_akshare_vendor.py tests\test_a_share_tools.py tests\test_vendor_routing.py tests\test_cli_symbol_handling.py tests\test_market_toolnode.py -q --basetemp C:\tmp\pytest-ashare-final
```

- [ ] **Step 2: Compile touched modules**

```powershell
C:\Users\35338\miniconda3\envs\tradingagents\python.exe -m compileall tradingagents tests
```

- [ ] **Step 3: Check Git diff**

```powershell
git status --short
git diff --stat
```

- [ ] **Step 4: Report remote push blocker**

Remote push currently fails because local GitHub HTTPS credentials authenticate as `nimabbq`, which lacks write permission to `4sahiSuperDry/TradingStar`.
