import subprocess

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.validators import validate_model


@pytest.mark.unit
def test_factory_routes_to_codex_cli_client():
    client = create_llm_client(provider="codex", model="codex")
    assert type(client).__name__ == "CodexCLIClient"


@pytest.mark.unit
def test_codex_provider_requires_no_api_key():
    assert get_api_key_env("codex") is None
    assert validate_model("codex", "any-codex-model") is True


@pytest.mark.unit
def test_plain_invoke_runs_codex_exec_and_reads_last_message(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_client import CodexCLIChatModel

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Codex answer")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tradingagents.llm_clients.codex_client.subprocess.run", fake_run)

    llm = CodexCLIChatModel(
        model="codex",
        codex_path=str(tmp_path / "codex.exe"),
        cwd="D:\\TradingAgents",
        sandbox="read-only",
    )

    result = llm.invoke("Summarize AAPL")

    assert result.content == "Codex answer"
    assert captured["cmd"][:2] == [str(tmp_path / "codex.exe"), "exec"]
    assert "--ephemeral" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("-C") + 1] == "D:\\TradingAgents"
    assert captured["cmd"][captured["cmd"].index("-s") + 1] == "read-only"
    assert "-m" not in captured["cmd"]
    assert captured["cmd"][-1] == "-"
    assert "Summarize AAPL" in captured["kwargs"]["input"]
    assert captured["kwargs"]["timeout"] is not None


@pytest.mark.unit
def test_non_default_model_is_forwarded_to_codex_exec(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_client import CodexCLIChatModel

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("ok")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tradingagents.llm_clients.codex_client.subprocess.run", fake_run)

    CodexCLIChatModel(
        model="gpt-5-codex",
        codex_path=str(tmp_path / "codex.exe"),
    ).invoke("hello")

    assert captured["cmd"][captured["cmd"].index("-m") + 1] == "gpt-5-codex"


@pytest.mark.unit
def test_empty_last_message_uses_stdout_fallback(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_client import CodexCLIChatModel

    def fake_run(cmd, **kwargs):
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")
        return subprocess.CompletedProcess(cmd, 0, stdout="stdout answer", stderr="")

    monkeypatch.setattr("tradingagents.llm_clients.codex_client.subprocess.run", fake_run)

    llm = CodexCLIChatModel(model="codex", codex_path=str(tmp_path / "codex.exe"))

    assert llm.invoke("hello").content == "stdout answer"


@pytest.mark.unit
def test_empty_last_message_returns_diagnostic_instead_of_crashing(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_client import CodexCLIChatModel

    def fake_run(cmd, **kwargs):
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tradingagents.llm_clients.codex_client.subprocess.run", fake_run)

    llm = CodexCLIChatModel(model="codex", codex_path=str(tmp_path / "codex.exe"))

    result = llm.invoke("hello")

    assert "Codex CLI completed successfully but did not write a final message" in result.content


@pytest.mark.unit
def test_bound_tools_parse_codex_tool_call_json(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_client import CodexCLIChatModel

    @tool
    def get_price(symbol: str) -> str:
        """Fetch the current price for a symbol."""
        return symbol

    def fake_run(cmd, **kwargs):
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                '{"type":"tool_calls","tool_calls":[{"name":"get_price","args":{"symbol":"AAPL"}}]}'
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tradingagents.llm_clients.codex_client.subprocess.run", fake_run)

    llm = CodexCLIChatModel(model="codex", codex_path=str(tmp_path / "codex.exe"))
    result = llm.bind_tools([get_price]).invoke([HumanMessage(content="price please")])

    assert result.content == ""
    assert result.tool_calls == [
        {
            "name": "get_price",
            "args": {"symbol": "AAPL"},
            "id": result.tool_calls[0]["id"],
            "type": "tool_call",
        }
    ]
    assert result.tool_calls[0]["id"].startswith("call_")


@pytest.mark.unit
def test_bound_tools_parse_codex_final_json(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_client import CodexCLIChatModel

    @tool
    def get_price(symbol: str) -> str:
        """Fetch the current price for a symbol."""
        return symbol

    def fake_run(cmd, **kwargs):
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('```json\n{"type":"final","content":"Final report"}\n```')
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tradingagents.llm_clients.codex_client.subprocess.run", fake_run)

    llm = CodexCLIChatModel(model="codex", codex_path=str(tmp_path / "codex.exe"))
    result = llm.bind_tools([get_price]).invoke([HumanMessage(content="done?")])

    assert result.content == "Final report"
    assert result.tool_calls == []


@pytest.mark.unit
def test_codex_structured_output_uses_existing_fallback_path(tmp_path):
    from tradingagents.llm_clients.codex_client import CodexCLIChatModel

    llm = CodexCLIChatModel(model="codex", codex_path=str(tmp_path / "codex.exe"))

    with pytest.raises(NotImplementedError, match="structured output"):
        llm.with_structured_output(dict)
