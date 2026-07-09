from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from .base_client import BaseLLMClient


_DEFAULT_CODEX_MODEL_NAMES = {"", "codex", "default", "codex-default"}


def _resolve_codex_path(explicit: str | None = None) -> str:
    """Resolve the Codex CLI executable without relying on WindowsApps aliases."""
    if explicit:
        return explicit

    for env_var in ("TRADINGAGENTS_CODEX_PATH", "CODEX_PATH", "CODEX_CLI_PATH"):
        value = os.environ.get(env_var)
        if value:
            return value

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates = list(Path(local_app_data).glob("OpenAI/Codex/bin/*/codex.exe"))
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return str(candidates[0])

    return "codex"


def _default_codex_cwd() -> str:
    return os.environ.get("TRADINGAGENTS_CODEX_CWD") or os.getcwd()


def _default_output_dir() -> str | None:
    configured = os.environ.get("TRADINGAGENTS_CODEX_OUTPUT_DIR")
    if configured:
        return configured
    windows_tmp = Path("C:/tmp")
    if os.name == "nt" and windows_tmp.exists():
        return str(windows_tmp)
    return None


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=True, default=str))
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_to_text(message: BaseMessage) -> str:
    role = getattr(message, "type", message.__class__.__name__)
    lines = [f"{role.upper()}:"]
    content = _content_to_text(message.content)
    if content:
        lines.append(content)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        lines.append("Tool calls requested:")
        lines.append(json.dumps(tool_calls, ensure_ascii=True, default=str, indent=2))

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        lines.append(f"Tool call id: {tool_call_id}")

    name = getattr(message, "name", None)
    if name:
        lines.append(f"Name: {name}")

    return "\n".join(lines)


def _tool_spec(tool: Any) -> dict[str, Any]:
    name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
    return {
        "name": name,
        "description": getattr(tool, "description", "") or "",
        "args_schema": getattr(tool, "args", {}) or {},
    }


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json(text: str) -> dict[str, Any]:
    candidate = _strip_json_fence(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(candidate[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Codex tool response must be a JSON object")
    return parsed


def _message_from_tool_protocol(text: str) -> AIMessage:
    try:
        data = _extract_json(text)
    except Exception:
        return AIMessage(content=text)

    response_type = str(data.get("type", "final")).lower()
    if response_type == "final":
        return AIMessage(content=str(data.get("content", "")))

    raw_calls = data.get("tool_calls") or []
    if response_type != "tool_calls" or not isinstance(raw_calls, list):
        return AIMessage(content=str(data.get("content", text)))

    tool_calls = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        name = raw_call.get("name")
        args = raw_call.get("args", {})
        if not name or not isinstance(args, dict):
            continue
        tool_calls.append({
            "name": str(name),
            "args": args,
            "id": str(raw_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
            "type": "tool_call",
        })

    return AIMessage(content="", tool_calls=tool_calls)


def _fallback_empty_codex_output(completed: subprocess.CompletedProcess) -> str:
    stdout = (completed.stdout or "").strip()
    if stdout:
        return stdout

    stderr = (completed.stderr or "").strip()
    if stderr:
        return (
            "Codex CLI completed successfully but did not write a final message.\n\n"
            "CLI diagnostics:\n"
            f"{stderr}"
        )

    return (
        "Codex CLI completed successfully but did not write a final message. "
        "Proceed with the available conversation and tool outputs, and report "
        "that the model response was unavailable instead of fabricating analysis."
    )


class CodexCLIChatModel(BaseChatModel):
    """LangChain chat wrapper that delegates each call to ``codex exec``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "codex"
    codex_path: str | None = None
    cwd: str = Field(default_factory=_default_codex_cwd)
    sandbox: str = "read-only"
    timeout: float = 1800.0
    extra_args: tuple[str, ...] = Field(default_factory=tuple)
    tools: tuple[Any, ...] = Field(default_factory=tuple, exclude=True)
    tool_choice: str | None = None

    @property
    def _llm_type(self) -> str:
        return "codex-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "codex_path": _resolve_codex_path(self.codex_path),
            "cwd": self.cwd,
            "sandbox": self.sandbox,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return self.model_copy(update={
            "tools": tuple(tools),
            "tool_choice": tool_choice,
        })

    def with_structured_output(self, schema: Any, *, method: str | None = None, **kwargs: Any):
        raise NotImplementedError(
            "Codex CLI provider does not support structured output yet; "
            "TradingAgents will fall back to free-text generation."
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._build_prompt(messages, stop=stop)
        output = self._run_codex(prompt)
        message = _message_from_tool_protocol(output) if self.tools else AIMessage(content=output)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _build_prompt(self, messages: list[BaseMessage], stop: list[str] | None = None) -> str:
        sections = [
            "You are Codex CLI running as the LLM provider inside TradingAgents.",
            "Use the conversation below to produce the next assistant message.",
        ]

        if self.tools:
            sections.append(
                "You are in a LangGraph tool-calling loop. Respond with exactly one JSON object "
                "and no surrounding prose or Markdown fences. To request tools, return "
                '{"type":"tool_calls","tool_calls":[{"name":"tool_name","args":{}}]}. '
                'To answer finally, return {"type":"final","content":"your markdown report"}. '
                "Only call tools from the provided list. If tool results are already present "
                "and sufficient, produce a final answer."
            )
            sections.append("Available tools:")
            sections.append(json.dumps(
                [_tool_spec(tool) for tool in self.tools],
                ensure_ascii=True,
                default=str,
                indent=2,
            ))
            if self.tool_choice:
                sections.append(f"Requested tool choice: {self.tool_choice}")

        if stop:
            sections.append("Stop sequences requested by caller: " + json.dumps(stop))

        sections.append("Conversation:")
        sections.extend(_message_to_text(message) for message in messages)
        sections.append("Respond now.")
        return "\n\n".join(sections)

    def _run_codex(self, prompt: str) -> str:
        output_path = self._make_output_path()
        cmd = [
            _resolve_codex_path(self.codex_path),
            "exec",
            "--ephemeral",
            "-C",
            self.cwd,
            "-s",
            self.sandbox,
            "--color",
            "never",
            "-o",
            output_path,
        ]
        if self.model.lower() not in _DEFAULT_CODEX_MODEL_NAMES:
            cmd.extend(["-m", self.model])
        cmd.extend(self.extra_args)
        cmd.append("-")

        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
            output = Path(output_path).read_text(encoding="utf-8")
        finally:
            try:
                Path(output_path).unlink()
            except OSError:
                pass

        if completed.returncode != 0:
            raise RuntimeError(
                "Codex CLI failed with exit code "
                f"{completed.returncode}: {completed.stderr or completed.stdout}"
            )
        if not output.strip():
            return _fallback_empty_codex_output(completed)
        return output.strip()

    def _make_output_path(self) -> str:
        fd, path = tempfile.mkstemp(
            prefix="tradingagents_codex_",
            suffix=".txt",
            dir=_default_output_dir(),
        )
        os.close(fd)
        return path


class CodexCLIClient(BaseLLMClient):
    """Client for running TradingAgents LLM calls through ``codex exec``."""

    def get_llm(self) -> Any:
        extra_args = self.kwargs.get("extra_args")
        if extra_args is None:
            extra_args = os.environ.get("TRADINGAGENTS_CODEX_EXTRA_ARGS", "")
        if isinstance(extra_args, str):
            extra_args = tuple(shlex.split(extra_args))
        else:
            extra_args = tuple(extra_args or ())

        timeout = self.kwargs.get("timeout")
        if timeout is None:
            timeout = float(os.environ.get("TRADINGAGENTS_CODEX_TIMEOUT", "1800"))

        return CodexCLIChatModel(
            model=self.model,
            codex_path=self.kwargs.get("codex_path"),
            cwd=self.kwargs.get("cwd") or _default_codex_cwd(),
            sandbox=self.kwargs.get("sandbox")
            or os.environ.get("TRADINGAGENTS_CODEX_SANDBOX", "read-only"),
            timeout=float(timeout),
            extra_args=extra_args,
            callbacks=self.kwargs.get("callbacks"),
        )

    def validate_model(self) -> bool:
        return True
