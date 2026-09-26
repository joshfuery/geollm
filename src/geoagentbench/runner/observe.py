from __future__ import annotations

import json
import time
from typing import Any, Callable

import numpy as np

from ..scorers.route_scorer import parse_route_answer
from .config import AgentConfig
from .providers import InfrastructureError, LLMProvider, LLMResponse

Emit = Callable[[dict[str, Any]], None]

_MAX_TEXT = 4000


def _cells(value: Any) -> list[list[int]] | None:
    if not isinstance(value, list):
        return None
    out = []
    for c in value:
        if isinstance(c, (list, tuple)) and len(c) == 2:
            try:
                out.append([int(c[0]), int(c[1])])
            except (TypeError, ValueError):
                continue
        elif isinstance(c, dict) and "row" in c and "col" in c:
            out.append([int(c["row"]), int(c["col"])])
    return out or None


def draft_from_text(text: str) -> list[list[int]] | None:
    from .agents import _extract_final_path

    if not text:
        return None
    parsed = parse_route_answer(_extract_final_path(text)).parsed
    return [list(c) for c in parsed] if parsed and len(parsed) >= 2 else None


class ObservedProvider:
    def __init__(self, inner: LLMProvider, emit: Emit):
        self.inner = inner
        self.emit = emit
        self.step = 0
        self._seen_tool_ids: set[str] = set()


    def _report_tool_results(self, messages: list) -> None:
        calls: dict[str, tuple[str, dict]] = {}
        for m in messages:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls[tc.get("id")] = (fn.get("name", "?"), args)
        for m in messages:
            if m.get("role") != "tool" or m.get("tool_call_id") in self._seen_tool_ids:
                continue
            self._seen_tool_ids.add(m.get("tool_call_id"))
            name, args = calls.get(m.get("tool_call_id"), ("?", {}))
            try:
                result = json.loads(m.get("content") or "{}")
            except json.JSONDecodeError:
                result = {"raw": m.get("content")}
            ev: dict[str, Any] = {"type": "tool", "name": name, "args": args}
            if name == "solve_between":
                ev.update(ok=bool(result.get("ok")), cells=_cells(result.get("cells")),
                          cost=result.get("cost"), error=result.get("error"))
            elif name == "check_path":
                issues = result.get("issues") or []
                ev.update(ok=bool(result.get("valid")), issues=issues[:6], n_issues=len(issues),
                          draft=_cells(args.get("cells") or args.get("path")))
            else:
                ev.update(ok=False, error=result.get("error"))
            self.emit(ev)

    def _report_response(self, resp: LLMResponse) -> None:
        tool_calls = [{"name": tc["name"], "args": tc["arguments"]} for tc in (resp.tool_calls or [])]
        text = resp.text or ""
        self.emit({
            "type": "llm",
            "step": self.step,
            "text": text[-_MAX_TEXT:],
            "truncated": len(text) > _MAX_TEXT,
            "tool_calls": tool_calls,
            "draft": None if tool_calls else draft_from_text(text),
            "tokens_in": resp.tokens_in,
            "tokens_out": resp.tokens_out,
            "wall_ms": resp.wall_ms,
        })

    def _call(self, fn: Callable[[], LLMResponse]) -> LLMResponse:
        self.step += 1
        self.emit({"type": "thinking", "step": self.step})
        try:
            resp = fn()
        except InfrastructureError as e:
            self.emit({"type": "infra_error", "step": self.step, "message": str(e)})
            raise
        self._report_response(resp)
        return resp


    def complete(self, system: str, user: str, cfg: AgentConfig) -> LLMResponse:
        return self._call(lambda: self.inner.complete(system, user, cfg))

    def complete_with_tools(self, messages: list, tools: list, cfg: AgentConfig) -> LLMResponse:
        self._report_tool_results(messages)
        return self._call(lambda: self.inner.complete_with_tools(messages, tools, cfg))


class ComposerResponder:
    def __init__(self, max_leg: int = 4, think_s: float = 0.0):
        self.max_leg = max(1, max_leg)
        self.think_s = think_s
        self.grid: np.ndarray | None = None
        self.legs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.path: list[list[int]] = []
        self.pending: tuple[tuple[int, int], tuple[int, int]] | None = None
        self.checked = False

    def _open_near(self, cell: tuple[int, int]) -> tuple[int, int]:
        assert self.grid is not None
        n = self.grid.shape[0]
        r0, c0 = cell
        for rad in range(n):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < n and 0 <= c < n and np.isfinite(self.grid[r, c]):
                        return (r, c)
        return cell

    def _plan(self, start: tuple[int, int], goal: tuple[int, int]) -> None:
        dist = max(abs(goal[0] - start[0]), abs(goal[1] - start[1]))
        k = max(1, int(np.ceil(dist / self.max_leg)))
        pts = [start]
        for i in range(1, k):
            t = i / k
            wp = (round(start[0] + (goal[0] - start[0]) * t), round(start[1] + (goal[1] - start[1]) * t))
            pts.append(self._open_near(wp))
        pts.append(goal)
        self.legs = list(zip(pts, pts[1:]))

    def _solve_call(self, a, b) -> dict:
        self.pending = (a, b)
        return {"tool_calls": [{"name": "solve_between", "arguments": {
            "from_row": a[0], "from_col": a[1], "to_row": b[0], "to_col": b[1]}}]}

    def _final(self) -> str:
        return "Joined all legs.\n\nFINAL PATH: " + " -> ".join(f"({r}, {c})" for r, c in self.path)

    def __call__(self, system: str, user: str, cfg: AgentConfig):
        if self.think_s:
            time.sleep(self.think_s)
        from .mock_agents import _extract_task

        if self.grid is None:
            got = _extract_task(user)
            if got is None:
                return "I need the structured cost grid to plan. FINAL PATH:"
            start, goal, self.grid = got
            self._plan(start, goal)
            return self._solve_call(*self.legs.pop(0))

        try:
            result = json.loads(user)
        except (json.JSONDecodeError, TypeError):
            result = {}

        if self.pending is not None:
            a, b = self.pending
            self.pending = None
            if result.get("ok") and result.get("cells"):
                cells = result["cells"]
                self.path.extend(cells if not self.path else cells[1:])
            else:
                mid = self._open_near(((a[0] + b[0]) // 2, (a[1] + b[1]) // 2))
                if mid not in (a, b):
                    self.legs[:0] = [(a, mid), (mid, b)]
                else:
                    self.path.append(list(b))
            if self.legs:
                return self._solve_call(*self.legs.pop(0))
            if not self.checked:
                self.checked = True
                return {"tool_calls": [{"name": "check_path", "arguments": {"cells": self.path}}]}
        return self._final()
