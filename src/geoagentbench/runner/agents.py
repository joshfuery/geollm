from __future__ import annotations

import re
import time


from ..core import NodeTrace, ParsedAnswer, TaskInstance
from ..scorers.route_scorer import parse_route_answer
from .config import AgentConfig
from .prompts import render_critique_prompt, render_prompt, render_revise_prompt
from .tools import (check_path_issues, execute_tool_call,
                    format_issues, max_segment_for_access, tool_specs_for_access)
from .providers import LLMProvider


_FINAL_PATH_RE = re.compile(r"FINAL\s*PATH\s*:\s*(.+?)(?:\n\n|\Z)", re.IGNORECASE | re.DOTALL)


def _extract_final_path(text: str) -> str:
    m = _FINAL_PATH_RE.search(text)
    return m.group(1).strip() if m else text


class SingleShotAgent:
    NAME = "single_shot"

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def run(
        self,
        task: TaskInstance,
        cfg: AgentConfig,
        cost_grid=None,
    ) -> tuple[ParsedAnswer, list[NodeTrace], int, int, int]:
        t0 = time.perf_counter()
        system, user = render_prompt(task, cfg.condition, cfg.prompt_version)
        resp = self.provider.complete(system, user, cfg)

        final_line = _extract_final_path(resp.text)
        parsed = parse_route_answer(final_line)
        parsed = parsed.model_copy(update={"raw": resp.text})

        wall_ms = int((time.perf_counter() - t0) * 1000)
        trace = NodeTrace(
            node_name="single_shot_llm",
            input={
                "system_len": len(system),
                "user_len": len(user),
                "condition": cfg.condition.value,
                "prompt_version": cfg.prompt_version,
            },
            output={
                "raw_len": len(resp.text),
                "final_path_len": len(final_line),
                "parsed_cells": len(parsed.parsed) if parsed.parsed else 0,
                "parse_error": parsed.parse_error,
                "model_reported": resp.model_reported,
            },
            valid=parsed.parsed is not None,
            wall_ms=resp.wall_ms,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
        )
        return parsed, [trace], resp.tokens_in, resp.tokens_out, wall_ms


class PlanCritiqueReviseAgent:
    NAME = "plan_critique_revise"

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def run(
        self,
        task: TaskInstance,
        cfg: AgentConfig,
        cost_grid=None,
    ) -> tuple[ParsedAnswer, list[NodeTrace], int, int, int]:
        import time as _time
        t0 = _time.perf_counter()
        traces: list[NodeTrace] = []
        tok_in = tok_out = 0

        sys_p, user_p = render_prompt(task, cfg.condition, cfg.prompt_version)
        r_plan = self.provider.complete(sys_p, user_p, cfg)
        plan_line = _extract_final_path(r_plan.text)
        plan_parsed = parse_route_answer(plan_line)
        traces.append(NodeTrace(
            node_name="plan",
            input={"system_len": len(sys_p), "user_len": len(user_p),
                   "condition": cfg.condition.value},
            output={"raw_len": len(r_plan.text), "final_path_len": len(plan_line),
                    "parsed_cells": len(plan_parsed.parsed) if plan_parsed.parsed else 0,
                    "parse_error": plan_parsed.parse_error},
            valid=plan_parsed.parsed is not None,
            wall_ms=r_plan.wall_ms, tokens_in=r_plan.tokens_in, tokens_out=r_plan.tokens_out,
        ))
        tok_in += r_plan.tokens_in; tok_out += r_plan.tokens_out

        sys_c, user_c = render_critique_prompt(task, cfg.condition, r_plan.text, cfg.prompt_version)
        r_crit = self.provider.complete(sys_c, user_c, cfg)
        crit_issues = _count_critique_issues(r_crit.text)
        traces.append(NodeTrace(
            node_name="critique",
            input={"system_len": len(sys_c), "user_len": len(user_c)},
            output={"raw_len": len(r_crit.text), "issues_found": crit_issues},
            valid=None,
            wall_ms=r_crit.wall_ms, tokens_in=r_crit.tokens_in, tokens_out=r_crit.tokens_out,
        ))
        tok_in += r_crit.tokens_in; tok_out += r_crit.tokens_out

        sys_r, user_r = render_revise_prompt(task, cfg.condition, r_plan.text, r_crit.text, cfg.prompt_version)
        r_rev = self.provider.complete(sys_r, user_r, cfg)
        rev_line = _extract_final_path(r_rev.text)
        rev_parsed = parse_route_answer(rev_line).model_copy(update={"raw": r_rev.text})
        traces.append(NodeTrace(
            node_name="revise",
            input={"system_len": len(sys_r), "user_len": len(user_r),
                   "critique_issues": crit_issues},
            output={"raw_len": len(r_rev.text), "final_path_len": len(rev_line),
                    "parsed_cells": len(rev_parsed.parsed) if rev_parsed.parsed else 0,
                    "parse_error": rev_parsed.parse_error,
                    "changed_from_plan": rev_line.strip() != plan_line.strip()},
            valid=rev_parsed.parsed is not None,
            wall_ms=r_rev.wall_ms, tokens_in=r_rev.tokens_in, tokens_out=r_rev.tokens_out,
        ))
        tok_in += r_rev.tokens_in; tok_out += r_rev.tokens_out

        wall_ms = int((_time.perf_counter() - t0) * 1000)
        return rev_parsed, traces, tok_in, tok_out, wall_ms


def _count_critique_issues(text: str) -> int:
    import re as _re
    m = _re.search(r"ISSUES\s*:\s*(.*)", text, flags=_re.IGNORECASE | _re.DOTALL)
    if not m:
        return 0
    body = m.group(1)
    return sum(1 for line in body.splitlines() if line.strip().startswith("-"))


class PlanVerifyReviseAgent:
    NAME = "plan_verify_revise"

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def run(
        self,
        task: TaskInstance,
        cfg: AgentConfig,
        cost_grid=None,
    ) -> tuple[ParsedAnswer, list[NodeTrace], int, int, int]:
        if cost_grid is None:
            raise ValueError("plan_verify_revise needs cost_grid — executor must pass it")

        import time as _time
        t0 = _time.perf_counter()
        traces: list[NodeTrace] = []
        tok_in = tok_out = 0
        max_rounds = max(1, cfg.max_revise_rounds)

        sys_p, user_p = render_prompt(task, cfg.condition, cfg.prompt_version)
        r_plan = self.provider.complete(sys_p, user_p, cfg)
        plan_line = _extract_final_path(r_plan.text)
        plan_parsed = parse_route_answer(plan_line)
        traces.append(NodeTrace(
            node_name="plan",
            input={"system_len": len(sys_p), "user_len": len(user_p),
                   "condition": cfg.condition.value},
            output={"raw_len": len(r_plan.text), "final_path_len": len(plan_line),
                    "parsed_cells": len(plan_parsed.parsed) if plan_parsed.parsed else 0,
                    "parse_error": plan_parsed.parse_error},
            valid=plan_parsed.parsed is not None,
            wall_ms=r_plan.wall_ms, tokens_in=r_plan.tokens_in, tokens_out=r_plan.tokens_out,
        ))
        tok_in += r_plan.tokens_in; tok_out += r_plan.tokens_out

        current_text = r_plan.text
        current_line = plan_line
        current_parsed = plan_parsed

        for round_idx in range(1, max_rounds + 1):
            t_v0 = _time.perf_counter()
            issues = check_path_issues(task, current_parsed.parsed, cost_grid)
            issues_text = format_issues(issues)
            v_ms = int((_time.perf_counter() - t_v0) * 1000)
            traces.append(NodeTrace(
                node_name=f"verify_{round_idx}",
                input={"tool": "check_path_issues", "plan_cells": len(current_parsed.parsed) if current_parsed.parsed else 0},
                output={"issues_found": len(issues), "issues_first_three": issues[:3]},
                valid=len(issues) == 0,
                wall_ms=v_ms, tokens_in=0, tokens_out=0,
            ))
            if not issues:
                wall_ms = int((_time.perf_counter() - t0) * 1000)
                return current_parsed.model_copy(update={"raw": current_text}), traces, tok_in, tok_out, wall_ms

            sys_r, user_r = render_revise_prompt(task, cfg.condition, current_text, issues_text, cfg.prompt_version)
            r_rev = self.provider.complete(sys_r, user_r, cfg)
            rev_line = _extract_final_path(r_rev.text)
            rev_parsed = parse_route_answer(rev_line)
            traces.append(NodeTrace(
                node_name=f"revise_{round_idx}",
                input={"system_len": len(sys_r), "user_len": len(user_r),
                       "verified_issues": len(issues), "round": round_idx},
                output={"raw_len": len(r_rev.text), "final_path_len": len(rev_line),
                        "parsed_cells": len(rev_parsed.parsed) if rev_parsed.parsed else 0,
                        "parse_error": rev_parsed.parse_error,
                        "changed_from_previous": rev_line.strip() != current_line.strip()},
                valid=rev_parsed.parsed is not None,
                wall_ms=r_rev.wall_ms, tokens_in=r_rev.tokens_in, tokens_out=r_rev.tokens_out,
            ))
            tok_in += r_rev.tokens_in; tok_out += r_rev.tokens_out

            current_text = r_rev.text
            current_line = rev_line
            current_parsed = rev_parsed

        wall_ms = int((_time.perf_counter() - t0) * 1000)
        return current_parsed.model_copy(update={"raw": current_text}), traces, tok_in, tok_out, wall_ms


TOOL_LOOP_SYSTEM = """You are a spatial planner. You MUST use the provided tools.
Do NOT describe calling tools in prose. Do NOT paste JSON tool arguments as code
blocks. INVOKE the tools directly through the tool-calling interface.

TOOLS:
  solve_between(from_row, from_col, to_row, to_col)
      Optimal Dijkstra path between ANY two cells on the grid. Returns
      {ok, cells, cost}. This is the primary tool — use it aggressively.

  check_path(cells)
      Validates a full proposed path. Returns {issues, valid}. Use it once,
      on your final path, right before emitting FINAL PATH.

RECOMMENDED WORKFLOW (fastest, most reliable):
  1. Call solve_between(start_row, start_col, goal_row, goal_col) ONCE
     to get the optimal path directly. This one call solves most tasks.
  2. Call check_path(cells) with the result to confirm validity.
  3. Emit the FINAL PATH line.

DO NOT:
  - Call solve_between on single-step segments like (r, c) -> (r, c+1).
    That is useless. Ask for the WHOLE path, or big multi-cell segments.
  - Write "```json {...}```" blocks pretending to be tool calls. Use the
    real tool interface.
  - Skip the tools and hand-write a path. It will be wrong.

OUTPUT FORMAT (parser reads only this line):
FINAL PATH: (r0, c0) -> (r1, c1) -> ... -> (rN, cN)

Rules the grader enforces:
- 8-connected moves (diagonals allowed).
- Diagonal step cost = sqrt(2) * destination cost.
- Cells marked "inf" are impassable — never step on them.
- First cell must equal start; last cell must equal goal.
"""


class ToolLoopAgent:
    NAME = "tool_loop"
    DEFAULT_MAX_TOOL_STEPS = 16

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def run(
        self,
        task: TaskInstance,
        cfg: AgentConfig,
        cost_grid=None,
    ) -> tuple[ParsedAnswer, list[NodeTrace], int, int, int]:
        if cost_grid is None:
            raise ValueError("tool_loop needs cost_grid — executor must pass it")

        import json as _json
        import time as _time

        max_steps = int(cfg.provider_kwargs.get("max_tool_steps", self.DEFAULT_MAX_TOOL_STEPS))

        t0 = _time.perf_counter()
        traces: list[NodeTrace] = []
        tok_in = tok_out = 0

        _, user_body = render_prompt(task, cfg.condition, cfg.prompt_version)
        messages: list[dict] = [
            {"role": "system", "content": TOOL_LOOP_SYSTEM},
            {"role": "user", "content": user_body},
        ]
        tools = tool_specs_for_access(cfg.tool_access.value)
        max_seg = cfg.max_segment_cells if cfg.max_segment_cells is not None else max_segment_for_access(cfg.tool_access.value)

        final_text = ""
        stopped_reason = "max_steps"
        for step in range(1, max_steps + 1):
            if cfg.max_input_tokens_per_task and tok_in >= cfg.max_input_tokens_per_task:
                stopped_reason = "token_budget"
                break
            resp = self.provider.complete_with_tools(messages, tools, cfg)
            tok_in += resp.tokens_in; tok_out += resp.tokens_out

            calls_summary = [
                {"name": tc["name"], "args_keys": sorted(tc["arguments"].keys())}
                for tc in (resp.tool_calls or [])
            ]
            traces.append(NodeTrace(
                node_name=f"llm_step_{step}",
                input={"messages_so_far": len(messages)},
                output={"finish_reason": resp.finish_reason,
                        "tool_calls": calls_summary,
                        "text_len": len(resp.text)},
                valid=None,
                wall_ms=resp.wall_ms,
                tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
            ))

            if resp.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": resp.text or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": _json.dumps(tc["arguments"])},
                        }
                        for tc in resp.tool_calls
                    ],
                }
                messages.append(assistant_msg)
                for tc in resp.tool_calls:
                    t_t0 = _time.perf_counter()
                    result = execute_tool_call(tc["name"], tc["arguments"], task, cost_grid, max_segment_cells=max_seg)
                    t_ms = int((_time.perf_counter() - t_t0) * 1000)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": _json.dumps(result),
                    })
                    traces.append(NodeTrace(
                        node_name=f"tool_{tc['name']}_step_{step}",
                        input={"arguments": tc["arguments"]},
                        output=_summarize_tool_result(tc["name"], result),
                        valid=result.get("valid") if isinstance(result, dict) else None,
                        wall_ms=t_ms, tokens_in=0, tokens_out=0,
                    ))
                messages = _truncate_messages(messages, keep_last_n=cfg.message_window)
                continue

            final_text = resp.text
            stopped_reason = "final"
            break

        parsed = parse_route_answer(_extract_final_path(final_text)).model_copy(
            update={"raw": final_text}
        )
        traces.append(NodeTrace(
            node_name="terminal",
            input={"stopped_reason": stopped_reason},
            output={"parsed_cells": len(parsed.parsed) if parsed.parsed else 0,
                    "parse_error": parsed.parse_error},
            valid=parsed.parsed is not None,
            wall_ms=0, tokens_in=0, tokens_out=0,
        ))

        wall_ms = int((_time.perf_counter() - t0) * 1000)
        return parsed, traces, tok_in, tok_out, wall_ms


def _truncate_messages(messages: list, keep_last_n: int) -> list:
    if keep_last_n <= 0 or len(messages) <= 3:
        return messages

    head_end = 0
    while head_end < len(messages) and messages[head_end]["role"] != "assistant":
        head_end += 1
    if head_end >= len(messages):
        return messages

    roundtrips: list[tuple[int, int]] = []
    i = head_end
    while i < len(messages):
        if messages[i]["role"] != "assistant":
            i += 1
            continue
        start = i
        i += 1
        while i < len(messages) and messages[i]["role"] == "tool":
            i += 1
        roundtrips.append((start, i))

    if len(roundtrips) <= keep_last_n:
        return messages

    kept = roundtrips[-keep_last_n:]
    dropped = len(roundtrips) - keep_last_n
    head = messages[:head_end]
    summary = {
        "role": "system",
        "content": f"[{dropped} earlier tool round-trips truncated from history]",
    }
    tail: list = []
    for start, end in kept:
        tail.extend(messages[start:end])
    return head + [summary] + tail


def _summarize_tool_result(name: str, result: dict) -> dict:
    if name == "check_path":
        issues = result.get("issues", [])
        return {"n_issues": len(issues), "first_three": issues[:3], "valid": result.get("valid")}
    if name == "solve_between":
        if result.get("ok"):
            return {"ok": True, "n_cells": len(result.get("cells", [])), "cost": result.get("cost")}
        return {"ok": False, "error": result.get("error")}
    return {"raw_keys": list(result.keys()) if isinstance(result, dict) else "not_dict"}


AGENTS = {
    SingleShotAgent.NAME: SingleShotAgent,
    PlanCritiqueReviseAgent.NAME: PlanCritiqueReviseAgent,
    PlanVerifyReviseAgent.NAME: PlanVerifyReviseAgent,
    ToolLoopAgent.NAME: ToolLoopAgent,
}
