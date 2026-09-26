from __future__ import annotations

import re

import numpy as np
import pytest

from geoagentbench.core import FailureCategory
from geoagentbench.runner.config import (
    AgentConfig, Condition, RunConfig, Topology, ToolAccess,
)
from geoagentbench.runner.executor import (
    build_task, estimate_cost, execute_one, run_suite,
)
from geoagentbench.runner.providers import (
    InfrastructureError, LLMResponse, MockProvider,
)
from geoagentbench.runner.store import Store
from geoagentbench.solvers.route_solver import dijkstra_route


def _oracle_responder_factory():
    def responder(system, user, cfg):
        m = re.search(r'"start":\s*\[(\d+),\s*(\d+)\]', user)
        assert m, "responder needs structured prompt"
        start = (int(m.group(1)), int(m.group(2)))
        m = re.search(r'"goal":\s*\[(\d+),\s*(\d+)\]', user)
        assert m
        goal = (int(m.group(1)), int(m.group(2)))
        gm = re.search(r'"cost_grid":\s*(\[\[.+?\]\])', user)
        assert gm
        import json
        raw_grid = json.loads(gm.group(1).replace('"inf"', "1e18"))
        grid = np.array(raw_grid, dtype=float)
        grid[grid >= 1e18] = np.inf
        path = dijkstra_route(grid, start, goal, connectivity=8)
        cells_str = " -> ".join(f"({r}, {c})" for r, c in path.cells)
        return f"Thinking...\n\nFINAL PATH: {cells_str}"
    return responder


def _straight_line_responder(system, user, cfg):
    m = re.search(r'"start":\s*\[(\d+),\s*(\d+)\]', user)
    start = (int(m.group(1)), int(m.group(2)))
    m = re.search(r'"goal":\s*\[(\d+),\s*(\d+)\]', user)
    goal = (int(m.group(1)), int(m.group(2)))
    return f"FINAL PATH: ({start[0]}, {start[1]}) -> ({goal[0]}, {goal[1]})"


def _gibberish_responder(system, user, cfg):
    return "I refuse."


def test_config_id_stable_and_sensitive():
    a = AgentConfig(model="gpt-4o-mini", condition=Condition.STRUCTURED_ONLY)
    b = AgentConfig(model="gpt-4o-mini", condition=Condition.STRUCTURED_ONLY)
    c = AgentConfig(model="gpt-4o-mini", condition=Condition.NARRATIVE_ONLY)
    d = AgentConfig(model="gpt-4o",       condition=Condition.STRUCTURED_ONLY)
    e = AgentConfig(model="gpt-4o-mini", condition=Condition.STRUCTURED_ONLY,
                    provider="mock:oracle")
    assert a.config_id == b.config_id
    assert a.config_id != c.config_id
    assert a.config_id != d.config_id
    assert a.config_id != e.config_id


def test_oracle_responder_scores_zero_regret():
    provider = MockProvider(responder=_oracle_responder_factory())
    task, cost, _ = build_task("route", seed=7, grid_size=12)
    cfg = RunConfig(
        seeds=[7], k=1, grid_size=12,
        agent=AgentConfig(model="mock", condition=Condition.STRUCTURED_ONLY),
    )
    outcome = execute_one(task, cost, provider, cfg)
    assert outcome.record.score.valid
    assert outcome.record.score.regret == pytest.approx(0.0, abs=1e-9)
    assert outcome.record.score.failure == FailureCategory.NONE


def test_straight_line_responder_flagged_as_infeasible():
    provider = MockProvider(responder=_straight_line_responder)
    task, cost, _ = build_task("route", seed=3, grid_size=10)
    cfg = RunConfig(
        seeds=[3], k=1, grid_size=10,
        agent=AgentConfig(model="mock"),
    )
    outcome = execute_one(task, cost, provider, cfg)
    assert not outcome.record.score.valid
    assert outcome.record.score.failure == FailureCategory.INFEASIBLE_STEP


def test_gibberish_responder_flagged_as_format_violation():
    provider = MockProvider(responder=_gibberish_responder)
    task, cost, _ = build_task("route", seed=3, grid_size=10)
    cfg = RunConfig(seeds=[3], k=1, grid_size=10, agent=AgentConfig(model="mock"))
    outcome = execute_one(task, cost, provider, cfg)
    assert not outcome.record.score.valid
    assert outcome.record.score.failure == FailureCategory.FORMAT_VIOLATION


class _FlakyProvider:
    def __init__(self, n_fails: int, then=None):
        self.n_fails = n_fails
        self.calls = 0
        self._then = then or _straight_line_responder

    def complete(self, system, user, cfg):
        self.calls += 1
        if self.calls <= self.n_fails:
            raise InfrastructureError("simulated rate limit")
        text = self._then(system, user, cfg)
        return LLMResponse(text=text, tokens_in=1, tokens_out=1, wall_ms=1)


def test_executor_retries_infra_errors():
    provider = _FlakyProvider(n_fails=2)
    task, cost, _ = build_task("route", seed=1, grid_size=8)
    cfg = RunConfig(
        seeds=[1], k=1, grid_size=8,
        agent=AgentConfig(model="mock"),
        max_infra_retries=5,
        infra_retry_backoff_s=0.0,
    )
    outcome = execute_one(task, cost, provider, cfg)
    assert provider.calls == 3
    assert outcome.record.error is None


def test_executor_gives_up_after_max_infra_retries():
    provider = _FlakyProvider(n_fails=99)
    task, cost, _ = build_task("route", seed=1, grid_size=8)
    cfg = RunConfig(
        seeds=[1], k=1, grid_size=8,
        agent=AgentConfig(model="mock"),
        max_infra_retries=2,
        infra_retry_backoff_s=0.0,
    )
    outcome = execute_one(task, cost, provider, cfg)
    assert provider.calls == 2
    assert outcome.record.error is not None
    assert "simulated" in outcome.record.error
    assert not outcome.record.score.valid


def test_run_suite_persists_and_reads(tmp_path):
    provider = MockProvider(responder=_oracle_responder_factory())
    cfg = RunConfig(
        seeds=[0, 1, 2], k=2, grid_size=10,
        agent=AgentConfig(model="mock-oracle", condition=Condition.STRUCTURED_ONLY),
    )
    store = Store.open(tmp_path / "results")
    result = run_suite(provider, cfg, store=store)

    assert len(result.outcomes) == 6
    assert result.n_valid == 6
    assert result.median_regret == pytest.approx(0.0, abs=1e-9)

    rows = store.list_runs()
    assert len(rows) == 6
    assert all(r["valid"] == 1 for r in rows)

    first = rows[0]
    traces = store.read_traces(first["run_id"])
    assert len(traces) == 1
    assert traces[0]["node_name"] == "single_shot_llm"
    assert traces[0]["valid"] is True

    cfgs = store.list_configs()
    assert len(cfgs) == 1
    assert cfgs[0]["model"] == "mock-oracle"


def test_estimate_cost_scales_with_seeds_and_k():
    cfg = RunConfig(seeds=[0, 1], k=2, grid_size=16, agent=AgentConfig(model="gpt-4o-mini"))
    est = estimate_cost(cfg)
    assert est.n_calls == 4
    assert est.tokens_in_total > 0 and est.tokens_out_total > 0
    assert est.est_usd is not None and est.est_usd > 0


def test_plan_critique_revise_records_three_node_traces():
    from geoagentbench.runner.mock_agents import oracle_responder

    def responder(system, user, cfg):
        if "path validator" in system.lower():
            return "ISSUES:"
        return oracle_responder(system, user, cfg)

    provider = MockProvider(responder=responder)
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=10,
        agent=AgentConfig(
            model="mock-pcr", condition=Condition.STRUCTURED_ONLY,
            topology=Topology.PLAN_CRITIQUE_REVISE,
        ),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=10)
    outcome = execute_one(task, cost, provider, cfg)

    assert outcome.record.score.valid
    assert outcome.record.score.regret == pytest.approx(0.0, abs=1e-9)
    assert len(outcome.record.node_traces) == 3
    names = [t.node_name for t in outcome.record.node_traces]
    assert names == ["plan", "critique", "revise"]
    assert outcome.record.tokens_in > 0
    assert outcome.record.tokens_out > 0


def test_plan_critique_revise_fixes_bad_plan():
    from geoagentbench.runner.mock_agents import (
        oracle_responder, straight_line_responder,
    )

    call_state = {"n": 0}
    def responder(system, user, cfg):
        call_state["n"] += 1
        if call_state["n"] == 1:
            return straight_line_responder(system, user, cfg)
        if call_state["n"] == 2:
            return "ISSUES:\n- non-adjacent step from start to goal"
        return oracle_responder(system, user, cfg)

    provider = MockProvider(responder=responder)
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=8,
        agent=AgentConfig(
            model="mock-pcr", condition=Condition.STRUCTURED_ONLY,
            topology=Topology.PLAN_CRITIQUE_REVISE,
        ),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=8)
    outcome = execute_one(task, cost, provider, cfg)

    assert outcome.record.score.valid
    plan_trace = outcome.record.node_traces[0]
    critique_trace = outcome.record.node_traces[1]
    revise_trace = outcome.record.node_traces[2]
    assert plan_trace.node_name == "plan"
    assert critique_trace.output["issues_found"] >= 1
    assert revise_trace.output["changed_from_plan"] is True


def test_verify_short_circuits_when_plan_is_valid():
    from geoagentbench.runner.mock_agents import oracle_responder

    calls = {"n": 0}
    def responder(system, user, cfg):
        calls["n"] += 1
        return oracle_responder(system, user, cfg)

    provider = MockProvider(responder=responder)
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=8,
        agent=AgentConfig(
            model="mock-verify", condition=Condition.STRUCTURED_ONLY,
            topology=Topology.PLAN_VERIFY_REVISE,
        ),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=8)
    outcome = execute_one(task, cost, provider, cfg)

    assert outcome.record.score.valid
    assert outcome.record.score.regret == pytest.approx(0.0, abs=1e-9)
    assert calls["n"] == 1
    names = [t.node_name for t in outcome.record.node_traces]
    assert names == ["plan", "verify_1"]
    assert outcome.record.node_traces[1].valid is True


def test_verify_finds_real_issues_and_revise_gets_them():
    from geoagentbench.runner.mock_agents import (
        oracle_responder, straight_line_responder,
    )

    calls = {"n": 0, "revise_saw_issues": None}
    def responder(system, user, cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            return straight_line_responder(system, user, cfg)
        if "Validator's critique" in user or "verified issues" in user or "ISSUES:" in user:
            calls["revise_saw_issues"] = user
        return oracle_responder(system, user, cfg)

    provider = MockProvider(responder=responder)
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=8,
        agent=AgentConfig(
            model="mock-verify", condition=Condition.STRUCTURED_ONLY,
            topology=Topology.PLAN_VERIFY_REVISE,
        ),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=8)
    outcome = execute_one(task, cost, provider, cfg)

    assert outcome.record.score.valid
    assert calls["n"] == 2

    verify_trace = outcome.record.node_traces[1]
    assert verify_trace.node_name == "verify_1"
    assert verify_trace.output["issues_found"] >= 1
    assert verify_trace.tokens_in == 0 and verify_trace.tokens_out == 0

    assert calls["revise_saw_issues"] is not None
    assert "ISSUES:" in calls["revise_saw_issues"]


def test_check_path_issues_matches_scorer_categories():
    from geoagentbench.runner.tools import check_path_issues

    task, cost, _ = build_task("route", seed=3, grid_size=8)
    start = tuple(task.params["start"])
    goal = tuple(task.params["goal"])

    bad = [start, goal]
    issues = check_path_issues(task, bad, cost)
    assert any("non-adjacent" in i for i in issues)

    bad2 = [start, (999, 999)]
    issues2 = check_path_issues(task, bad2, cost)
    assert any("out of bounds" in i for i in issues2)

    assert check_path_issues(task, [], cost) == ["path is empty"]
    assert check_path_issues(task, None, cost) == ["path could not be parsed from FINAL PATH line"]


def test_verify_revise_iterates_until_valid():
    from geoagentbench.runner.mock_agents import (
        oracle_responder, straight_line_responder,
    )

    call_state = {"n": 0}
    def responder(system, user, cfg):
        call_state["n"] += 1
        if call_state["n"] == 1:
            return straight_line_responder(system, user, cfg)
        if call_state["n"] == 2:
            import re
            ms = re.search(r'"start":\s*\[(\d+),\s*(\d+)\]', user)
            mg = re.search(r'"goal":\s*\[(\d+),\s*(\d+)\]', user)
            start = (int(ms.group(1)), int(ms.group(2)))
            goal = (int(mg.group(1)), int(mg.group(2)))
            mid1 = (start[0] + 1, start[1] + 1)
            mid_bad = (goal[0] - 1, goal[1] - 1)
            cells_str = " -> ".join(f"({r}, {c})" for r, c in [start, mid1, mid_bad, goal])
            return f"FINAL PATH: {cells_str}"
        return oracle_responder(system, user, cfg)

    provider = MockProvider(responder=responder)
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=10,
        agent=AgentConfig(
            model="mock-iter", condition=Condition.STRUCTURED_ONLY,
            topology=Topology.PLAN_VERIFY_REVISE,
            max_revise_rounds=3,
        ),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=10)
    outcome = execute_one(task, cost, provider, cfg)

    assert outcome.record.score.valid
    assert call_state["n"] == 3

    names = [t.node_name for t in outcome.record.node_traces]
    assert names[0] == "plan"
    assert "verify_1" in names and "verify_2" in names and "verify_3" in names
    assert "revise_1" in names and "revise_2" in names
    last_verify = [t for t in outcome.record.node_traces if t.node_name.startswith("verify_")][-1]
    assert last_verify.output["issues_found"] == 0


def test_max_revise_rounds_1_matches_old_behavior():
    from geoagentbench.runner.mock_agents import straight_line_responder

    def responder(system, user, cfg):
        return straight_line_responder(system, user, cfg)

    provider = MockProvider(responder=responder)
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=8,
        agent=AgentConfig(model="mock", topology=Topology.PLAN_VERIFY_REVISE),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=8)
    outcome = execute_one(task, cost, provider, cfg)

    names = [t.node_name for t in outcome.record.node_traces]
    assert names == ["plan", "verify_1", "revise_1"]
    assert not outcome.record.score.valid


def test_tool_loop_uses_solve_between_to_recover():
    call_state = {"n": 0, "last_solved": None}

    def responder(system, user, cfg):
        call_state["n"] += 1
        if call_state["n"] == 1:
            import re
            ms = re.search(r'"start":\s*\[(\d+),\s*(\d+)\]', user)
            mg = re.search(r'"goal":\s*\[(\d+),\s*(\d+)\]', user)
            return {"tool_calls": [{
                "name": "solve_between",
                "arguments": {
                    "from_row": int(ms.group(1)), "from_col": int(ms.group(2)),
                    "to_row":   int(mg.group(1)), "to_col":   int(mg.group(2)),
                },
            }]}
        if call_state["n"] == 2:
            import re, json
            m = re.search(r'"cells":\s*(\[\[.*?\]\])', user)
            assert m, f"expected cells in tool result, got: {user[:200]}"
            cells = json.loads(m.group(1))
            call_state["last_solved"] = cells
            return "FINAL PATH: " + " -> ".join(f"({r}, {c})" for r, c in cells)
        return "FINAL PATH:"

    provider = MockProvider(responder=responder)
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=10,
        agent=AgentConfig(
            model="mock-tool", condition=Condition.STRUCTURED_ONLY,
            topology=Topology.TOOL_LOOP,
        ),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=10)
    outcome = execute_one(task, cost, provider, cfg)

    assert outcome.record.score.valid
    assert outcome.record.score.regret == pytest.approx(0.0, abs=1e-9)
    assert call_state["last_solved"] is not None

    names = [t.node_name for t in outcome.record.node_traces]
    assert "llm_step_1" in names
    assert any(n.startswith("tool_solve_between") for n in names)
    assert names[-1] == "terminal"


def test_tool_loop_check_path_finds_no_issues_on_valid_path():
    from geoagentbench.runner.tools import execute_tool_call
    from geoagentbench.solvers.route_solver import dijkstra_route

    task, cost, _ = build_task("route", seed=1, grid_size=10)
    start = tuple(task.params["start"])
    goal = tuple(task.params["goal"])
    optimal = dijkstra_route(cost, start, goal, connectivity=8)

    result = execute_tool_call(
        "check_path",
        {"cells": [list(c) for c in optimal.cells]},
        task, cost,
    )
    assert result["valid"] is True
    assert result["issues"] == []


def test_solve_between_returns_valid_subpath():
    from geoagentbench.runner.tools import solve_between
    task, cost, _ = build_task("route", seed=1, grid_size=10)
    n = cost.shape[0]
    from geoagentbench.solvers.route_solver import dijkstra_route
    for a, b in [((0, 0), (n - 1, n - 1)), ((0, n - 1), (n - 1, 0))]:
        result = solve_between(task, a, b, cost)
        if result["ok"]:
            direct = dijkstra_route(cost, a, b, connectivity=8)
            assert abs(result["cost"] - direct.cost) < 1e-9
            return
    pytest.fail("no passable pair found — should be extremely unlikely")


def test_tool_loop_terminates_at_max_steps():
    def endless(system, user, cfg):
        return {"tool_calls": [{
            "name": "check_path",
            "arguments": {"cells": [[0, 0]]},
        }]}
    provider = MockProvider(responder=endless)
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=8,
        agent=AgentConfig(
            model="mock", condition=Condition.STRUCTURED_ONLY,
            topology=Topology.TOOL_LOOP,
            provider_kwargs={"max_tool_steps": 3},
        ),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=8)
    outcome = execute_one(task, cost, provider, cfg)
    assert not outcome.record.score.valid
    term = outcome.record.node_traces[-1]
    assert term.node_name == "terminal"
    assert term.input["stopped_reason"] == "max_steps"


def test_execute_tool_call_accepts_alternate_shapes():
    from geoagentbench.runner.tools import execute_tool_call

    task, cost, _ = build_task("route", seed=0, grid_size=8)

    r1 = execute_tool_call("solve_between",
        {"from_row": 0, "from_col": 0, "to_row": 7, "to_col": 7}, task, cost)
    r2 = execute_tool_call("solve_between",
        {"from": [0, 0], "to": [7, 7]}, task, cost)
    r3 = execute_tool_call("solve_between",
        {"start_row": 0, "start_col": 0, "end_row": 7, "end_col": 7}, task, cost)
    r4 = execute_tool_call("solve_between", {"garbage": True}, task, cost)

    for r in (r1, r2, r3):
        assert r.get("ok") is True, f"expected ok=True, got {r}"
    assert r4.get("ok") is False and "could not parse" in r4.get("error", "")


def test_restricted_solver_rejects_oversize_segment():
    from geoagentbench.runner.tools import (
        execute_tool_call, max_segment_for_access,
    )

    task, cost, _ = build_task("route", seed=0, grid_size=16)
    cap = max_segment_for_access("restricted_solver")
    assert cap == 5

    start = tuple(task.params["start"])
    goal = tuple(task.params["goal"])
    result = execute_tool_call(
        "solve_between",
        {"from_row": start[0], "from_col": start[1], "to_row": goal[0], "to_col": goal[1]},
        task, cost, max_segment_cells=cap,
    )
    assert result["ok"] is False
    assert "segment too long" in result["error"]

    r2 = execute_tool_call(
        "solve_between",
        {"from_row": start[0], "from_col": start[1], "to_row": start[0], "to_col": start[1] + 1},
        task, cost, max_segment_cells=cap,
    )
    assert r2["ok"] is True


def test_tool_specs_scale_with_access():
    from geoagentbench.runner.tools import tool_specs_for_access
    assert tool_specs_for_access("none") == []
    assert len(tool_specs_for_access("calculator")) == 1
    assert len(tool_specs_for_access("restricted_solver")) == 2
    assert len(tool_specs_for_access("full_solver")) == 2


def test_tool_loop_respects_token_budget():

    class BigProvider(MockProvider):
        def complete_with_tools(self, messages, tools, cfg):
            resp = super().complete_with_tools(messages, tools, cfg)
            from dataclasses import replace
            return replace(resp, tokens_in=100_000, tokens_out=10)

    provider = BigProvider(responder=lambda s, u, c: {"tool_calls": [{
        "name": "solve_between",
        "arguments": {"from_row": 0, "from_col": 0, "to_row": 5, "to_col": 5},
    }]})
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=8,
        agent=AgentConfig(
            model="mock", topology=Topology.TOOL_LOOP,
            max_input_tokens_per_task=50_000,
        ),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=8)
    outcome = execute_one(task, cost, provider, cfg)
    term = outcome.record.node_traces[-1]
    assert term.node_name == "terminal"
    assert term.input["stopped_reason"] == "token_budget"


def test_message_truncation_preserves_round_trip_pairs():
    from geoagentbench.runner.agents import _truncate_messages

    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "u"}]
    for i in range(20):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "check_path", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}a"})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}b"})

    trimmed = _truncate_messages(msgs, keep_last_n=3)
    assert len(trimmed) == 2 + 1 + 3 * 3
    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["role"] == "user"
    assert "truncated" in trimmed[2]["content"]

    for i, m in enumerate(trimmed):
        if m["role"] == "tool":
            j = i - 1
            while j >= 0 and trimmed[j]["role"] == "tool":
                j -= 1
            assert j >= 0 and trimmed[j]["role"] == "assistant", (
                f"tool at index {i} orphaned from any assistant"
            )

    assert _truncate_messages(msgs, keep_last_n=0) == msgs


def test_message_truncation_never_orphans_tools():
    from geoagentbench.runner.agents import _truncate_messages

    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "u"}]
    for i in range(4):
        msgs.append({"role": "assistant", "tool_calls": [
            {"id": f"a{i}", "type": "function",
             "function": {"name": "solve_between", "arguments": "{}"}}
        ], "content": None})
        for j in range(3):
            msgs.append({"role": "tool", "tool_call_id": f"a{i}", "content": f"{i}_{j}"})

    trimmed = _truncate_messages(msgs, keep_last_n=2)
    for i, m in enumerate(trimmed):
        if m["role"] == "tool":
            j = i - 1
            while j >= 0 and trimmed[j]["role"] == "tool":
                j -= 1
            assert j >= 0 and trimmed[j]["role"] == "assistant"


def test_max_segment_cells_override_wins():

    def responder(system, user, cfg):
        if "cells" in user and "cost" in user:
            import re, json
            m = re.search(r'"cells":\s*(\[\[.*?\]\])', user)
            if m:
                cells = json.loads(m.group(1))
                return "FINAL PATH: " + " -> ".join(f"({r}, {c})" for r, c in cells)
        return {"tool_calls": [{
            "name": "solve_between",
            "arguments": {"from_row": 0, "from_col": 0, "to_row": 7, "to_col": 7},
        }]}

    provider = MockProvider(responder=responder)
    cfg = RunConfig(
        seeds=[0], k=1, grid_size=8,
        agent=AgentConfig(
            model="mock", topology=Topology.TOOL_LOOP,
            tool_access=ToolAccess.RESTRICTED_SOLVER,
            max_segment_cells=2,
        ),
    )
    task, cost, _ = build_task("route", seed=0, grid_size=8)
    outcome = execute_one(task, cost, provider, cfg)
    tool_traces = [t for t in outcome.record.node_traces
                   if t.node_name.startswith("tool_solve_between")]
    assert tool_traces
    assert tool_traces[0].output.get("ok") is False
    assert "segment too long" in tool_traces[0].output.get("error", "")
