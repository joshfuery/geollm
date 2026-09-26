from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core import FailureCategory, ParsedAnswer, RunRecord, ScoreBundle, TaskInstance
from ..scorers.route_scorer import score_route_answer
from ..solvers.route_solver import solve_and_attach
from ..tasks.route import (
    RouteParams,
    generate_route_from_dem,
    generate_synthetic_route_task,
)
from .agents import AGENTS
from .config import RunConfig
from .prompts import approx_tokens, render_prompt
from .providers import (
    InfrastructureError,
    LLMProvider,
    openai_cost_usd,
)
from .store import Store


def build_task(family: str, seed: int, grid_size: int, region: str | None = None):
    if family != "route":
        raise ValueError(f"family {family!r} not yet implemented")
    if region:
        task, cost, elev = generate_route_from_dem(
            region=region, seed=seed, grid_size=grid_size,
        )
    else:
        task, cost, elev = generate_synthetic_route_task(
            seed=seed, params=RouteParams(grid_size=grid_size),
        )
    task = solve_and_attach(task, cost)
    return task, cost, elev


@dataclass
class RunOutcome:
    record: RunRecord
    task: TaskInstance
    cost_grid: np.ndarray


def _new_run_id() -> str:
    return "run_" + uuid.uuid4().hex[:12]


def _sleep_backoff(base_s: float, attempt: int) -> None:
    time.sleep(base_s * (2 ** (attempt - 1)))


def execute_one(
    task: TaskInstance,
    cost_grid: np.ndarray,
    provider: LLMProvider,
    cfg: RunConfig,
) -> RunOutcome:
    agent_cls = AGENTS.get(cfg.agent.topology.value)
    if agent_cls is None:
        raise ValueError(f"no agent registered for topology {cfg.agent.topology.value!r}")
    agent = agent_cls(provider)
    last_infra_error: str | None = None

    for attempt in range(1, cfg.max_infra_retries + 1):
        try:
            parsed, traces, tokens_in, tokens_out, wall_ms = agent.run(task, cfg.agent, cost_grid=cost_grid)
        except InfrastructureError as e:
            last_infra_error = str(e)
            if attempt < cfg.max_infra_retries:
                _sleep_backoff(cfg.infra_retry_backoff_s, attempt)
                continue
            return RunOutcome(
                record=RunRecord(
                    run_id=_new_run_id(),
                    task_id=task.task_id,
                    config_id=cfg.agent.config_id,
                    answer=ParsedAnswer(raw="", parse_error=last_infra_error),
                    score=ScoreBundle(
                        valid=False,
                        failure=FailureCategory.NONE,
                        notes=f"infrastructure: {last_infra_error}",
                    ),
                    node_traces=[],
                    tokens_in=0, tokens_out=0, wall_ms=0,
                    error=last_infra_error,
                ),
                task=task, cost_grid=cost_grid,
            )
        score = score_route_answer(task, parsed, cost_grid)
        return RunOutcome(
            record=RunRecord(
                run_id=_new_run_id(),
                task_id=task.task_id,
                config_id=cfg.agent.config_id,
                answer=parsed,
                score=score,
                node_traces=traces,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                wall_ms=wall_ms,
                error=None,
            ),
            task=task, cost_grid=cost_grid,
        )

    raise RuntimeError("execute_one exited its retry loop unexpectedly")


@dataclass
class SuiteResult:
    outcomes: list[RunOutcome]
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_wall_ms: int = 0

    @property
    def n_valid(self) -> int:
        return sum(1 for o in self.outcomes if o.record.score.valid)

    @property
    def median_regret(self) -> float | None:
        rs = [o.record.score.regret for o in self.outcomes if o.record.score.valid]
        return float(np.median(rs)) if rs else None


def run_suite(
    provider: LLMProvider,
    cfg: RunConfig,
    store: Store | None = None,
    on_progress: Callable[[int, int, RunOutcome], None] | None = None,
    region: str | None = None,
) -> SuiteResult:
    result = SuiteResult(outcomes=[])
    if store is not None:
        store.upsert_config(cfg.agent)

    n = len(cfg.seeds) * cfg.k
    i = 0
    for seed in cfg.seeds:
        task, cost, _elev = build_task(cfg.family, seed=seed, grid_size=cfg.grid_size, region=region)
        if store is not None:
            store.upsert_task(task)
        for k_idx in range(cfg.k):
            i += 1
            outcome = execute_one(task, cost, provider, cfg)
            result.outcomes.append(outcome)
            result.total_tokens_in += outcome.record.tokens_in
            result.total_tokens_out += outcome.record.tokens_out
            result.total_wall_ms += outcome.record.wall_ms
            if store is not None:
                store.write_run(outcome.record, outcome.record.node_traces)
            if on_progress:
                on_progress(i, n, outcome)
    return result


@dataclass
class CostEstimate:
    n_calls: int
    tokens_in_total: int
    tokens_out_total: int
    est_usd: float | None
    per_task_tokens_in: int
    per_task_tokens_out: int


def estimate_cost(cfg: RunConfig, region: str | None = None) -> CostEstimate:
    if not cfg.seeds:
        return CostEstimate(0, 0, 0, 0.0, 0, 0)

    task, _cost, _elev = build_task(cfg.family, seed=cfg.seeds[0], grid_size=cfg.grid_size, region=region)
    system, user = render_prompt(task, cfg.agent.condition, cfg.agent.prompt_version)
    tokens_in = approx_tokens(system) + approx_tokens(user)
    tokens_out = min(cfg.agent.max_output_tokens, max(200, 20 * cfg.grid_size))
    if cfg.agent.topology.value == "plan_verify_revise":
        calls_per_task = 1 + cfg.agent.max_revise_rounds
    elif cfg.agent.topology.value == "tool_loop":
        calls_per_task = 4
    else:
        calls_per_task = {
            "single_shot": 1,
            "plan_critique_revise": 3,
            "plan_execute": 2,
        }.get(cfg.agent.topology.value, 1)
    n = len(cfg.seeds) * cfg.k * calls_per_task
    total_in = tokens_in * n
    total_out = tokens_out * n
    usd = openai_cost_usd(cfg.agent.model, total_in, total_out)
    return CostEstimate(
        n_calls=n,
        tokens_in_total=total_in,
        tokens_out_total=total_out,
        est_usd=usd,
        per_task_tokens_in=tokens_in,
        per_task_tokens_out=tokens_out,
    )
