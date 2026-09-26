from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from ..core import Cell, FailureCategory, ParsedAnswer, ScoreBundle, TaskInstance

_CELL_RE = re.compile(r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]")


def parse_route_answer(raw: str) -> ParsedAnswer:
    matches = _CELL_RE.findall(raw)
    if not matches:
        return ParsedAnswer(raw=raw, parse_error="no (r, c) tuples found in answer")
    cells: list[Cell] = [(int(r), int(c)) for r, c in matches]
    return ParsedAnswer(raw=raw, parsed=cells)


def _path_cost(cells: list[Cell], cost_grid: np.ndarray) -> float:
    total = 0.0
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        mult = math.sqrt(2) if (r0 != r1 and c0 != c1) else 1.0
        total += cost_grid[r1, c1] * mult
    return total


def score_route_answer(
    task: TaskInstance,
    answer: ParsedAnswer,
    cost_grid: np.ndarray,
) -> ScoreBundle:
    if answer.parsed is None:
        return ScoreBundle(
            valid=False,
            failure=FailureCategory.FORMAT_VIOLATION,
            notes=answer.parse_error or "unparseable",
        )

    cells: list[Cell] = answer.parsed
    n = cost_grid.shape[0]
    start = tuple(task.params["start"])
    goal = tuple(task.params["goal"])

    if not cells:
        return ScoreBundle(valid=False, failure=FailureCategory.FORMAT_VIOLATION, notes="empty path")

    for i, (r, c) in enumerate(cells):
        if not (0 <= r < n and 0 <= c < n):
            return ScoreBundle(
                valid=False,
                failure=FailureCategory.HALLUCINATED_COORD,
                notes=f"cell {(r, c)} at index {i} is out of bounds",
            )
        if i > 0:
            pr, pc = cells[i - 1]
            if max(abs(r - pr), abs(c - pc)) != 1:
                return ScoreBundle(
                    valid=False,
                    failure=FailureCategory.INFEASIBLE_STEP,
                    notes=f"non-adjacent step {(pr, pc)}→{(r, c)}",
                )
        if not np.isfinite(cost_grid[r, c]):
            return ScoreBundle(
                valid=False,
                failure=FailureCategory.INFEASIBLE_STEP,
                notes=f"step onto impassable cell {(r, c)}",
            )

    if tuple(cells[0]) != start:
        return ScoreBundle(
            valid=False,
            failure=FailureCategory.IGNORED_CONSTRAINT,
            notes=f"path does not start at {start}",
        )
    if tuple(cells[-1]) != goal:
        return ScoreBundle(
            valid=False,
            failure=FailureCategory.IGNORED_CONSTRAINT,
            notes=f"path does not end at {goal}",
        )

    agent_cost = _path_cost(cells, cost_grid)
    optimal = task.ground_truth.optimal_value
    regret = (agent_cost - optimal) / optimal if optimal > 0 else 0.0

    return ScoreBundle(
        valid=True,
        agent_value=agent_cost,
        regret=regret,
        failure=FailureCategory.NONE,
        notes=f"length={len(cells)}",
    )


def score_from_solution(
    task: TaskInstance,
    solution: list[Any],
    cost_grid: np.ndarray,
) -> ScoreBundle:
    cells = [(int(r), int(c)) for r, c in solution]
    return score_route_answer(
        task,
        ParsedAnswer(raw=str(solution), parsed=cells),
        cost_grid,
    )
