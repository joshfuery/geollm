from __future__ import annotations

import math

import numpy as np
import pytest

from geoagentbench.core import FailureCategory, ParsedAnswer, RouteParams
from geoagentbench.scorers.route_scorer import (
    parse_route_answer,
    score_route_answer,
)
from geoagentbench.solvers.route_solver import dijkstra_route, solve_and_attach
from geoagentbench.tasks.route import generate_route_task


def test_uniform_grid_straight_line():
    cost = np.ones((5, 5))
    path = dijkstra_route(cost, (0, 0), (4, 4), connectivity=8)
    assert path.cells[0] == (0, 0)
    assert path.cells[-1] == (4, 4)
    assert path.cost == pytest.approx(4 * math.sqrt(2))


def test_uniform_grid_manhattan_when_4_connected():
    cost = np.ones((5, 5))
    path = dijkstra_route(cost, (0, 0), (4, 4), connectivity=4)
    assert path.cost == pytest.approx(8.0)


def test_wall_forces_detour():
    cost = np.ones((5, 5))
    cost[:, 2] = np.inf
    cost[4, 2] = 1.0
    path = dijkstra_route(cost, (0, 0), (0, 4), connectivity=8)
    for r, c in path.cells:
        assert np.isfinite(cost[r, c])
    assert path.cells[-1] == (0, 4)


def test_no_path_raises():
    cost = np.ones((3, 3))
    cost[1, :] = np.inf
    with pytest.raises(ValueError, match="no path"):
        dijkstra_route(cost, (0, 0), (2, 2), connectivity=4)


def test_task_generation_is_deterministic():
    params = RouteParams(grid_size=16, ruggedness=0.7, impassable_frac=0.05)
    t1, c1, e1 = generate_route_task(seed=42, params=params)
    t2, c2, e2 = generate_route_task(seed=42, params=params)
    assert t1.task_id == t2.task_id
    assert np.array_equal(np.nan_to_num(c1, nan=-1, posinf=-2), np.nan_to_num(c2, nan=-1, posinf=-2))
    assert np.array_equal(e1, e2)


def test_different_seeds_give_different_tasks():
    params = RouteParams(grid_size=16)
    t1, _, _ = generate_route_task(seed=1, params=params)
    t2, _, _ = generate_route_task(seed=2, params=params)
    assert t1.task_id != t2.task_id


def test_optimal_solution_scores_zero_regret():
    params = RouteParams(grid_size=20, impassable_frac=0.03)
    task, cost, _ = generate_route_task(seed=7, params=params)
    task = solve_and_attach(task, cost)

    optimal_cells = task.ground_truth.optimal_solution
    parsed = ParsedAnswer(raw="", parsed=[(int(r), int(c)) for r, c in optimal_cells])
    score = score_route_answer(task, parsed, cost)

    assert score.valid
    assert score.regret == pytest.approx(0.0, abs=1e-9)
    assert score.failure == FailureCategory.NONE


def test_scorer_flags_non_adjacent_step():
    params = RouteParams(grid_size=10, impassable_frac=0.0)
    task, cost, _ = generate_route_task(seed=3, params=params)
    task = solve_and_attach(task, cost)

    parsed = ParsedAnswer(
        raw="",
        parsed=[tuple(task.params["start"]), tuple(task.params["goal"])],
    )
    score = score_route_answer(task, parsed, cost)
    assert not score.valid
    assert score.failure == FailureCategory.INFEASIBLE_STEP


def test_scorer_flags_out_of_bounds():
    params = RouteParams(grid_size=10, impassable_frac=0.0)
    task, cost, _ = generate_route_task(seed=3, params=params)
    task = solve_and_attach(task, cost)

    parsed = ParsedAnswer(
        raw="",
        parsed=[tuple(task.params["start"]), (999, 999)],
    )
    score = score_route_answer(task, parsed, cost)
    assert not score.valid
    assert score.failure == FailureCategory.HALLUCINATED_COORD


def test_scorer_parses_multiple_tuple_shapes():
    raw = "The path is: (0, 0) -> [1,1] -> (2, 2)"
    parsed = parse_route_answer(raw)
    assert parsed.parsed == [(0, 0), (1, 1), (2, 2)]
