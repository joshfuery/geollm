from __future__ import annotations

import json
import random
import re
from enum import Enum
from typing import Callable

import numpy as np

from .config import AgentConfig


class MockMode(str, Enum):
    ORACLE = "oracle"
    STRAIGHT = "straight"
    RANDOM = "random"
    SILENT = "silent"


_START = re.compile(r'"start":\s*\[(\d+),\s*(\d+)\]')
_GOAL = re.compile(r'"goal":\s*\[(\d+),\s*(\d+)\]')
_GRID = re.compile(r'"cost_grid":\s*(\[\[.+?\]\])')


def _extract_task(user: str) -> tuple[tuple[int, int], tuple[int, int], np.ndarray] | None:
    ms, mg, mgrid = _START.search(user), _GOAL.search(user), _GRID.search(user)
    if not (ms and mg and mgrid):
        return None
    start = (int(ms.group(1)), int(ms.group(2)))
    goal = (int(mg.group(1)), int(mg.group(2)))
    raw = json.loads(mgrid.group(1).replace('"inf"', "1e18"))
    grid = np.array(raw, dtype=float)
    grid[grid >= 1e18] = np.inf
    return start, goal, grid


def _format_path(cells) -> str:
    return "FINAL PATH: " + " -> ".join(f"({r}, {c})" for r, c in cells)


def oracle_responder(system: str, user: str, cfg: AgentConfig) -> str:
    from ..solvers.route_solver import dijkstra_route
    got = _extract_task(user)
    if got is None:
        return "FINAL PATH:"
    start, goal, grid = got
    path = dijkstra_route(grid, start, goal, connectivity=8)
    return "Reasoning...\n\n" + _format_path(path.cells)


def straight_line_responder(system: str, user: str, cfg: AgentConfig) -> str:
    got = _extract_task(user)
    if got is None:
        return "FINAL PATH:"
    start, goal, _ = got
    return _format_path([start, goal])


def random_walk_responder(seed: int = 0) -> Callable[[str, str, AgentConfig], str]:
    call_counter = {"n": 0}

    def responder(system: str, user: str, cfg: AgentConfig) -> str:
        got = _extract_task(user)
        if got is None:
            return "FINAL PATH:"
        start, goal, grid = got
        n = grid.shape[0]
        call_counter["n"] += 1
        rng = random.Random(seed + call_counter["n"] * 7919 + start[0] * 1000 + start[1])
        max_steps = 4 * n
        r, c = start
        cells = [(r, c)]
        for _ in range(max_steps):
            if (r, c) == goal:
                break
            if rng.random() < 0.6:
                dr = int(np.sign(goal[0] - r))
                dc = int(np.sign(goal[1] - c))
            else:
                dr = rng.choice([-1, 0, 1])
                dc = rng.choice([-1, 0, 1])
            if dr == 0 and dc == 0:
                dc = 1
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            r, c = nr, nc
            cells.append((r, c))
        return _format_path(cells)
    return responder


def silent_responder(system: str, user: str, cfg: AgentConfig) -> str:
    return "FINAL PATH:"


RESPONDERS: dict[MockMode, Callable[[str, str, AgentConfig], str]] = {
    MockMode.ORACLE: oracle_responder,
    MockMode.STRAIGHT: straight_line_responder,
    MockMode.RANDOM: random_walk_responder(seed=42),
    MockMode.SILENT: silent_responder,
}


def responder_for(mode: MockMode) -> Callable[[str, str, AgentConfig], str]:
    return RESPONDERS[mode]
