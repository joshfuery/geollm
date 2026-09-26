from __future__ import annotations

import heapq
import math
import time
from typing import Literal

import numpy as np

from ..core import Cell, GroundTruth, RoutePath, TaskInstance

SOLVER_ID = "dijkstra-heapq-v1"


def _neighbors(r: int, c: int, n: int, connectivity: int):
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            yield nr, nc, (math.sqrt(2) if dr and dc else 1.0)


def dijkstra_route(
    cost_grid: np.ndarray,
    start: Cell,
    goal: Cell,
    connectivity: Literal[4, 8] = 8,
) -> RoutePath:
    n = cost_grid.shape[0]
    if cost_grid.shape != (n, n):
        raise ValueError(f"cost_grid must be square, got {cost_grid.shape}")

    dist = np.full((n, n), np.inf)
    dist[start] = 0.0
    prev: dict[Cell, Cell] = {}

    pq: list[tuple[float, Cell]] = [(0.0, start)]
    visited = np.zeros((n, n), dtype=bool)

    while pq:
        d, (r, c) = heapq.heappop(pq)
        if visited[r, c]:
            continue
        visited[r, c] = True
        if (r, c) == goal:
            break

        for nr, nc, mult in _neighbors(r, c, n, connectivity):
            step = cost_grid[nr, nc] * mult
            if not np.isfinite(step):
                continue
            nd = d + step
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (nd, (nr, nc)))

    if not np.isfinite(dist[goal]):
        raise ValueError(f"no path from {start} to {goal}")

    path: list[Cell] = [goal]
    cur = goal
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    path.reverse()

    return RoutePath(cells=path, cost=float(dist[goal]))


def solve_and_attach(task: TaskInstance, cost_grid: np.ndarray) -> TaskInstance:
    start = tuple(task.params["start"])
    goal = tuple(task.params["goal"])
    connectivity = task.params.get("connectivity", 8)

    t0 = time.perf_counter()
    path = dijkstra_route(cost_grid, start, goal, connectivity=connectivity)
    dt_ms = int((time.perf_counter() - t0) * 1000)

    return task.model_copy(update={
        "ground_truth": GroundTruth(
            optimal_value=path.cost,
            optimal_solution=[list(c) for c in path.cells],
            solver=SOLVER_ID,
            compute_ms=dt_ms,
        )
    })
