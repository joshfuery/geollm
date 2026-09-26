from __future__ import annotations

import numpy as np

from ..core import Cell, TaskInstance


def check_path_issues(
    task: TaskInstance,
    cells: list[Cell] | None,
    cost_grid: np.ndarray,
) -> list[str]:
    issues: list[str] = []

    if cells is None:
        return ["path could not be parsed from FINAL PATH line"]
    if not cells:
        return ["path is empty"]

    n = cost_grid.shape[0]
    start = tuple(task.params["start"])
    goal = tuple(task.params["goal"])

    if tuple(cells[0]) != start:
        issues.append(f"first cell {tuple(cells[0])} is not the start {start}")
    if tuple(cells[-1]) != goal:
        issues.append(f"last cell {tuple(cells[-1])} is not the goal {goal}")

    for i, (r, c) in enumerate(cells):
        if not (0 <= r < n and 0 <= c < n):
            issues.append(f"cell {(r, c)} at step {i} is out of bounds (grid is {n}x{n})")
            continue
        if not np.isfinite(cost_grid[r, c]):
            issues.append(f"cell {(r, c)} at step {i} is impassable (cost = inf)")
        if i > 0:
            pr, pc = cells[i - 1]
            dr, dc = abs(r - pr), abs(c - pc)
            if dr > 1 or dc > 1:
                issues.append(f"step {i}: {(pr, pc)} -> {(r, c)} is non-adjacent (|dr|={dr}, |dc|={dc})")
            elif dr == 0 and dc == 0:
                issues.append(f"step {i}: {(pr, pc)} -> {(r, c)} does not move")

    return issues


def format_issues(issues: list[str]) -> str:
    if not issues:
        return "ISSUES:"
    return "ISSUES:\n" + "\n".join(f"- {i}" for i in issues)


def solve_between(
    task: TaskInstance,
    from_cell: tuple[int, int],
    to_cell: tuple[int, int],
    cost_grid: np.ndarray,
    max_segment_cells: int | None = None,
) -> dict:
    from ..solvers.route_solver import dijkstra_route

    try:
        path = dijkstra_route(
            cost_grid,
            (int(from_cell[0]), int(from_cell[1])),
            (int(to_cell[0]), int(to_cell[1])),
            connectivity=int(task.params.get("connectivity", 8)),
        )
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    if max_segment_cells is not None and len(path.cells) > max_segment_cells:
        return {
            "ok": False,
            "error": (
                f"segment too long ({len(path.cells)} cells > {max_segment_cells} cell limit). "
                f"Pick a waypoint between {tuple(from_cell)} and {tuple(to_cell)} and solve "
                "each half separately."
            ),
            "cells_would_have_been": len(path.cells),
        }
    return {"ok": True, "cells": [list(c) for c in path.cells], "cost": path.cost}


CHECK_PATH_TOOL = {
    "type": "function",
    "function": {
        "name": "check_path",
        "description": (
            "Validate a proposed path against the terrain grid. Returns a list "
            "of issues (impassable cells, non-adjacent steps, wrong endpoints). "
            "Empty list means the path is feasible. Call this BEFORE emitting "
            "FINAL PATH to make sure you got it right."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cells": {
                    "type": "array",
                    "description": "Ordered list of [row, col] cells from start to goal.",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                }
            },
            "required": ["cells"],
        },
    },
}

SOLVE_BETWEEN_TOOL = {
    "type": "function",
    "function": {
        "name": "solve_between",
        "description": (
            "Compute the optimal (minimum-cost) sub-path between any two cells "
            "using Dijkstra. Use this to route around obstacles: e.g. after "
            "you commit to visiting some waypoint, call solve_between to fill "
            "in the segment. Returns {ok, cells, cost} on success or {ok:false, "
            "error} if the cells cannot be connected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_row": {"type": "integer"},
                "from_col": {"type": "integer"},
                "to_row": {"type": "integer"},
                "to_col": {"type": "integer"},
            },
            "required": ["from_row", "from_col", "to_row", "to_col"],
        },
    },
}


def all_tool_specs() -> list[dict]:
    return [CHECK_PATH_TOOL, SOLVE_BETWEEN_TOOL]


def _extract_point(args: dict, prefix: str) -> tuple[int, int] | None:
    syn = {"from": "from", "start": "from", "to": "to", "end": "to", "goal": "to"}
    target = syn.get(prefix, prefix)
    if f"{target}_row" in args and f"{target}_col" in args:
        return int(args[f"{target}_row"]), int(args[f"{target}_col"])
    for alt in [k for k, v in syn.items() if v == target]:
        if f"{alt}_row" in args and f"{alt}_col" in args:
            return int(args[f"{alt}_row"]), int(args[f"{alt}_col"])
        v = args.get(alt)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return int(v[0]), int(v[1])
        if isinstance(v, dict):
            if "row" in v and "col" in v:
                return int(v["row"]), int(v["col"])
            if "r" in v and "c" in v:
                return int(v["r"]), int(v["c"])
    short = f"{target[0]}"
    if f"{short}_r" in args and f"{short}_c" in args:
        return int(args[f"{short}_r"]), int(args[f"{short}_c"])
    return None


def execute_tool_call(
    name: str,
    arguments: dict,
    task: TaskInstance,
    cost_grid: np.ndarray,
    max_segment_cells: int | None = None,
) -> dict:
    try:
        if name == "check_path":
            raw_cells = arguments.get("cells") or arguments.get("path") or []
            cells = []
            for c in raw_cells:
                if isinstance(c, (list, tuple)) and len(c) == 2:
                    cells.append((int(c[0]), int(c[1])))
                elif isinstance(c, dict) and "row" in c and "col" in c:
                    cells.append((int(c["row"]), int(c["col"])))
            issues = check_path_issues(task, cells if cells else None, cost_grid)
            return {"issues": issues, "valid": len(issues) == 0}

        if name == "solve_between":
            src_pt = _extract_point(arguments, "from")
            dst_pt = _extract_point(arguments, "to")
            if src_pt is None or dst_pt is None:
                return {
                    "ok": False,
                    "error": (
                        f"could not parse from/to cells from arguments={arguments!r}. "
                        "Expected shape: {\"from_row\": r, \"from_col\": c, "
                        "\"to_row\": r, \"to_col\": c}."
                    ),
                }
            return solve_between(task, src_pt, dst_pt, cost_grid, max_segment_cells=max_segment_cells)

        return {"error": f"unknown tool: {name}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def tool_specs_for_access(level: str) -> list[dict]:
    if level == "none":
        return []
    if level == "calculator":
        return [CHECK_PATH_TOOL]
    if level in ("restricted_solver", "full_solver"):
        return [CHECK_PATH_TOOL, SOLVE_BETWEEN_TOOL]
    return all_tool_specs()


def max_segment_for_access(level: str, default_restricted: int = 5) -> int | None:
    if level == "restricted_solver":
        return default_restricted
    return None
