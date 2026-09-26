from __future__ import annotations

import json

from ..core import TaskInstance
from .config import Condition


SYSTEM_ROUTE_V1 = """You are a spatial-reasoning planner solving a shortest-path problem on a terrain grid.

Rules:
- You will be given a start cell (row, col) and a goal cell (row, col).
- You may step to any of the neighboring cells (8-connected by default): up, down, left, right, and the four diagonals.
- Diagonal steps cost sqrt(2) times the destination cell's traversal cost; straight steps cost 1x.
- Some cells are impassable (marked "inf" in the cost grid). Never step on those.
- Return the FULL sequence of cells from start to goal, minimizing total cost.

Output format (STRICT — the grader parses tuples of the form (row, col)):
FINAL PATH: (r0, c0) -> (r1, c1) -> ... -> (rN, cN)

The first cell must be the start; the last must be the goal. Do not add any cells after the FINAL PATH line."""


def render_prompt(task: TaskInstance, condition: Condition, prompt_version: str) -> tuple[str, str]:
    if prompt_version != "route/v1":
        raise ValueError(f"unknown prompt version {prompt_version!r}")

    ctx = task.context
    start, goal = task.params["start"], task.params["goal"]

    if condition == Condition.NARRATIVE_ONLY:
        user = ctx.narrative
    elif condition == Condition.STRUCTURED_ONLY:
        user = _structured_block(task)
    elif condition == Condition.BOTH:
        user = ctx.narrative + "\n\n" + _structured_block(task)
    else:
        raise ValueError(f"unsupported condition {condition!r}")

    user += (
        f"\n\nStart: ({start[0]}, {start[1]})"
        f"\nGoal:  ({goal[0]}, {goal[1]})"
        "\n\nThink step by step, then output the FINAL PATH line."
    )
    return SYSTEM_ROUTE_V1, user


def _structured_block(task: TaskInstance) -> str:
    s = task.context.structured
    payload = {
        "grid_size": s["grid_size"],
        "connectivity": s["connectivity"],
        "start": s["start"],
        "goal": s["goal"],
        "cost_grid": s["cost_grid"],
    }
    return "TASK (structured):\n```json\n" + json.dumps(payload, separators=(",", ":")) + "\n```"


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


CRITIQUE_SYSTEM = """You are a strict path validator.

You will be given a route-planning task and a proposed path from another agent.
Your job is to find every problem with the proposed path. Check EACH cell in
order.

For each cell in the path, verify:
  1. The cell is inside the grid (0 <= row < grid_size, 0 <= col < grid_size).
  2. The step from the previous cell is 8-adjacent — |dr| <= 1 AND |dc| <= 1
     AND at least one is nonzero.
  3. The cell is NOT impassable — its value in the cost grid is not "inf".

Also verify:
  4. The first cell equals the given start.
  5. The last cell equals the given goal.

Output format (STRICT — one bullet per issue you find; empty if none):
ISSUES:
- <issue 1>
- <issue 2>
...

Do not propose a fix. Just list issues."""


REVISE_SYSTEM = """You are a path revision agent.

You will be given a route-planning task, a proposed path, and a list of issues
a validator found with that path. Produce a corrected path.

If the issues list is empty, return the original path unchanged.

If there are issues, output a new path that:
  - starts at the given start cell,
  - ends at the given goal cell,
  - steps only to 8-adjacent cells,
  - never enters an impassable ("inf") cell,
  - minimizes total traversal cost.

Output format (STRICT — the grader parses tuples of the form (row, col)):
FINAL PATH: (r0, c0) -> (r1, c1) -> ... -> (rN, cN)

Do not add anything after the FINAL PATH line."""


def render_critique_prompt(
    task, condition, plan_text: str, prompt_version: str
) -> tuple[str, str]:
    if prompt_version != "route/v1":
        raise ValueError(f"unknown prompt version {prompt_version!r}")
    s = task.context.structured
    start, goal = s["start"], s["goal"]
    body = (
        f"Task: {task.context.narrative if condition.value == 'narrative_only' else _structured_block(task)}\n\n"
        f"Start: ({start[0]}, {start[1]})\n"
        f"Goal:  ({goal[0]}, {goal[1]})\n\n"
        f"Proposed path (full text from the planning agent):\n---\n{plan_text}\n---\n\n"
        "List every issue you find, or output 'ISSUES:' with nothing after it if the path is valid."
    )
    return CRITIQUE_SYSTEM, body


def render_revise_prompt(
    task, condition, plan_text: str, critique_text: str, prompt_version: str
) -> tuple[str, str]:
    if prompt_version != "route/v1":
        raise ValueError(f"unknown prompt version {prompt_version!r}")
    s = task.context.structured
    start, goal = s["start"], s["goal"]
    body = (
        f"Task:\n{_structured_block(task)}\n\n"
        f"Start: ({start[0]}, {start[1]})\n"
        f"Goal:  ({goal[0]}, {goal[1]})\n\n"
        f"Original proposed path:\n---\n{plan_text}\n---\n\n"
        f"Validator's critique:\n---\n{critique_text}\n---\n\n"
        "Produce the corrected FINAL PATH line."
    )
    return REVISE_SYSTEM, body
