from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskFamily(str, Enum):
    ROUTE = "route"
    VIEWSHED = "viewshed"
    COVERAGE = "coverage"
    ALLOCATION = "allocation"


class BBox(BaseModel):
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)


class TaskContext(BaseModel):
    narrative: str
    structured: dict[str, Any]
    artifacts: dict[str, str] = Field(default_factory=dict)


class GroundTruth(BaseModel):
    optimal_value: float
    optimal_solution: Any
    solver: str
    compute_ms: int


class TaskInstance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    family: TaskFamily
    seed: int
    region: BBox
    params: dict[str, Any]
    context: TaskContext
    ground_truth: GroundTruth


def make_task_id(family: TaskFamily, seed: int, params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"family": family.value, "seed": seed, "params": params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ParsedAnswer(BaseModel):
    raw: str
    parsed: Any | None = None
    parse_error: str | None = None


class FailureCategory(str, Enum):
    HALLUCINATED_COORD = "hallucinated_coord"
    INFEASIBLE_STEP = "infeasible_step"
    IGNORED_CONSTRAINT = "ignored_constraint"
    ARITHMETIC_ERROR = "arithmetic_error"
    FORMAT_VIOLATION = "format_violation"
    TOOL_CALL_MALFORMED = "tool_call_malformed"
    NONE = "none"


class ScoreBundle(BaseModel):
    valid: bool
    regret: float | None = None
    agent_value: float | None = None
    failure: FailureCategory = FailureCategory.NONE
    notes: str = ""


class NodeTrace(BaseModel):
    node_name: str
    input: dict[str, Any]
    output: dict[str, Any]
    valid: bool | None = None
    wall_ms: int
    tokens_in: int = 0
    tokens_out: int = 0


class RunRecord(BaseModel):
    run_id: str
    task_id: str
    config_id: str
    answer: ParsedAnswer
    score: ScoreBundle
    node_traces: list[NodeTrace] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    wall_ms: int = 0
    error: str | None = None


Cell = tuple[int, int]


class RoutePath(BaseModel):
    cells: list[Cell]
    cost: float | None = None


class RouteParams(BaseModel):
    grid_size: int = 32
    ruggedness: float = 0.5
    impassable_frac: float = 0.05
    connectivity: Literal[4, 8] = 8
