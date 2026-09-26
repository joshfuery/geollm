from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Condition(str, Enum):
    NARRATIVE_ONLY = "narrative_only"
    STRUCTURED_ONLY = "structured_only"
    BOTH = "both"


class Topology(str, Enum):
    SINGLE_SHOT = "single_shot"
    PLAN_EXECUTE = "plan_execute"
    PLAN_CRITIQUE_REVISE = "plan_critique_revise"
    PLAN_VERIFY_REVISE = "plan_verify_revise"
    TOOL_LOOP = "tool_loop"


class ToolAccess(str, Enum):
    NONE = "none"
    CALCULATOR = "calculator"
    RESTRICTED_SOLVER = "restricted_solver"
    FULL_SOLVER = "full_solver"


class AgentConfig(BaseModel):
    model: str
    temperature: float = 0.7
    max_output_tokens: int = 4096
    prompt_version: str = "route/v1"
    condition: Condition = Condition.STRUCTURED_ONLY
    topology: Topology = Topology.SINGLE_SHOT
    tool_access: ToolAccess = ToolAccess.NONE
    provider: str = "openai"
    max_revise_rounds: int = 1
    max_input_tokens_per_task: int = 60_000
    message_window: int = 6
    max_segment_cells: int | None = None
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def config_id(self) -> str:
        return _hash_config(self)


class RunConfig(BaseModel):
    family: str = "route"
    grid_size: int = 16
    seeds: list[int] = Field(default_factory=lambda: list(range(5)))
    k: int = 3
    agent: AgentConfig
    max_infra_retries: int = 5
    infra_retry_backoff_s: float = 4.0
    dry_run: bool = False


def _hash_config(cfg: AgentConfig) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "cfg_" + hashlib.sha256(payload.encode()).hexdigest()[:14]
