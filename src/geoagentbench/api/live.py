from __future__ import annotations

import asyncio
import json
import math
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core import BBox, RouteParams
from ..regions import REGIONS, resolve_region
from ..runner.config import AgentConfig, Condition, RunConfig, ToolAccess, Topology
from ..runner.store import Store

router = APIRouter(prefix="/api/live", tags=["live"])

MOCK_MODES = ("oracle", "random", "straight", "composer")


def _data_dir() -> Path:
    from . import app as app_mod
    return app_mod.DATA_DIR


def live_store() -> Store:
    from . import app as app_mod
    return Store.open(app_mod.LIVE_DIR)


def available_dem_regions() -> list[str]:
    d = _data_dir()
    names = [n for n in REGIONS if (d / n / "dem.tif").exists()]
    places = d / "places"
    if places.exists():
        names += [f"places/{p.name}" for p in sorted(places.iterdir()) if (p / "dem.tif").exists()]
    return names


class NewTaskRequest(BaseModel):
    source: Literal["region", "anywhere", "synthetic"] = "region"
    region: str | None = None
    grid_size: int = Field(16, ge=6, le=48)
    seed: int | None = None


def _random_window(dem_path: Path, rng: random.Random) -> BBox:
    import rasterio

    with rasterio.open(dem_path) as src:
        b = src.bounds
    lat_mid = 0.5 * (b.bottom + b.top)
    m_lon = 111_320.0 * math.cos(math.radians(lat_mid))
    w_m, h_m = (b.right - b.left) * m_lon, (b.top - b.bottom) * 111_320.0
    side_m = rng.uniform(0.45, 0.8) * min(w_m, h_m)
    dlon, dlat = side_m / m_lon, side_m / 111_320.0
    x0 = rng.uniform(b.left, b.right - dlon)
    y0 = rng.uniform(b.bottom, b.top - dlat)
    return BBox(min_x=round(x0, 6), min_y=round(y0, 6),
                max_x=round(x0 + dlon, 6), max_y=round(y0 + dlat, 6))


def make_live_task(req: NewTaskRequest, rng: random.Random | None = None):
    from ..places import fetch_random_place
    from ..solvers.route_solver import solve_and_attach
    from ..tasks.route import generate_route_from_dem, generate_synthetic_route_task

    rng = rng or random.Random()
    data_dir = _data_dir()
    last_err: Exception | None = None

    for _attempt in range(6):
        seed = req.seed if req.seed is not None else rng.randrange(1_000_000)
        try:
            if req.source == "synthetic":
                params = RouteParams(grid_size=req.grid_size,
                                     ruggedness=round(rng.uniform(0.3, 0.8), 2))
                task, cost, elev = generate_synthetic_route_task(seed, params)
            elif req.source == "anywhere":
                spec = fetch_random_place(data_dir, rng=rng)
                task, cost, elev = generate_route_from_dem(
                    spec, seed, data_dir=data_dir, grid_size=req.grid_size, include_water=False)
            else:
                choices = [req.region] if req.region else available_dem_regions()
                if not choices:
                    raise HTTPException(409, "No DEMs cached. Run scripts/fetch_data.py first.")
                name = rng.choice(choices)
                resolve_region(name, data_dir)
                window = _random_window(data_dir / name / "dem.tif", rng)
                task, cost, elev = generate_route_from_dem(
                    name, seed, data_dir=data_dir, grid_size=req.grid_size,
                    include_water=(data_dir / name / "water.osm.json").exists(), window=window)
            if float(elev.max() - elev.min()) < 30 and req.source != "synthetic":
                last_err = RuntimeError("location too flat")
                continue
            return solve_and_attach(task, cost), cost, elev
        except (RuntimeError, ValueError) as e:
            last_err = e
            if req.seed is not None:
                break
    raise HTTPException(422, f"couldn't build a task here: {last_err}")


@router.post("/tasks")
def create_task(req: NewTaskRequest) -> dict[str, Any]:
    from . import app as app_mod

    try:
        task, _cost, _elev = make_live_task(req)
    except PermissionError as e:
        raise HTTPException(400, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except OSError as e:
        raise HTTPException(502, f"couldn't fetch terrain: {e}") from e
    store = live_store()
    store.upsert_task(task)
    return app_mod._task_row_to_shape(store.get_task(task.task_id))


class NewRunRequest(BaseModel):
    task_id: str
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    topology: Topology = Topology.SINGLE_SHOT
    tool_access: ToolAccess = ToolAccess.NONE
    condition: Condition = Condition.STRUCTURED_ONLY
    temperature: float = Field(0.7, ge=0, le=2)
    max_segment_cells: int | None = Field(None, ge=2, le=200)
    max_revise_rounds: int = Field(1, ge=1, le=5)
    max_tool_steps: int = Field(16, ge=1, le=60)


@dataclass
class LiveRun:
    live_id: str
    task_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    started: float = field(default_factory=time.perf_counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, ev: dict[str, Any]) -> None:
        with self.lock:
            ev = {"seq": len(self.events), "t_ms": int((time.perf_counter() - self.started) * 1000), **ev}
            self.events.append(ev)


LIVE_RUNS: dict[str, LiveRun] = {}
MAX_ACTIVE = 3


def _build_provider(req: NewRunRequest):
    from ..runner.mock_agents import MockMode, responder_for
    from ..runner.observe import ComposerResponder
    from ..runner.providers import MockProvider, OpenAIProvider, ProviderConfigError

    if req.provider == "openai":
        try:
            return OpenAIProvider()
        except ProviderConfigError as e:
            raise HTTPException(400, f"OpenAI isn't configured: {e}") from e
    if req.provider.startswith("mock:"):
        mode = req.provider.split(":", 1)[1]
        if mode == "composer":
            if req.topology != Topology.TOOL_LOOP:
                raise HTTPException(400, "mock:composer only drives the tool_loop topology")
            cap = req.max_segment_cells or 5
            return MockProvider(responder=ComposerResponder(max_leg=max(1, cap - 1)), latency_ms=450)
        if mode not in MOCK_MODES:
            raise HTTPException(400, f"unknown mock mode {mode!r}")
        return MockProvider(responder=responder_for(MockMode(mode)), latency_ms=900)
    raise HTTPException(400, f"unknown provider {req.provider!r}")


def _agent_config(req: NewRunRequest) -> AgentConfig:
    is_mock = req.provider.startswith("mock:")
    extras: dict[str, Any] = {}
    if req.topology == Topology.TOOL_LOOP and req.max_tool_steps != 16:
        extras["max_tool_steps"] = req.max_tool_steps
    return AgentConfig(
        model=f"mock-{req.provider.split(':', 1)[1]}" if is_mock else req.model,
        temperature=req.temperature,
        condition=req.condition,
        topology=req.topology,
        tool_access=req.tool_access,
        provider=req.provider,
        max_segment_cells=req.max_segment_cells if req.tool_access == ToolAccess.RESTRICTED_SOLVER else None,
        max_revise_rounds=req.max_revise_rounds,
        provider_kwargs=extras,
    )


def _run_thread(live: LiveRun, provider, cfg: AgentConfig) -> None:
    from . import app as app_mod
    from ..runner.executor import execute_one
    from ..runner.observe import ObservedProvider
    from ..solvers.route_solver import solve_and_attach

    try:
        store = live_store()
        row = store.get_task(live.task_id)
        task, cost, _elev = app_mod._regenerate_task(row)
        task = solve_and_attach(task, cost)
        store.upsert_config(cfg)
        live.emit({"type": "status", "message": f"running {cfg.topology.value} with {cfg.model}"})
        observed = ObservedProvider(provider, live.emit)
        outcome = execute_one(task, cost, observed,
                              RunConfig(agent=cfg, max_infra_retries=3, infra_retry_backoff_s=2.0))
        store.write_run(outcome.record, outcome.record.node_traces)
        live.emit({"type": "result", "run": app_mod._run_row_to_shape(store.get_run(outcome.record.run_id))})
    except Exception as e:
        live.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        live.emit({"type": "done"})
        live.done = True


@router.post("/runs")
def start_run(req: NewRunRequest) -> dict[str, Any]:
    if not live_store().get_task(req.task_id):
        raise HTTPException(404, f"no live task {req.task_id!r}")
    active = sum(1 for r in LIVE_RUNS.values() if not r.done)
    if active >= MAX_ACTIVE:
        raise HTTPException(429, f"{active} runs already in progress; wait for one to finish")
    provider = _build_provider(req)
    cfg = _agent_config(req)
    live = LiveRun(live_id="live_" + uuid.uuid4().hex[:10], task_id=req.task_id)
    LIVE_RUNS[live.live_id] = live
    threading.Thread(target=_run_thread, args=(live, provider, cfg), daemon=True).start()
    return {"live_id": live.live_id, "config_id": cfg.config_id}


@router.get("/runs/{live_id}/events")
async def run_events(live_id: str, request: Request) -> StreamingResponse:
    live = LIVE_RUNS.get(live_id)
    if live is None:
        raise HTTPException(404, f"no live run {live_id!r}")

    async def stream():
        i = 0
        while True:
            if await request.is_disconnected():
                return
            with live.lock:
                batch = live.events[i:]
            for ev in batch:
                yield f"data: {json.dumps(ev, default=str)}\n\n"
            i += len(batch)
            if live.done and i >= len(live.events):
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/options")
def options() -> dict[str, Any]:
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "opentopo": bool(os.environ.get("OPENTOPO_API_KEY")),
        "regions": [
            {"name": n, "label": resolve_region(n, _data_dir()).label}
            for n in available_dem_regions()
        ],
        "mock_modes": list(MOCK_MODES),
        "topologies": [t.value for t in Topology if t != Topology.PLAN_EXECUTE],
        "tool_access": [t.value for t in ToolAccess],
        "conditions": [c.value for c in Condition],
    }
