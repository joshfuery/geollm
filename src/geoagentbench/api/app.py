from __future__ import annotations

import json
import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from ..regions import REGIONS, resolve_region
from ..runner.store import Store

load_dotenv()

DATA_DIR = Path(os.environ.get("GEOAB_DATA_DIR", "data"))
RESULTS_DIR = Path(os.environ.get("GEOAB_RESULTS_DIR", "results"))
LIVE_DIR = Path(os.environ.get("GEOAB_LIVE_DIR", "results-live"))

StoreName = Literal["main", "live"]

_DEFAULT_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173",
]
_env_origins = os.environ.get("GEOAB_ALLOW_ORIGINS", "").strip()
CORS_ORIGINS = _DEFAULT_ORIGINS + [o for o in _env_origins.split(",") if o.strip()]

app = FastAPI(
    title="GeoAgentBench API",
    description="Serves regions, terrain heightmaps, tasks, and agent runs.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _store(which: StoreName = "main") -> Store:
    return Store.open(LIVE_DIR if which == "live" else RESULTS_DIR)


def _region_has_terrain(name: str) -> bool:
    d = DATA_DIR / name
    return (d / "terrain.png").exists() and (d / "terrain.meta.json").exists()


@app.get("/api/health")
def health() -> dict[str, Any]:
    fetched = {
        name: {
            "dem": (DATA_DIR / name / "dem.tif").exists(),
            "water": (DATA_DIR / name / "water.osm.json").exists(),
            "terrain_png": _region_has_terrain(name),
        }
        for name in REGIONS
    }
    fetched["synthetic"] = {"dem": False, "water": False, "terrain_png": _region_has_terrain("synthetic")}
    store = _store()
    return {
        "ok": True,
        "data_dir": str(DATA_DIR.resolve()),
        "results_dir": str(RESULTS_DIR.resolve()),
        "regions": fetched,
        "counts": {
            "tasks": len(store.list_tasks(limit=10_000)),
            "configs": len(store.list_configs()),
            "runs": len(store.list_runs(limit=10_000)),
        },
    }


@app.get("/api/regions")
def list_regions() -> list[dict[str, Any]]:
    out = []
    for r in REGIONS.values():
        out.append({
            "name": r.name, "label": r.label, "description": r.description,
            "dem_type": r.dem_type, "bbox": r.bbox.model_dump(),
            "has_terrain": _region_has_terrain(r.name),
        })
    out.append({
        "name": "synthetic", "label": "Synthetic terrain",
        "description": "Deterministic diamond-square DEM.",
        "dem_type": "synthetic", "bbox": None,
        "has_terrain": _region_has_terrain("synthetic"),
    })
    return out


@app.get("/api/regions/{name}")
def get_region(name: str) -> dict[str, Any]:
    if name == "synthetic":
        spec: dict[str, Any] = {"name": "synthetic", "label": "Synthetic terrain",
                                "description": "Deterministic diamond-square DEM.", "bbox": None}
    elif name in REGIONS:
        r = REGIONS[name]
        spec = {"name": r.name, "label": r.label, "description": r.description,
                "dem_type": r.dem_type, "bbox": r.bbox.model_dump()}
    else:
        raise HTTPException(404, f"unknown region: {name}")
    meta_path = DATA_DIR / name / "terrain.meta.json"
    spec["terrain"] = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return spec


@app.get("/api/regions/{name}/terrain.png")
def get_terrain_png(name: str) -> FileResponse:
    if name not in REGIONS and name != "synthetic":
        raise HTTPException(404, f"unknown region: {name}")
    png_path = DATA_DIR / name / "terrain.png"
    if not png_path.exists():
        raise HTTPException(404, f"No terrain.png for {name!r}. Run scripts/encode_terrain.py.")
    return FileResponse(png_path, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/regions/{name}/meta")
def get_terrain_meta(name: str) -> dict[str, Any]:
    if name not in REGIONS and name != "synthetic":
        raise HTTPException(404, f"unknown region: {name}")
    meta_path = DATA_DIR / name / "terrain.meta.json"
    if not meta_path.exists():
        raise HTTPException(404, f"No terrain.meta.json for {name!r}")
    return JSONResponse(json.loads(meta_path.read_text()))


def _task_row_to_shape(row: dict[str, Any]) -> dict[str, Any]:
    params = json.loads(row["params_json"])
    gt = json.loads(row["ground_truth_json"])
    return {
        "task_id": row["task_id"],
        "family": row["family"],
        "seed": row["seed"],
        "params": params,
        "ground_truth": {
            "optimal_value": gt.get("optimal_value"),
            "optimal_solution": gt.get("optimal_solution"),
            "solver": gt.get("solver"),
            "compute_ms": gt.get("compute_ms"),
        },
        "region": json.loads(row["region_json"]),
        "start": params.get("start"),
        "goal": params.get("goal"),
        "grid_size": params.get("grid_size"),
        "source": params.get("source"),
        "region_name": params.get("region"),
    }


@app.get("/api/tasks")
def list_tasks(
    family: str | None = None,
    limit: int = Query(50, ge=1, le=1000),
    store: StoreName = "main",
) -> list[dict[str, Any]]:
    rows = _store(store).list_tasks(family=family, limit=limit)
    return [_task_row_to_shape(r) for r in rows]


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str, store: StoreName = "main") -> dict[str, Any]:
    row = _store(store).get_task(task_id)
    if not row:
        raise HTTPException(404, f"no task {task_id!r}")
    return _task_row_to_shape(row)


SYNTHETIC_CELL_M = 30.0
_M_PER_DEG = 111_320.0


def _regenerate_task(row: dict[str, Any]):
    from ..core import RouteParams
    from ..tasks.route import generate_route_from_dem, generate_synthetic_route_task

    from ..core import BBox

    params = json.loads(row["params_json"])
    if params.get("source") == "dem":
        w = params.get("window")
        return generate_route_from_dem(
            params["region"], row["seed"], data_dir=DATA_DIR,
            grid_size=params["grid_size"], connectivity=params.get("connectivity", 8),
            include_water=params.get("water_cells", 0) > 0,
            window=BBox(min_x=w[0], min_y=w[1], max_x=w[2], max_y=w[3]) if w else None,
        )
    rp = RouteParams(**{k: params[k] for k in RouteParams.model_fields if k in params})
    return generate_synthetic_route_task(row["seed"], rp)


def _dem_task_bbox(params: dict[str, Any]) -> list[float]:
    if params.get("window"):
        return list(params["window"])
    import rasterio

    with rasterio.open(DATA_DIR / params["region"] / "dem.tif") as src:
        b = src.bounds
    return [b.left, b.bottom, b.right, b.top]


def _display_frame(params: dict[str, Any], n: int, task_id: str, store: str) -> dict[str, Any]:
    if params.get("source") == "dem":
        region = resolve_region(params["region"], DATA_DIR)
        return {"kind": "dem", "region": params["region"], "region_label": region.label,
                "bbox": _dem_task_bbox(params), "pixel_size_m": params.get("pixel_size_m"),
                "terrain_url": f"/api/tasks/{task_id}/terrain.png?store={store}"}
    half_deg = 0.5 * n * SYNTHETIC_CELL_M / _M_PER_DEG
    return {"kind": "synthetic", "region": None, "region_label": "Synthetic terrain",
            "bbox": [-half_deg, -half_deg, half_deg, half_deg],
            "pixel_size_m": SYNTHETIC_CELL_M, "terrain_url": None}


@app.get("/api/tasks/{task_id}/grid")
def get_task_grid(task_id: str, store: StoreName = "main") -> dict[str, Any]:
    row = _store(store).get_task(task_id)
    if not row:
        raise HTTPException(404, f"no task {task_id!r}")
    try:
        task, cost, elev = _regenerate_task(row)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    if task.task_id != task_id:
        raise HTTPException(
            409, f"regenerated task id {task.task_id} != {task_id}; generator or data changed")

    import numpy as np

    n = int(cost.shape[0])
    finite = np.isfinite(cost)
    blocked = np.argwhere(~finite).tolist()
    return {
        "task_id": task_id,
        "grid_size": n,
        "frame": _display_frame(task.params, n, task_id, store),
        "elev_m": np.round(elev, 2).tolist(),
        "cost": [[(round(float(v), 4) if np.isfinite(v) else None) for v in r] for r in cost],
        "cost_range": [float(cost[finite].min()), float(cost[finite].max())],
        "blocked": blocked,
    }


@lru_cache(maxsize=32)
def _terrain_png(region: str, bbox: tuple[float, float, float, float], max_edge: int = 1024) -> bytes:
    import io

    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    from ..terrain_encode import encode_terrarium

    with rasterio.open(DATA_DIR / region / "dem.tif") as src:
        win = from_bounds(*bbox, transform=src.transform)
        w, h = max(2, round(win.width)), max(2, round(win.height))
        scale = min(1.0, max_edge / max(w, h))
        elev = src.read(1, window=win, out_shape=(max(2, round(h * scale)), max(2, round(w * scale))),
                        resampling=Resampling.bilinear, boundless=True, fill_value=0).astype(np.float64)
    buf = io.BytesIO()
    Image.fromarray(encode_terrarium(elev), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@app.get("/api/tasks/{task_id}/terrain.png")
def get_task_terrain(task_id: str, store: StoreName = "main") -> Response:
    row = _store(store).get_task(task_id)
    if not row:
        raise HTTPException(404, f"no task {task_id!r}")
    params = json.loads(row["params_json"])
    if params.get("source") != "dem":
        raise HTTPException(404, "synthetic tasks have no DEM; use /grid")
    png = _terrain_png(params["region"], tuple(_dem_task_bbox(params)))
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/tasks/{task_id}/runs")
def get_task_runs(task_id: str, limit: int = 200, store: StoreName = "main") -> list[dict[str, Any]]:
    rows = _store(store).list_runs(task_id=task_id, limit=limit)
    return [_run_row_to_shape(r) for r in rows]


def _run_row_to_shape(row: dict[str, Any]) -> dict[str, Any]:
    answer = json.loads(row["answer_json"])
    return {
        "run_id": row["run_id"],
        "task_id": row["task_id"],
        "config_id": row["config_id"],
        "valid": bool(row["valid"]),
        "regret": row["regret"],
        "agent_value": row["agent_value"],
        "failure": row["failure"],
        "notes": row["notes"],
        "tokens_in": row["tokens_in"],
        "tokens_out": row["tokens_out"],
        "wall_ms": row["wall_ms"],
        "error": row["error"],
        "predicted_path": answer.get("parsed"),
        "raw_answer": answer.get("raw"),
        "created_at": row["created_at"],
    }


@app.get("/api/runs")
def list_runs(
    config_id: str | None = None,
    task_id: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    rows = _store().list_runs(config_id=config_id, task_id=task_id, limit=limit)
    return [_run_row_to_shape(r) for r in rows]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, store: StoreName = "main") -> dict[str, Any]:
    row = _store(store).get_run(run_id)
    if not row:
        raise HTTPException(404, f"no run {run_id!r}")
    out = _run_row_to_shape(row)
    out["traces"] = _store(store).read_traces(run_id)
    return out


@app.get("/api/configs")
def list_configs(store: StoreName = "main") -> list[dict[str, Any]]:
    rows = _store(store).list_configs()
    return [{
        "config_id": r["config_id"],
        "model": r["model"],
        "temperature": r["temperature"],
        "prompt_version": r["prompt_version"],
        "condition": r["condition"],
        "topology": r["topology"],
        "tool_access": r["tool_access"],
        "payload": json.loads(r["payload_json"]),
    } for r in rows]


from .live import router as live_router  # noqa: E402

app.include_router(live_router)
