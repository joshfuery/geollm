from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core import (
    BBox,
    GroundTruth,
    RouteParams,
    TaskContext,
    TaskFamily,
    TaskInstance,
    make_task_id,
)
from ..regions import RegionSpec, resolve_region


def _slope_cost(elev_m: np.ndarray, pixel_size_m: float = 30.0) -> np.ndarray:
    gy, gx = np.gradient(elev_m.astype(np.float64), pixel_size_m)
    slope = np.hypot(gx, gy)
    return 1.0 + 8.0 * slope


def _sample_start_goal(
    cost: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 100,
) -> tuple[tuple[int, int], tuple[int, int]]:
    n = cost.shape[0]
    third = max(1, n // 3)
    for _ in range(max_tries):
        start = (int(rng.integers(0, third)), int(rng.integers(0, third)))
        goal = (int(rng.integers(n - third, n)), int(rng.integers(n - third, n)))
        if np.isfinite(cost[start]) and np.isfinite(cost[goal]):
            return start, goal
    raise RuntimeError("could not sample passable start/goal after many tries")


def _diamond_square(size: int, rng: np.random.Generator, roughness: float) -> np.ndarray:
    k = 1
    while (1 << k) + 1 < size:
        k += 1
    n = (1 << k) + 1
    g = np.zeros((n, n), dtype=np.float64)
    g[0, 0] = rng.random()
    g[0, -1] = rng.random()
    g[-1, 0] = rng.random()
    g[-1, -1] = rng.random()

    step = n - 1
    scale = 1.0
    while step > 1:
        half = step // 2
        for i in range(0, n - 1, step):
            for j in range(0, n - 1, step):
                avg = (g[i, j] + g[i + step, j] + g[i, j + step] + g[i + step, j + step]) / 4
                g[i + half, j + half] = avg + (rng.random() - 0.5) * scale
        for i in range(0, n, half):
            start = 0 if (i // half) % 2 else half
            for j in range(start, n, step):
                vals, count = 0.0, 0
                if i - half >= 0:
                    vals += g[i - half, j]; count += 1
                if i + half < n:
                    vals += g[i + half, j]; count += 1
                if j - half >= 0:
                    vals += g[i, j - half]; count += 1
                if j + half < n:
                    vals += g[i, j + half]; count += 1
                g[i, j] = vals / count + (rng.random() - 0.5) * scale
        step = half
        scale *= roughness

    g = g[:size, :size]
    lo, hi = g.min(), g.max()
    return (g - lo) / (hi - lo) if hi > lo else np.zeros_like(g)


def generate_synthetic_route_task(
    seed: int,
    params: RouteParams | None = None,
) -> tuple[TaskInstance, np.ndarray, np.ndarray]:
    p = params or RouteParams()
    rng = np.random.default_rng(seed)
    elev01 = _diamond_square(p.grid_size, rng, roughness=0.5 + 0.5 * p.ruggedness)
    elev_m = elev01 * 1000.0
    cost = _slope_cost(elev_m, pixel_size_m=30.0)

    n_blocked = int(p.impassable_frac * cost.size)
    if n_blocked > 0:
        idx = rng.choice(cost.size, size=n_blocked, replace=False)
        rr, cc = np.unravel_index(idx, cost.shape)
        cost[rr, cc] = np.inf

    start, goal = _sample_start_goal(cost, rng)
    return _package_task(
        seed=seed,
        params_extra={**p.model_dump(), "source": "synthetic"},
        region_bbox=BBox(min_x=0, min_y=0, max_x=p.grid_size, max_y=p.grid_size),
        elev_m=elev_m,
        cost=cost,
        start=start,
        goal=goal,
        connectivity=p.connectivity,
        narrative_intro=(
            f"You are planning a traverse across a {p.grid_size}x{p.grid_size} "
            "synthetic terrain grid. "
        ),
    ), cost, elev_m


def generate_route_from_dem(
    region: RegionSpec | str,
    seed: int,
    data_dir: Path | str = "data",
    grid_size: int = 64,
    connectivity: int = 8,
    include_water: bool = True,
    window: BBox | None = None,
) -> tuple[TaskInstance, np.ndarray, np.ndarray]:
    try:
        import rasterio
        from rasterio.features import rasterize
        from rasterio.enums import Resampling
        import shapely  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Real-DEM tasks need the 'geo' extra. Install with: uv sync --extra geo"
        ) from e

    if isinstance(region, str):
        region = resolve_region(region, data_dir)

    dem_path = Path(data_dir) / region.name / "dem.tif"
    water_path = Path(data_dir) / region.name / "water.osm.json"
    if not dem_path.exists():
        raise FileNotFoundError(
            f"No DEM at {dem_path}. Run: uv run python scripts/fetch_data.py --region {region.name}"
        )

    with rasterio.open(dem_path) as src:
        if window is None:
            elev = src.read(
                1,
                out_shape=(grid_size, grid_size),
                resampling=Resampling.average,
            ).astype(np.float64)
            transform = src.transform * src.transform.scale(
                (src.width / grid_size), (src.height / grid_size)
            )
            extent = region.bbox
        else:
            from rasterio.windows import from_bounds

            win = from_bounds(window.min_x, window.min_y, window.max_x, window.max_y,
                              transform=src.transform)
            elev = src.read(
                1,
                window=win,
                out_shape=(grid_size, grid_size),
                resampling=Resampling.average,
            ).astype(np.float64)
            win_tf = src.window_transform(win)
            transform = win_tf * win_tf.scale(win.width / grid_size, win.height / grid_size)
            extent = window
        lon_span = extent.max_x - extent.min_x
        lat_span = extent.max_y - extent.min_y
        lat_mid = 0.5 * (extent.min_y + extent.max_y)
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat_mid))
        pixel_x_m = lon_span * m_per_deg_lon / grid_size
        pixel_y_m = lat_span * m_per_deg_lat / grid_size
        pixel_size_m = float(0.5 * (pixel_x_m + pixel_y_m))

    cost = _slope_cost(elev, pixel_size_m=pixel_size_m)

    water_mask = np.zeros_like(elev, dtype=bool)
    if include_water and water_path.exists():
        try:
            osm = json.loads(water_path.read_text())
            geoms = _osm_water_to_geoms(osm)
            if geoms:
                mask = rasterize(
                    [(g, 1) for g in geoms],
                    out_shape=(grid_size, grid_size),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                water_mask = mask.astype(bool)
                cost[water_mask] = np.inf
        except Exception as e:
            print(f"  [warn] could not rasterize water for {region.name}: {e}")

    rng = np.random.default_rng(seed)
    start, goal = _sample_start_goal(cost, rng)

    params_extra = {
        "grid_size": grid_size,
        "connectivity": connectivity,
        "source": "dem",
        "region": region.name,
        "dem_type": region.dem_type,
        "pixel_size_m": pixel_size_m,
        "elev_min_m": float(elev.min()),
        "elev_max_m": float(elev.max()),
        "water_cells": int(water_mask.sum()),
    }
    if window is not None:
        params_extra["window"] = [round(v, 6) for v in window.as_tuple()]
    narrative_intro = (
        f"You are planning a traverse across {region.label}. "
        f"The terrain covers roughly {lon_span:.3f}° lon × {lat_span:.3f}° lat "
        f"(~{pixel_size_m:.0f} m per cell), gridded to {grid_size}x{grid_size}. "
        f"Elevation ranges from {elev.min():.0f} m to {elev.max():.0f} m. "
    )
    if include_water and water_mask.any():
        narrative_intro += (
            f"{int(water_mask.sum())} cells covering rivers and lakes are impassable. "
        )

    task = _package_task(
        seed=seed,
        params_extra=params_extra,
        region_bbox=extent,
        elev_m=elev,
        cost=cost,
        start=start,
        goal=goal,
        connectivity=connectivity,
        narrative_intro=narrative_intro,
    )
    return task, cost, elev


def _osm_water_to_geoms(osm: dict) -> list:
    from shapely.geometry import LineString, Polygon

    geoms = []
    for el in osm.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        if tags.get("natural") == "water" and coords[0] == coords[-1] and len(coords) >= 4:
            try:
                geoms.append(Polygon(coords))
            except Exception:
                pass
        else:
            geoms.append(LineString(coords).buffer(0.0002))
    return geoms


def _package_task(
    *,
    seed: int,
    params_extra: dict,
    region_bbox: BBox,
    elev_m: np.ndarray,
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    connectivity: int,
    narrative_intro: str,
) -> TaskInstance:
    params = {**params_extra, "start": list(start), "goal": list(goal)}
    task_id = make_task_id(TaskFamily.ROUTE, seed, params)
    n_blocked = int(np.isinf(cost).sum())

    narrative = (
        narrative_intro
        + "Traversal cost per step scales with local slope. "
        + f"There are {n_blocked} impassable cells (marked 'inf' in the cost grid). "
        + f"Your start is cell {start} (row, col) and your goal is cell {goal}. "
        + f"You may move to any of the {connectivity} neighboring cells at each step. "
        + "Return the sequence of cells that minimizes total traversal cost."
    )
    context = TaskContext(
        narrative=narrative,
        structured={
            "grid_size": cost.shape[0],
            "connectivity": connectivity,
            "start": list(start),
            "goal": list(goal),
            "cost_grid": [
                [("inf" if not np.isfinite(v) else float(v)) for v in row] for row in cost
            ],
        },
    )

    return TaskInstance(
        task_id=task_id,
        family=TaskFamily.ROUTE,
        seed=seed,
        region=region_bbox,
        params=params,
        context=context,
        ground_truth=GroundTruth(
            optimal_value=float("nan"),
            optimal_solution=None,
            solver="pending",
            compute_ms=0,
        ),
    )


def generate_route_task(seed: int, params: RouteParams | None = None):
    return generate_synthetic_route_task(seed=seed, params=params)
