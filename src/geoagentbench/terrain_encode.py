from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TERRARIUM_OFFSET = 32768.0


def encode_terrarium(elev_m: np.ndarray) -> np.ndarray:
    if elev_m.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {elev_m.shape}")

    z = np.asarray(elev_m, dtype=np.float64)
    z = np.where(np.isfinite(z), z, 0.0)
    z = np.clip(z, -TERRARIUM_OFFSET, TERRARIUM_OFFSET - 1e-3)

    shifted = z + TERRARIUM_OFFSET
    fixed = np.round(shifted * 256.0).astype(np.int64)

    r = (fixed >> 16) & 0xFF
    g = (fixed >> 8) & 0xFF
    b = fixed & 0xFF

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return rgb


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return (r * 256.0 + g + b / 256.0) - TERRARIUM_OFFSET


def write_terrain_png(elev_m: np.ndarray, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = encode_terrarium(elev_m)
    Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG", optimize=True)
    return out_path


def write_terrain_meta(
    out_path: Path | str,
    *,
    bbox: tuple[float, float, float, float],
    elev_min_m: float,
    elev_max_m: float,
    pixel_size_m: float,
    shape: tuple[int, int],
    source: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(out_path)
    meta: dict[str, Any] = {
        "encoding": "terrarium",
        "elev_offset_m": TERRARIUM_OFFSET,
        "shape": {"height": shape[0], "width": shape[1]},
        "bbox": {
            "min_x": bbox[0], "min_y": bbox[1],
            "max_x": bbox[2], "max_y": bbox[3],
            "crs": "EPSG:4326",
        },
        "elev_m": {"min": float(elev_min_m), "max": float(elev_max_m)},
        "pixel_size_m": float(pixel_size_m),
        "source": source,
    }
    if extra:
        meta.update(extra)
    out_path.write_text(json.dumps(meta, indent=2))
    return out_path


def encode_dem_file(
    dem_path: Path | str,
    out_png: Path | str,
    out_meta: Path | str,
    *,
    max_edge: int | None = 1024,
) -> dict[str, Any]:
    import rasterio
    from rasterio.enums import Resampling

    dem_path = Path(dem_path)
    with rasterio.open(dem_path) as src:
        h, w = src.height, src.width
        if max_edge is not None and max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            h_out, w_out = int(h * scale), int(w * scale)
        else:
            h_out, w_out = h, w
        elev = src.read(
            1,
            out_shape=(h_out, w_out),
            resampling=Resampling.average,
        ).astype(np.float64)
        bounds = src.bounds

    lat_mid = 0.5 * (bounds.bottom + bounds.top)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat_mid))
    pixel_x_m = (bounds.right - bounds.left) * m_per_deg_lon / w_out
    pixel_y_m = (bounds.top - bounds.bottom) * m_per_deg_lat / h_out
    pixel_size_m = float(0.5 * (pixel_x_m + pixel_y_m))

    write_terrain_png(elev, out_png)
    meta = {
        "bbox": (bounds.left, bounds.bottom, bounds.right, bounds.top),
        "elev_min_m": float(np.nanmin(elev)),
        "elev_max_m": float(np.nanmax(elev)),
        "pixel_size_m": pixel_size_m,
        "shape": (h_out, w_out),
        "source": f"GeoTIFF {dem_path.name}",
        "extra": {"native_shape": {"height": h, "width": w}},
    }
    write_terrain_meta(out_meta, **meta)
    return meta


def encode_synthetic_dem(
    seed: int,
    grid_size: int,
    out_png: Path | str,
    out_meta: Path | str,
) -> dict[str, Any]:
    from .core import RouteParams
    from .tasks.route import generate_synthetic_route_task

    task, _cost, elev = generate_synthetic_route_task(
        seed=seed, params=RouteParams(grid_size=grid_size)
    )
    write_terrain_png(elev, out_png)
    meta = {
        "bbox": (0.0, 0.0, float(grid_size), float(grid_size)),
        "elev_min_m": float(elev.min()),
        "elev_max_m": float(elev.max()),
        "pixel_size_m": 30.0,
        "shape": elev.shape,
        "source": f"synthetic seed={seed}",
        "extra": {"task_id": task.task_id, "synthetic": True},
    }
    write_terrain_meta(out_meta, **meta)
    return meta
