from __future__ import annotations

import datetime as dt
import json
import math
import os
import random
from pathlib import Path

from .core import BBox
from .regions import RegionSpec

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

PLACES: list[tuple[str, float, float]] = [
    ("Mount Rainier, USA", 46.853, -121.760),
    ("Grand Teton, USA", 43.741, -110.802),
    ("Zion Canyon, USA", 37.270, -112.950),
    ("Mount Whitney, USA", 36.578, -118.292),
    ("Glacier National Park, USA", 48.696, -113.718),
    ("Maroon Bells, USA", 39.071, -106.989),
    ("Olympic Mountains, USA", 47.801, -123.711),
    ("Mount Hood, USA", 45.373, -121.696),
    ("Denali, USA", 63.069, -151.007),
    ("Banff, Canada", 51.178, -115.571),
    ("Torres del Paine, Chile", -50.942, -72.993),
    ("Fitz Roy, Argentina", -49.271, -73.043),
    ("Aconcagua, Argentina", -32.653, -70.011),
    ("Huascarán, Peru", -9.122, -77.604),
    ("Everest, Nepal", 27.988, 86.925),
    ("Annapurna, Nepal", 28.596, 83.820),
    ("K2, Pakistan", 35.881, 76.513),
    ("Mont Blanc, France", 45.833, 6.865),
    ("Dolomites, Italy", 46.550, 11.850),
    ("Eiger, Switzerland", 46.577, 8.005),
    ("Lofoten, Norway", 68.200, 13.900),
    ("Kilimanjaro, Tanzania", -3.068, 37.356),
    ("Toubkal, Morocco", 31.060, -7.915),
    ("Aoraki / Mount Cook, New Zealand", -43.595, 170.142),
    ("Fiordland, New Zealand", -44.670, 167.930),
    ("Mount Fuji, Japan", 35.361, 138.727),
    ("Khan Tengri, Kazakhstan", 42.211, 80.175),
    ("Elbrus, Russia", 43.350, 42.440),
    ("Aneto, Spain", 42.631, 0.657),
    ("Ben Nevis, Scotland", 56.797, -5.004),
    ("Landmannalaugar, Iceland", 63.990, -19.060),
    ("High Tatras, Slovakia", 49.179, 20.088),
    ("Drakensberg, South Africa", -28.950, 29.200),
    ("Kinabalu, Malaysia", 6.075, 116.558),
    ("Mauna Kea, USA", 19.821, -155.468),
]


def place_bbox(lat: float, lon: float, size_km: float = 9.0) -> BBox:
    half_lat = 0.5 * size_km / 111.32
    half_lon = half_lat / max(0.2, math.cos(math.radians(lat)))
    return BBox(
        min_x=round(lon - half_lon, 5), min_y=round(lat - half_lat, 5),
        max_x=round(lon + half_lon, 5), max_y=round(lat + half_lat, 5),
    )


def _slug(lat: float, lon: float) -> str:
    return f"{'n' if lat >= 0 else 's'}{abs(lat):.3f}_{'e' if lon >= 0 else 'w'}{abs(lon):.3f}".replace(".", "p")


def fetch_random_place(
    data_dir: Path | str = "data",
    rng: random.Random | None = None,
    api_key: str | None = None,
    jitter_deg: float = 0.05,
    timeout_s: float = 120.0,
) -> RegionSpec:
    import requests

    rng = rng or random.Random()
    api_key = api_key or os.environ.get("OPENTOPO_API_KEY")
    if not api_key:
        raise PermissionError(
            "OPENTOPO_API_KEY is not set. Add it to .env (free key at "
            "https://portal.opentopography.org/) to fetch random locations."
        )

    label, lat, lon = rng.choice(PLACES)
    lat += rng.uniform(-jitter_deg, jitter_deg)
    lon += rng.uniform(-jitter_deg, jitter_deg)
    bbox = place_bbox(lat, lon)
    name = f"places/{_slug(lat, lon)}"
    spec = RegionSpec(
        name=name, label=f"Near {label}", bbox=bbox, dem_type="COP30",
        description=f"Random location near {label} ({lat:.3f}, {lon:.3f}).",
    )

    out_dir = Path(data_dir) / name
    dem_path = out_dir / "dem.tif"
    if dem_path.exists():
        return spec

    resp = requests.get(OPENTOPO_URL, timeout=timeout_s, params={
        "demtype": spec.dem_type,
        "south": bbox.min_y, "north": bbox.max_y, "west": bbox.min_x, "east": bbox.max_x,
        "outputFormat": "GTiff", "API_Key": api_key,
    })
    if resp.status_code == 401:
        raise PermissionError("OpenTopography rejected OPENTOPO_API_KEY (401).")
    resp.raise_for_status()
    if not resp.content.startswith((b"II*\x00", b"MM\x00*")):
        raise RuntimeError(f"OpenTopography did not return a GeoTIFF: {resp.text[:200]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    dem_path.write_bytes(resp.content)
    (out_dir / "meta.json").write_text(json.dumps({
        "region": spec.model_dump(),
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dem": {"path": "dem.tif", "bytes": len(resp.content),
                "source": f"OpenTopography {spec.dem_type}"},
    }, indent=2))
    return spec
