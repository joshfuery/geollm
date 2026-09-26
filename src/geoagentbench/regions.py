from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .core import BBox


class RegionSpec(BaseModel):
    name: str
    label: str
    bbox: BBox
    dem_type: str = "COP30"
    description: str


REGIONS: dict[str, RegionSpec] = {
    "yosemite": RegionSpec(
        name="yosemite",
        label="Yosemite Valley, USA",
        bbox=BBox(min_x=-119.68, min_y=37.70, max_x=-119.52, max_y=37.78),
        description="Iconic granite walls, dramatic elevation, the Merced River threading the valley floor.",
    ),
    "grand_canyon": RegionSpec(
        name="grand_canyon",
        label="Grand Canyon, USA",
        bbox=BBox(min_x=-112.20, min_y=36.03, max_x=-112.05, max_y=36.14),
        description="Stepped mesas and deep incisions; tests multi-modal path finding around long ridges.",
    ),
    "matterhorn": RegionSpec(
        name="matterhorn",
        label="Matterhorn, Swiss Alps",
        bbox=BBox(min_x=7.60, min_y=45.94, max_x=7.72, max_y=46.02),
        description="Extreme relief, glaciers, near-vertical faces — a stress test for cost accumulation.",
    ),
}


def resolve_region(name: str, data_dir: Path | str = "data") -> RegionSpec:
    if name in REGIONS:
        return REGIONS[name]
    meta_path = Path(data_dir) / name / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "region" in meta:
            return RegionSpec(**meta["region"])
    raise KeyError(f"unknown region {name!r}; available: {sorted(REGIONS)}")


def get_region(name: str) -> RegionSpec:
    if name not in REGIONS:
        raise KeyError(f"unknown region {name!r}; available: {sorted(REGIONS)}")
    return REGIONS[name]
