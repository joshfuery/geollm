"""Download a region's DEM (OpenTopography) and water features (OpenStreetMap).

Requires OPENTOPO_API_KEY in .env (free at https://portal.opentopography.org/).

    uv run python scripts/fetch_data.py --region yosemite
    uv run python scripts/fetch_data.py --region all --no-water
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geoagentbench.regions import REGIONS, RegionSpec  # noqa: E402


OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def fetch_dem(region: RegionSpec, out_dir: Path, api_key: str) -> Path:
    b = region.bbox
    params = {
        "demtype": region.dem_type,
        "south": b.min_y,
        "north": b.max_y,
        "west": b.min_x,
        "east": b.max_x,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    print(f"  [DEM  ] {region.dem_type}  bbox=({b.min_x:.4f}, {b.min_y:.4f}, {b.max_x:.4f}, {b.max_y:.4f})")
    t0 = time.time()
    r = requests.get(OPENTOPO_URL, params=params, timeout=180)
    if r.status_code == 401:
        raise SystemExit(
            "OpenTopography rejected the API key (401). Check OPENTOPO_API_KEY in your .env."
        )
    if r.status_code == 400:
        raise SystemExit(f"OpenTopography rejected the request (400): {r.text[:300]}")
    r.raise_for_status()
    dem_path = out_dir / "dem.tif"
    dem_path.write_bytes(r.content)
    dt_s = time.time() - t0
    print(f"           wrote {dem_path.name}  {len(r.content):,} bytes  ({dt_s:.1f}s)")
    return dem_path


def fetch_water(region: RegionSpec, out_dir: Path) -> Path | None:
    b = region.bbox
    query = f"""[out:json][timeout:90];
(
  way["natural"="water"]({b.min_y},{b.min_x},{b.max_y},{b.max_x});
  way["waterway"~"river|stream|canal"]({b.min_y},{b.min_x},{b.max_y},{b.max_x});
  relation["natural"="water"]({b.min_y},{b.min_x},{b.max_y},{b.max_x});
);
out geom;"""
    for url in OVERPASS_MIRRORS:
        print(f"  [OSM  ] querying {url}")
        try:
            t0 = time.time()
            r = requests.post(url, data={"data": query}, timeout=180)
            if r.status_code == 429:
                print("           rate-limited, trying next mirror ...")
                continue
            r.raise_for_status()
            water_path = out_dir / "water.osm.json"
            water_path.write_text(r.text)
            n = len(json.loads(r.text).get("elements", []))
            dt_s = time.time() - t0
            print(f"           wrote {water_path.name}  {len(r.text):,} bytes  {n} elements  ({dt_s:.1f}s)")
            return water_path
        except requests.HTTPError as e:
            print(f"           {url} failed: {e}")
            continue
    print("  [WARN ] all Overpass mirrors failed; skipping OSM water")
    return None


def write_meta(region: RegionSpec, out_dir: Path, dem_path: Path, water_path: Path | None) -> None:
    meta = {
        "region": region.model_dump(),
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dem": {
            "path": dem_path.name,
            "bytes": dem_path.stat().st_size,
            "source": f"OpenTopography {region.dem_type}",
        },
    }
    if water_path is not None:
        meta["water"] = {
            "path": water_path.name,
            "bytes": water_path.stat().st_size,
            "source": "OpenStreetMap via Overpass API",
        }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def fetch_region(name: str, data_root: Path, api_key: str, include_water: bool) -> None:
    region = REGIONS[name]
    out_dir = data_root / region.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n== {region.label} ({region.name}) ==")
    print(f"  {region.description}")
    dem_path = fetch_dem(region, out_dir, api_key)
    water_path = fetch_water(region, out_dir) if include_water else None
    write_meta(region, out_dir, dem_path, water_path)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, choices=[*REGIONS.keys(), "all"], help="named region or 'all'")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--no-water", action="store_true", help="skip OSM water fetch")
    args = ap.parse_args()

    api_key = os.environ.get("OPENTOPO_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENTOPO_API_KEY not set. Register a free key at https://portal.opentopography.org/, "
            "then put it in your .env or export it."
        )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    names = list(REGIONS.keys()) if args.region == "all" else [args.region]
    for name in names:
        try:
            fetch_region(name, args.data_dir, api_key, include_water=not args.no_water)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    print("\nDone. Try:  uv run python scripts/demo_route.py --region " + names[0])


if __name__ == "__main__":
    main()
