"""Encode data/<region>/dem.tif as a terrarium PNG for the web viewer.

    uv run python scripts/encode_terrain.py --region all
    uv run python scripts/encode_terrain.py --synthetic --grid 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geoagentbench.regions import REGIONS
from geoagentbench.terrain_encode import encode_dem_file, encode_synthetic_dem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", choices=[*REGIONS.keys(), "all"], default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--max-edge", type=int, default=1024, help="max PNG dimension; larger = smoother 3D")
    ap.add_argument("--synthetic", action="store_true", help="encode a synthetic DEM (no data/ needed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=int, default=256)
    args = ap.parse_args()

    if args.synthetic:
        out_dir = args.data_dir / "synthetic"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = encode_synthetic_dem(
            seed=args.seed,
            grid_size=args.grid,
            out_png=out_dir / "terrain.png",
            out_meta=out_dir / "terrain.meta.json",
        )
        print(f"synthetic (seed={args.seed}, {args.grid}x{args.grid})")
        print(f"  elev: {meta['elev_min_m']:.1f} → {meta['elev_max_m']:.1f} m")
        print(f"  wrote {out_dir}/terrain.png  and  terrain.meta.json")
        return

    if args.region is None:
        raise SystemExit("--region or --synthetic required")

    names = list(REGIONS.keys()) if args.region == "all" else [args.region]
    for name in names:
        region_dir = args.data_dir / name
        dem_path = region_dir / "dem.tif"
        if not dem_path.exists():
            print(f"[skip] {name}: no {dem_path} — run scripts/fetch_data.py first")
            continue
        meta = encode_dem_file(
            dem_path=dem_path,
            out_png=region_dir / "terrain.png",
            out_meta=region_dir / "terrain.meta.json",
            max_edge=args.max_edge,
        )
        shape = meta["shape"]
        print(f"{name}: {shape[1]}x{shape[0]}  elev {meta['elev_min_m']:.0f}→{meta['elev_max_m']:.0f} m")
        print(f"  wrote {region_dir}/terrain.png  and  terrain.meta.json")


if __name__ == "__main__":
    main()
