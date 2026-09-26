"""Generate, solve and render one route task.

    uv run python scripts/demo_route.py
    uv run python scripts/demo_route.py --region yosemite
    uv run python scripts/demo_route.py --region matterhorn --grid 96
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from geoagentbench.core import ParsedAnswer, RouteParams
from geoagentbench.regions import REGIONS
from geoagentbench.scorers.route_scorer import score_route_answer
from geoagentbench.solvers.route_solver import solve_and_attach
from geoagentbench.tasks.route import (
    generate_route_from_dem,
    generate_synthetic_route_task,
)


def build_task(region: str | None, seed: int, grid_size: int, data_dir: Path):
    if region is None:
        params = RouteParams(grid_size=grid_size, ruggedness=0.65, impassable_frac=0.04)
        return generate_synthetic_route_task(seed=seed, params=params), "Synthetic terrain"
    task, cost, elev = generate_route_from_dem(
        region=region, seed=seed, data_dir=data_dir, grid_size=grid_size
    )
    return (task, cost, elev), REGIONS[region].label


def _hillshade(elev: np.ndarray, azdeg: float = 315, altdeg: float = 45) -> np.ndarray:
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    return ls.hillshade(elev, vert_exag=1.5, dx=1, dy=1)


def visualize(
    elev: np.ndarray,
    cost: np.ndarray,
    path_cells: list[tuple[int, int]],
    optimal_cost: float,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="#111318")
    ax.set_facecolor("#111318")

    impassable = ~np.isfinite(cost)
    elev_min, elev_max = float(elev[np.isfinite(elev)].min()), float(elev.max())

    ls = LightSource(azdeg=315, altdeg=45)
    terrain = plt.get_cmap("gist_earth")
    rgb = ls.shade(
        elev,
        cmap=terrain,
        blend_mode="soft",
        vert_exag=2.0,
        dx=1, dy=1,
        vmin=elev_min, vmax=elev_max,
    )
    ax.imshow(rgb, origin="upper", interpolation="bilinear")

    n_contours = 10
    levels = np.linspace(elev_min, elev_max, n_contours)
    ax.contour(
        elev,
        levels=levels,
        colors="white",
        linewidths=0.4,
        alpha=0.35,
    )

    if impassable.any():
        water_overlay = np.zeros(cost.shape + (4,))
        water_overlay[impassable] = (0.15, 0.45, 0.85, 0.75)
        ax.imshow(water_overlay, origin="upper", interpolation="nearest")

    rs = [c[0] for c in path_cells]
    cs2 = [c[1] for c in path_cells]
    ax.plot(cs2, rs, color="black", lw=4.5, alpha=0.35, solid_capstyle="round")
    ax.plot(cs2, rs, color="#ff2d55", lw=2.4, solid_capstyle="round", label="optimal path")

    ax.scatter([cs2[0]], [rs[0]], s=140, marker="o",
               facecolor="#00ff88", edgecolor="black", linewidth=1.4, zorder=6, label="start")
    ax.scatter([cs2[-1]], [rs[-1]], s=180, marker="*",
               facecolor="#ffcc00", edgecolor="black", linewidth=1.4, zorder=6, label="goal")

    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.set_xticks([]); ax.set_yticks([])

    info = (
        f"grid: {elev.shape[0]}×{elev.shape[1]}\n"
        f"elev: {elev_min:.0f}–{elev_max:.0f} m\n"
        f"impassable: {int(impassable.sum())} cells\n"
        f"path length: {len(path_cells)} steps\n"
        f"optimal cost: {optimal_cost:.2f}"
    )
    ax.text(
        0.015, 0.015, info,
        transform=ax.transAxes,
        color="white", fontsize=9.5, family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#000000cc", edgecolor="#ffffff40"),
    )

    leg = ax.legend(loc="upper right", framealpha=0.85, facecolor="#000000cc", edgecolor="#ffffff40")
    for text in leg.get_texts():
        text.set_color("white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", choices=list(REGIONS.keys()), default=None,
                    help="named real-DEM region; omit for synthetic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=int, default=64, help="grid size (both axes)")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    (task_bundle, label) = build_task(args.region, args.seed, args.grid, args.data_dir)
    task, cost, elev = task_bundle
    task = solve_and_attach(task, cost)

    optimal_cells = [(int(r), int(c)) for r, c in task.ground_truth.optimal_solution]
    score = score_route_answer(task, ParsedAnswer(raw="", parsed=optimal_cells), cost)
    assert score.valid and abs(score.regret) < 1e-9, "oracle disagrees with itself"

    print(f"region:        {label}")
    print(f"task_id:       {task.task_id}")
    print(f"start / goal:  {task.params['start']} → {task.params['goal']}")
    print(f"optimal cost:  {task.ground_truth.optimal_value:.3f}")
    print(f"solver:        {task.ground_truth.solver} ({task.ground_truth.compute_ms} ms)")

    tag = args.region or "synthetic"
    out_path = args.out or Path("results") / f"demo_route_{tag}_seed{args.seed}.png"
    title = f"{label}   ·   seed {args.seed}   ·   optimal cost {task.ground_truth.optimal_value:.2f}"
    visualize(elev, cost, optimal_cells, task.ground_truth.optimal_value, title, out_path)


if __name__ == "__main__":
    main()
