"""Sweep the restricted-solver segment cap and print a summary table.

    uv run python scripts/sweep_segments.py --caps 3,5,8,12,20 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from geoagentbench.runner.config import (
    AgentConfig, Condition, RunConfig, Topology, ToolAccess,
)
from geoagentbench.runner.executor import estimate_cost, run_suite
from geoagentbench.runner.providers import OpenAIProvider, openai_cost_usd
from geoagentbench.runner.store import Store


def _parse_seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caps", default="3,5,8,12,20",
                    help="comma-separated segment caps to sweep")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--results-dir", type=Path, default=Path("results-sweep"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    caps = [int(x) for x in args.caps.split(",")]
    seeds = _parse_seeds(args.seeds)
    console = Console()

    console.print(f"[bold]sweep:[/bold] model={args.model} caps={caps} seeds={seeds} k={args.k}")
    total_est_cost = 0.0
    for cap in caps:
        cfg = RunConfig(
            family="route", grid_size=args.grid, seeds=seeds, k=args.k,
            agent=AgentConfig(
                model=args.model,
                condition=Condition.STRUCTURED_ONLY,
                topology=Topology.TOOL_LOOP,
                tool_access=ToolAccess.RESTRICTED_SOLVER,
                max_segment_cells=cap,
                provider="openai",
            ),
        )
        est = estimate_cost(cfg)
        console.print(f"  cap={cap}: {est.n_calls} calls, ~${est.est_usd:.3f}")
        total_est_cost += est.est_usd or 0.0

    console.print(f"[bold]total estimated cost:[/bold] ${total_est_cost:.3f}")
    if args.dry_run:
        return

    provider = OpenAIProvider()
    store = Store.open(args.results_dir)

    summary_rows = []
    for cap in caps:
        cfg = RunConfig(
            family="route", grid_size=args.grid, seeds=seeds, k=args.k,
            agent=AgentConfig(
                model=args.model,
                condition=Condition.STRUCTURED_ONLY,
                topology=Topology.TOOL_LOOP,
                tool_access=ToolAccess.RESTRICTED_SOLVER,
                max_segment_cells=cap,
                provider="openai",
            ),
        )
        console.rule(f"[bold]cap = {cap} cells[/bold]")

        def _on(i, n, outcome):
            s = outcome.record.score
            tag = "OK" if s.valid else "FAIL"
            regret = f"r={s.regret:.3f}" if s.valid else f"({s.failure.value})"
            console.print(f"  [{i:>2}/{n}] {tag} {regret}  tok={outcome.record.tokens_in}+{outcome.record.tokens_out}")

        res = run_suite(provider, cfg, store=store, on_progress=_on)
        infra = sum(1 for o in res.outcomes if o.record.error is not None)
        real_completed = len(res.outcomes) - infra
        valid = res.n_valid
        regrets = [o.record.score.regret for o in res.outcomes if o.record.score.valid]
        median = statistics.median(regrets) if regrets else None
        p25 = statistics.quantiles(regrets, n=4)[0] if len(regrets) >= 4 else None
        p75 = statistics.quantiles(regrets, n=4)[2] if len(regrets) >= 4 else None
        actual_usd = openai_cost_usd(args.model, res.total_tokens_in, res.total_tokens_out)
        summary_rows.append({
            "cap": cap, "n": len(res.outcomes), "infra": infra,
            "valid": valid, "success_of_completed": (valid / real_completed) if real_completed else None,
            "median_regret": median, "p25": p25, "p75": p75,
            "tokens_in": res.total_tokens_in, "tokens_out": res.total_tokens_out,
            "cost_usd": actual_usd,
        })

    console.rule("[bold]sweep summary[/bold]")
    tbl = Table(show_header=True, header_style="bold")
    for col in ["cap", "valid/total", "infra", "valid%*", "median regret", "IQR", "tokens", "cost"]:
        tbl.add_column(col)
    for r in summary_rows:
        iqr = f"{r['p25']:.2f}–{r['p75']:.2f}" if r["p25"] is not None else "-"
        sr = f"{r['success_of_completed']:.0%}" if r["success_of_completed"] is not None else "-"
        mr = f"{r['median_regret']:.3f}" if r["median_regret"] is not None else "-"
        tok = f"{r['tokens_in']:,}+{r['tokens_out']:,}"
        cost = f"${r['cost_usd']:.3f}" if r["cost_usd"] is not None else "-"
        tbl.add_row(str(r["cap"]), f"{r['valid']}/{r['n']}", str(r["infra"]), sr, mr, iqr, tok, cost)
    console.print(tbl)
    console.print("[dim]* valid% is over runs that actually completed (excludes infra failures).[/dim]")


if __name__ == "__main__":
    main()
