from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .regions import REGIONS
from .runner.config import AgentConfig, Condition, RunConfig, Topology, ToolAccess
from .runner.executor import estimate_cost, run_suite
from .runner.mock_agents import MockMode, responder_for
from .runner.providers import (
    MockProvider,
    OpenAIProvider,
    ProviderConfigError,
    openai_cost_usd,
)
from .runner.store import Store


app = typer.Typer(add_completion=False, help="GeoAgentBench CLI")
console = Console()


def _parse_seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


@app.command()
def run(
    seeds: Annotated[str, typer.Option(help="e.g. 0-4 or 0,2,5")] = "0-4",
    k: Annotated[int, typer.Option(help="repeats per (task, config)")] = 3,
    model: str = "gpt-4o-mini",
    condition: Annotated[Condition, typer.Option()] = Condition.STRUCTURED_ONLY,
    topology: Annotated[Topology, typer.Option(help='single_shot | plan_critique_revise | plan_verify_revise | tool_loop')] = Topology.SINGLE_SHOT,
    max_revise_rounds: Annotated[int, typer.Option('--max-revise-rounds', help='iterations for plan_verify_revise (default 1)')] = 1,
    tool_access: Annotated[ToolAccess, typer.Option('--tool-access', help='for tool_loop: none | calculator | restricted_solver | full_solver')] = ToolAccess.FULL_SOLVER,
    max_segment_cells: Annotated[Optional[int], typer.Option('--max-segment-cells', help='restricted_solver: cap on solve_between segment length (default 5)')] = None,
    max_input_tokens_per_task: Annotated[int, typer.Option('--max-input-tokens', help='tool_loop budget per task; 0 = unlimited (default 60000)')] = 60_000,
    message_window: Annotated[int, typer.Option('--message-window', help='tool_loop: how many recent messages to keep (default 12; 0 = keep all)')] = 12,
    max_infra_retries: Annotated[int, typer.Option('--max-infra-retries', help='retries on rate limit/timeout (default 5)')] = 5,
    infra_backoff: Annotated[float, typer.Option('--infra-backoff', help='base seconds for exponential backoff (default 4)')] = 4.0,
    temperature: float = 0.7,
    grid_size: Annotated[int, typer.Option("--grid", help="LLM grid size (small!)")] = 16,
    region: Annotated[Optional[str], typer.Option(help="named region or omit for synthetic")] = None,
    prompt_version: str = "route/v1",
    max_tokens: int = 4096,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="estimate cost, don't call the API")] = False,
    mock: Annotated[bool, typer.Option("--mock", help="use MockProvider (skips OpenAI)")] = False,
    mock_mode: Annotated[MockMode, typer.Option("--mock-mode", help="mock behavior: oracle|straight|random|silent")] = MockMode.ORACLE,
    results_dir: Path = Path("results"),
    env_file: Path = Path(".env"),
) -> None:
    """Run a benchmark suite."""
    if env_file.exists():
        load_dotenv(env_file)

    if region is not None and region not in REGIONS:
        raise typer.BadParameter(f"unknown region {region!r}; try one of {list(REGIONS)}")

    provider_name = f"mock:{mock_mode.value}" if mock else "openai"
    cfg = RunConfig(
        family="route",
        grid_size=grid_size,
        seeds=_parse_seeds(seeds),
        k=k,
        agent=AgentConfig(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            prompt_version=prompt_version,
            condition=condition,
            topology=topology,
            tool_access=tool_access,
            max_segment_cells=max_segment_cells,
            max_input_tokens_per_task=max_input_tokens_per_task,
            message_window=message_window,
            provider=provider_name,
            max_revise_rounds=max_revise_rounds,
        ),
        max_infra_retries=max_infra_retries,
        infra_retry_backoff_s=infra_backoff,
    )

    est = estimate_cost(cfg, region=region)
    console.print(f"[bold]suite:[/bold] {cfg.family}  seeds={cfg.seeds}  k={cfg.k}  grid={cfg.grid_size}  region={region or 'synthetic'}")
    console.print(f"[bold]agent:[/bold] {cfg.agent.model}  T={cfg.agent.temperature}  condition={cfg.agent.condition.value}")
    console.print(f"[bold]est:[/bold] {est.n_calls} calls · ~{est.tokens_in_total:,} tokens in / ~{est.tokens_out_total:,} out"
                  + (f"  ~${est.est_usd:.4f}" if est.est_usd is not None else ""))
    if dry_run:
        console.print("[yellow]--dry-run — not calling the API.[/yellow]")
        raise typer.Exit(0)

    if mock:
        provider = MockProvider(responder=responder_for(mock_mode))
        console.print(f"[cyan]using MockProvider (mode={mock_mode.value}, no LLM calls made)[/cyan]")
    else:
        try:
            provider = OpenAIProvider()
        except ProviderConfigError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    store = Store.open(results_dir)

    def on_progress(i: int, n: int, outcome) -> None:
        s = outcome.record.score
        tag = "[green]OK[/green]" if s.valid else "[red]FAIL[/red]"
        regret = f"regret={s.regret:.3f}" if s.valid else f"({s.failure.value})"
        console.print(f"  [{i:>3}/{n}] {tag}  task={outcome.record.task_id[:10]}  {regret}  tok={outcome.record.tokens_in}+{outcome.record.tokens_out}")

    result = run_suite(provider, cfg, store=store, on_progress=on_progress, region=region)

    console.print(f"\n[bold]done.[/bold]  valid: {result.n_valid}/{len(result.outcomes)}  "
                  f"median_regret: {result.median_regret if result.median_regret is not None else 'n/a'}")
    console.print(f"total tokens: {result.total_tokens_in:,} in / {result.total_tokens_out:,} out"
                  + (f"  actual $: {openai_cost_usd(cfg.agent.model, result.total_tokens_in, result.total_tokens_out):.4f}"
                     if openai_cost_usd(cfg.agent.model, result.total_tokens_in, result.total_tokens_out) is not None else ""))
    console.print(f"stored under: [cyan]{results_dir}/geoab.db[/cyan]")


@app.command("list")
def list_cmd(
    what: Annotated[str, typer.Argument(help="runs | configs | tasks")] = "runs",
    limit: int = 20,
    results_dir: Path = Path("results"),
) -> None:
    """List what's in the store."""
    store = Store.open(results_dir)
    tbl = Table(show_header=True, header_style="bold")
    if what == "runs":
        rows = store.list_runs(limit=limit)
        for col in ["run_id", "task_id", "config_id", "valid", "regret", "failure", "tokens_in", "tokens_out"]:
            tbl.add_column(col)
        for r in rows:
            tbl.add_row(
                r["run_id"],
                r["task_id"][:12],
                r["config_id"][:16],
                "✓" if r["valid"] else "✗",
                f"{r['regret']:.3f}" if r["regret"] is not None else "-",
                r["failure"],
                str(r["tokens_in"]), str(r["tokens_out"]),
            )
    elif what == "configs":
        rows = store.list_configs()
        for col in ["config_id", "model", "temp", "condition", "topology"]:
            tbl.add_column(col)
        for r in rows:
            tbl.add_row(r["config_id"], r["model"], f"{r['temperature']:.2f}", r["condition"], r["topology"])
    elif what == "tasks":
        rows = store.list_tasks(limit=limit)
        for col in ["task_id", "family", "seed"]:
            tbl.add_column(col)
        for r in rows:
            tbl.add_row(r["task_id"], r["family"], str(r["seed"]))
    else:
        raise typer.BadParameter(f"unknown target {what!r}")
    console.print(tbl)


@app.command()
def show(
    run_id: str,
    results_dir: Path = Path("results"),
) -> None:
    """Show one run — score, answer, traces. Accepts a run_id prefix."""
    store = Store.open(results_dir)
    run = store.get_run(run_id)
    if not run:
        candidates = [r for r in store.list_runs(limit=10_000) if r["run_id"].startswith(run_id)]
        if len(candidates) == 1:
            run = candidates[0]
        elif len(candidates) > 1:
            console.print(f"[red]prefix {run_id!r} matches {len(candidates)} runs — be more specific[/red]")
            for c in candidates[:5]:
                console.print(f"  {c['run_id']}")
            raise typer.Exit(1)
        else:
            console.print(f"[red]no run matching {run_id!r}[/red]")
            raise typer.Exit(1)

    console.rule(f"[bold]{run['run_id']}[/bold]")
    console.print(f"task={run['task_id']}  config={run['config_id']}")
    verdict = "[green]VALID[/green]" if run["valid"] else f"[red]FAIL[/red] ({run['failure']})"
    console.print(f"{verdict}  regret={run['regret']}  tokens={run['tokens_in']}+{run['tokens_out']}  wall={run['wall_ms']}ms")
    if run["notes"]:
        console.print(f"notes: {run['notes']}")
    if run["error"]:
        console.print(f"[yellow]infra error:[/yellow] {run['error']}")

    import json as _json
    answer = _json.loads(run["answer_json"])
    console.rule("[bold]agent output (raw)[/bold]")
    console.print(answer.get("raw", ""))
    if answer.get("parsed"):
        console.rule(f"[bold]parsed path[/bold]  ({len(answer['parsed'])} cells)")
        console.print(answer["parsed"])

    traces = store.read_traces(run["run_id"])
    if traces:
        console.rule(f"[bold]traces[/bold] ({len(traces)})")
        for t in traces:
            console.print_json(_json.dumps(t))


@app.command()
def prompt(
    seed: int = 0,
    grid_size: Annotated[int, typer.Option("--grid")] = 16,
    condition: Condition = Condition.STRUCTURED_ONLY,
    region: Optional[str] = None,
    prompt_version: str = "route/v1",
    show_grid: Annotated[bool, typer.Option("--show-grid/--hide-grid")] = True,
) -> None:
    """Render the exact prompt the model would receive — no API call."""
    from .runner.executor import build_task
    from .runner.prompts import approx_tokens, render_prompt

    if region is not None and region not in REGIONS:
        raise typer.BadParameter(f"unknown region {region!r}; try one of {list(REGIONS)}")

    task, _cost, _elev = build_task("route", seed=seed, grid_size=grid_size, region=region)
    system, user = render_prompt(task, condition, prompt_version)
    if not show_grid:
        import re as _re
        user = _re.sub(r'"cost_grid":\s*\[\[.+?\]\]', '"cost_grid": "<omitted>"', user, count=1, flags=_re.DOTALL)

    console.rule(f"[bold]task[/bold]  {task.task_id}  seed={seed}  grid={grid_size}  region={region or 'synthetic'}")
    console.print(f"start={task.params['start']}  goal={task.params['goal']}  optimal_cost=[hidden]")

    console.rule("[bold]system[/bold]")
    console.print(system)
    console.rule(f"[bold]user[/bold]  ({len(user):,} chars, ~{approx_tokens(user):,} tokens)")
    console.print(user)

    console.rule("[bold]sizes[/bold]")
    console.print(f"system: {len(system):,} chars   ~{approx_tokens(system):,} tokens")
    console.print(f"user:   {len(user):,} chars   ~{approx_tokens(user):,} tokens")
    console.print(f"total:  ~{approx_tokens(system) + approx_tokens(user):,} input tokens per call")


@app.command()
def preflight(
    results_dir: Path = Path("results-preflight"),
    grid_size: int = 12,
) -> None:
    """Smoke-test the whole pipeline offline: every mock mode × every condition."""
    from .runner.mock_agents import MockMode
    import shutil

    shutil.rmtree(results_dir, ignore_errors=True)
    store = Store.open(results_dir)
    conditions = [Condition.NARRATIVE_ONLY, Condition.STRUCTURED_ONLY, Condition.BOTH]
    modes = [MockMode.ORACLE, MockMode.RANDOM, MockMode.STRAIGHT, MockMode.SILENT]

    from .runner.config import AgentConfig, RunConfig, Topology
    from .runner.executor import run_suite
    from .runner.providers import MockProvider

    results_summary = []
    for mode in modes:
        for cond in conditions:
            cfg = RunConfig(
                seeds=[0, 1, 2], k=1, grid_size=grid_size,
                agent=AgentConfig(
                    model=f"mock-{mode.value}",
                    condition=cond,
                    topology=Topology.SINGLE_SHOT,
                ),
            )
            provider = MockProvider(responder=responder_for(mode))
            res = run_suite(provider, cfg, store=store)
            failures = {}
            for o in res.outcomes:
                cat = o.record.score.failure.value if not o.record.score.valid else "valid"
                failures[cat] = failures.get(cat, 0) + 1
            results_summary.append((mode.value, cond.value, res.n_valid, len(res.outcomes), failures))

    tbl = Table(show_header=True, header_style="bold")
    for col in ["mock_mode", "condition", "valid/total", "outcomes"]:
        tbl.add_column(col)
    for mode, cond, nv, nt, fails in results_summary:
        outc = " ".join(f"{k}={v}" for k, v in sorted(fails.items()))
        tbl.add_row(mode, cond, f"{nv}/{nt}", outc)
    console.print(tbl)

    all_categories = set()
    for _, _, _, _, fails in results_summary:
        all_categories.update(fails.keys())
    console.print(f"\n[bold]categories hit:[/bold] {sorted(all_categories)}")
    console.print(f"[bold]configs written:[/bold] {len(store.list_configs())}")
    console.print(f"[bold]runs written:[/bold]    {len(store.list_runs(limit=10_000))}")
    console.print(f"[green]preflight OK[/green]  (see {results_dir}/geoab.db)")


if __name__ == "__main__":
    app()
