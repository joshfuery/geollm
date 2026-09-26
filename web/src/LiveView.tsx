import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  argPoint,
  type Cell,
  type Config,
  type LiveEvent,
  type LiveOptions,
  type NewTaskRequest,
  type Run,
} from "./api";
import { failureLabel, fmtRegret, RunDetail, RunList } from "./components";
import { runColor, type LiveOverlay, type Scene } from "./layers";
import { loadScene } from "./scene";
import { Viewer } from "./Viewer";

type Source = NewTaskRequest["source"];

interface AgentForm {
  provider: string;
  model: string;
  topology: string;
  tool_access: string;
  condition: string;
  max_segment_cells: number;
  max_revise_rounds: number;
}

const PRESETS: { label: string; hint: string; form: Partial<AgentForm> }[] = [
  { label: "Single shot", hint: "one call", form: { topology: "single_shot", tool_access: "none" } },
  { label: "Verify loop", hint: "3 checked revisions", form: { topology: "plan_verify_revise", tool_access: "none", max_revise_rounds: 3 } },
  { label: "Capped solver", hint: "short legs only", form: { topology: "tool_loop", tool_access: "restricted_solver", max_segment_cells: 5 } },
  { label: "Full solver", hint: "delegation ceiling", form: { topology: "tool_loop", tool_access: "full_solver" } },
];

const PROVIDERS = [
  { value: "openai", label: "OpenAI" },
  { value: "mock:composer", label: "Demo: waypoint composer" },
  { value: "mock:random", label: "Demo: random walk" },
  { value: "mock:straight", label: "Demo: straight line" },
  { value: "mock:oracle", label: "Demo: oracle replay" },
];

const MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o4-mini"];

const nice = (s: string) => s.replace(/_/g, " ");

function overlayFrom(events: LiveEvent[]): LiveOverlay {
  let draft: Cell[] | null = null;
  const legs: Cell[][] = [];
  let pending: [Cell, Cell] | null = null;
  for (const ev of events) {
    if (ev.type === "llm") {
      if (ev.draft) draft = ev.draft;
      const call = ev.tool_calls.find((t) => t.name === "solve_between");
      if (call) {
        const a = argPoint(call.args, "from");
        const b = argPoint(call.args, "to");
        pending = a && b ? [a, b] : null;
      }
    } else if (ev.type === "tool") {
      if (ev.name === "solve_between") {
        pending = null;
        if (ev.ok && ev.cells) legs.push(ev.cells);
      } else if (ev.name === "check_path" && ev.draft) {
        draft = ev.draft;
      }
    }
  }
  return { draft, legs, pending };
}

const pt = (c: Cell | null) => (c ? `${c[0]},${c[1]}` : "?");

function LogLine({ ev, last }: { ev: LiveEvent; last: boolean }) {
  const line = (n: string, k: string, body: React.ReactNode, cls = "") => (
    <li><span className="n">{n}</span><span className={`k ${cls}`}>{k}</span><span className={cls}>{body}</span></li>
  );
  switch (ev.type) {
    case "thinking":
      return last ? line(String(ev.step).padStart(2, "0"), "…", <span className="wait">waiting on model</span>) : null;
    case "llm": {
      const n = String(ev.step).padStart(2, "0");
      const c = ev.tool_calls[0];
      if (c?.name === "solve_between") return line(n, "ask", `solve ${pt(argPoint(c.args, "from"))} → ${pt(argPoint(c.args, "to"))}`);
      if (c?.name === "check_path") return line(n, "ask", `check ${Array.isArray(c.args.cells) ? c.args.cells.length : "?"} cells`);
      if (c) return line(n, "ask", c.name);
      return line(n, "say", ev.draft ? `path, ${ev.draft.length} cells` : "no path");
    }
    case "tool":
      if (ev.name === "solve_between")
        return ev.ok
          ? line("", "↳", `${ev.cells?.length ?? 0} cells, cost ${ev.cost?.toFixed(1)}`, "ok")
          : line("", "↳", String(ev.error ?? "refused").split(".")[0], "no");
      if (ev.name === "check_path")
        return ev.ok
          ? line("", "↳", "path checks out", "ok")
          : line("", "↳", `${ev.n_issues} issue${ev.n_issues === 1 ? "" : "s"}: ${ev.issues?.[0] ?? ""}`, "no");
      return line("", "↳", ev.error ?? ev.name);
    case "infra_error":
      return line("", "!", "API error, retrying", "no");
    case "error":
      return line("", "!", ev.message, "no");
    default:
      return null;
  }
}

export function LiveView({ header }: { header: React.ReactNode }) {
  const [options, setOptions] = useState<LiveOptions | null>(null);
  const [source, setSource] = useState<Source>("region");
  const [region, setRegion] = useState<string>("");
  const [gridSize, setGridSize] = useState(16);
  const [scene, setScene] = useState<Scene | null>(null);
  const [configs, setConfigs] = useState<Map<string, Config>>(new Map());
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [form, setForm] = useState<AgentForm>({
    provider: "mock:composer",
    model: "gpt-4o-mini",
    topology: "tool_loop",
    tool_access: "restricted_solver",
    condition: "structured_only",
    max_segment_cells: 5,
    max_revise_rounds: 3,
  });

  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [lastRun, setLastRun] = useState<Run | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const logRef = useRef<HTMLOListElement | null>(null);

  const [hiddenRuns, setHiddenRuns] = useState<Set<string>>(new Set());
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const refreshConfigs = useCallback(
    () => api.configs("live").then((c) => setConfigs(new Map(c.map((x) => [x.config_id, x])))),
    [],
  );

  const newLocation = useCallback(
    async (src: Source = source) => {
      esRef.current?.close();
      setRunning(false);
      setEvents([]);
      setLastRun(null);
      setSelectedRunId(null);
      setHiddenRuns(new Set());
      setError(null);
      setLoading(src === "anywhere" ? "Fetching terrain…" : "Surveying…");
      try {
        const task = await api.newLiveTask({ source: src, region: src === "region" ? region || null : null, grid_size: gridSize });
        setLoading("Loading terrain…");
        setScene(await loadScene(task, "live"));
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(null);
      }
    },
    [source, region, gridSize],
  );

  useEffect(() => {
    api
      .liveOptions()
      .then((o) => {
        setOptions(o);
        if (o.openai) setForm((f) => ({ ...f, provider: "openai" }));
        refreshConfigs();
        const src: Source = o.regions.length ? "region" : "synthetic";
        setSource(src);
        void newLocation(src);
      })
      .catch((e) => setError(`Can't reach the API — is uvicorn running on :8000? (${e.message})`));
    return () => esRef.current?.close();
  }, []);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [events]);

  const runAgent = async () => {
    if (!scene) return;
    esRef.current?.close();
    setEvents([]);
    setLastRun(null);
    setSelectedRunId(null);
    setError(null);
    const isComposer = form.provider === "mock:composer";
    try {
      const { live_id } = await api.startLiveRun({
        task_id: scene.task.task_id,
        provider: form.provider,
        model: form.model,
        topology: isComposer ? "tool_loop" : form.topology,
        tool_access: isComposer && form.tool_access === "none" ? "restricted_solver" : form.tool_access,
        condition: form.condition,
        max_segment_cells: form.tool_access === "restricted_solver" || isComposer ? form.max_segment_cells : null,
        max_revise_rounds: form.max_revise_rounds,
      });
      setRunning(true);
      const es = new EventSource(api.liveEventsUrl(live_id));
      esRef.current = es;
      es.onmessage = (msg) => {
        const ev = JSON.parse(msg.data) as LiveEvent;
        setEvents((prev) => [...prev, ev]);
        if (ev.type === "result") {
          setLastRun(ev.run);
          setScene((s) => (s && s.task.task_id === ev.run.task_id ? { ...s, runs: [...s.runs, ev.run] } : s));
          setSelectedRunId(ev.run.run_id);
          void refreshConfigs();
        }
        if (ev.type === "done") {
          es.close();
          setRunning(false);
        }
      };
      es.onerror = () => {
        es.close();
        setRunning(false);
      };
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const overlay = useMemo(() => (running ? overlayFrom(events) : null), [events, running]);

  const visibleRunIds = useMemo(() => {
    const ids = new Set<string>();
    for (const r of scene?.runs ?? []) {
      if (running) continue;
      if (!hiddenRuns.has(r.run_id)) ids.add(r.run_id);
    }
    return ids;
  }, [scene, hiddenRuns, running]);

  const set = (patch: Partial<AgentForm>) => setForm((f) => ({ ...f, ...patch }));
  const isMock = form.provider.startsWith("mock:");
  const isComposer = form.provider === "mock:composer";
  const optimal = scene?.task.ground_truth.optimal_value ?? null;
  const selectedRun = scene?.runs.find((r) => r.run_id === selectedRunId) ?? null;
  const tokens = events.reduce(
    (acc, e) => (e.type === "llm" ? [acc[0] + e.tokens_in, acc[1] + e.tokens_out] : acc),
    [0, 0],
  );

  const lastText = [...events].reverse().find((e) => e.type === "llm" && e.text) as
    | Extract<LiveEvent, { type: "llm" }>
    | undefined;
  const elapsed = events.length ? events[events.length - 1].t_ms / 1000 : 0;
  const f = scene?.grid.frame;
  const lat = scene ? (scene.bbox[1] + scene.bbox[3]) / 2 : 0;
  const lon = scene ? (scene.bbox[0] + scene.bbox[2]) / 2 : 0;

  return (
    <>
      <aside className="sidebar">
        {header}
        {error && <div className="error">{error}</div>}

        <section className="block">
          <h2 className="label">Location</h2>
          <div className="seg" role="group" aria-label="Location source">
            {([
              ["region", "Cached", "Random window inside a cached DEM"],
              ["anywhere", "Anywhere", options?.opentopo ? "Random mountain range via OpenTopography" : "Set OPENTOPO_API_KEY in .env"],
              ["synthetic", "Synthetic", "Fractal terrain"],
            ] as [Source, string, string][]).map(([v, label, hint]) => (
              <button key={v} className={source === v ? "on" : ""} title={hint}
                disabled={(v === "anywhere" && !options?.opentopo) || (v === "region" && !options?.regions.length)}
                onClick={() => setSource(v)}>
                {label}
              </button>
            ))}
          </div>
          <div className="row">
            {source === "region" && (
              <select value={region} onChange={(e) => setRegion(e.target.value)} aria-label="Region">
                <option value="">Any region</option>
                {options?.regions.map((r) => <option key={r.name} value={r.name}>{r.label}</option>)}
              </select>
            )}
            <select value={gridSize} onChange={(e) => setGridSize(Number(e.target.value))} aria-label="Grid size"
              title="Size of the cost grid in the prompt">
              {[12, 16, 20, 24, 32].map((g) => <option key={g} value={g}>{g}×{g} grid</option>)}
            </select>
          </div>
          <button className="go ghost" disabled={!!loading || running} onClick={() => void newLocation()}>
            New location
          </button>

          {scene && f && (
            <div className="place">
              <h3>{f.region_label.replace(/, (USA|Swiss Alps)$/, "")}</h3>
              {f.kind === "dem" && (
                <div className="coords num">
                  {Math.abs(lat).toFixed(3)}°{lat >= 0 ? "N" : "S"} {Math.abs(lon).toFixed(3)}°{lon >= 0 ? "E" : "W"}
                </div>
              )}
              <div className="stats">
                <div><b>Grid</b><span className="num">{scene.grid.grid_size}²</span></div>
                <div><b>Cell</b><span className="num">{f.kind === "dem" ? `${Math.round(f.pixel_size_m ?? 0)} m` : "—"}</span></div>
                <div><b>Relief</b><span className="num">{Math.round(scene.elevation.max - scene.elevation.min)} m</span></div>
                <div><b>Optimal</b><span className="num">{optimal?.toFixed(1) ?? "—"}</span></div>
              </div>
            </div>
          )}
        </section>

        <section className="block">
          <h2 className="label">Agent</h2>
          <div className="grid2" role="group" aria-label="Setup">
            {PRESETS.map((p) => {
              const on = Object.entries(p.form).every(([k, v]) => form[k as keyof AgentForm] === v);
              return (
                <button key={p.label} className={on ? "on" : ""} onClick={() => set(p.form)}>
                  {p.label}<small>{p.hint}</small>
                </button>
              );
            })}
          </div>
          <div className="row">
            <select value={form.provider} onChange={(e) => set({ provider: e.target.value })} aria-label="Provider">
              {PROVIDERS.map((p) => (
                <option key={p.value} value={p.value} disabled={p.value === "openai" && !options?.openai}>
                  {p.label}{p.value === "openai" && !options?.openai ? " (no key)" : ""}
                </option>
              ))}
            </select>
            {!isMock && (
              <>
                <input list="models" aria-label="Model" value={form.model} onChange={(e) => set({ model: e.target.value })} />
                <datalist id="models">{MODELS.map((m) => <option key={m} value={m} />)}</datalist>
              </>
            )}
          </div>
          <details className="more">
            <summary>Options</summary>
            <label className="field">
              <span>Topology</span>
              <select value={isComposer ? "tool_loop" : form.topology} disabled={isComposer}
                onChange={(e) => set({ topology: e.target.value })}>
                {options?.topologies.map((t) => <option key={t} value={t}>{nice(t)}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Tools</span>
              <select value={form.tool_access} onChange={(e) => set({ tool_access: e.target.value })}
                disabled={!isComposer && form.topology !== "tool_loop"}>
                {options?.tool_access.map((t) => <option key={t} value={t}>{nice(t)}</option>)}
              </select>
            </label>
            {(form.tool_access === "restricted_solver" || isComposer) && (
              <label className="field">
                <span>Leg cap</span>
                <input type="number" min={2} max={100} value={form.max_segment_cells}
                  onChange={(e) => set({ max_segment_cells: Math.max(2, Number(e.target.value) || 2) })} />
              </label>
            )}
            {form.topology === "plan_verify_revise" && !isComposer && (
              <label className="field">
                <span>Rounds</span>
                <input type="number" min={1} max={5} value={form.max_revise_rounds}
                  onChange={(e) => set({ max_revise_rounds: Math.min(5, Math.max(1, Number(e.target.value) || 1)) })} />
              </label>
            )}
            <label className="field">
              <span>Prompt</span>
              <select value={form.condition} onChange={(e) => set({ condition: e.target.value })}>
                {options?.conditions.map((c) => <option key={c} value={c}>{nice(c)}</option>)}
              </select>
            </label>
          </details>
          <button className="go" disabled={!scene || running || !!loading} onClick={() => void runAgent()}>
            {running ? "Running…" : "Run agent"}
          </button>
        </section>

        {(events.length > 0 || lastRun) && (
          <section className="block">
            <h2 className="label">
              Run
              <span className="num">{tokens[0] + tokens[1] > 0 && `${(tokens[0] + tokens[1]).toLocaleString()} tok · `}{elapsed.toFixed(1)} s</span>
            </h2>
            {lastRun && (
              <div className={`verdict ${lastRun.valid ? "good" : "bad"}`}>
                <span className="big" style={{ color: lastRun.error ? undefined : `rgb(${runColor(lastRun).slice(0, 3).join(",")})` }}>{lastRun.error ? "Error" : lastRun.valid ? fmtRegret(lastRun.regret) : "Invalid"}</span>
                <span className="sub">
                  {lastRun.valid && lastRun.agent_value != null && optimal != null
                    ? `${lastRun.agent_value.toFixed(1)} vs ${optimal.toFixed(1)}`
                    : lastRun.error ? "not scored" : failureLabel(lastRun.failure)}
                </span>
              </div>
            )}
            <ol className="log" ref={logRef}>
              {events.map((ev, i) => <LogLine key={ev.seq} ev={ev} last={i === events.length - 1} />)}
            </ol>
            {lastText && !running && (
              <details className="raw">
                <summary>Model output</summary>
                <pre>{lastText.truncated ? "…" : ""}{lastText.text}</pre>
              </details>
            )}
          </section>
        )}

        {scene && scene.runs.length > 0 && (
          <section className="block">
            <h2 className="label">History <span className="num">{scene.runs.length}</span></h2>
            <RunList runs={scene.runs} configs={configs} hidden={hiddenRuns} setHidden={setHiddenRuns}
              selectedRunId={selectedRunId} onSelect={setSelectedRunId} />
          </section>
        )}

        {selectedRun && !running && selectedRun.run_id !== lastRun?.run_id && (
          <RunDetail run={selectedRun} config={configs.get(selectedRun.config_id)} optimal={optimal}
            onClose={() => setSelectedRunId(null)} />
        )}
      </aside>

      <main className="map">
        <Viewer
          scene={scene}
          configs={configs}
          visibleRunIds={visibleRunIds}
          selectedRunId={selectedRunId}
          onSelectRun={setSelectedRunId}
          overlay={overlay}
          onError={setError}
          loading={loading}
        />
      </main>
    </>
  );
}
