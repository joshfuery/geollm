import { useEffect, useMemo, useState } from "react";
import { api, type Config, type Health, type Region, type Task } from "./api";
import { fmtPct, RunDetail, RunGroups } from "./components";
import type { Scene } from "./layers";
import { loadScene } from "./scene";
import { Viewer } from "./Viewer";

export function BrowseView({ header }: { header: React.ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [regions, setRegions] = useState<Region[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [configs, setConfigs] = useState<Map<string, Config>>(new Map());
  const [taskId, setTaskId] = useState<string | null>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showInvalid, setShowInvalid] = useState(true);
  const [hiddenConfigs, setHiddenConfigs] = useState<Set<string>>(new Set());
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.health(), api.regions(), api.tasks(), api.configs()])
      .then(([h, r, t, c]) => {
        setHealth(h);
        setRegions(r);
        setTasks(t);
        setConfigs(new Map(c.map((x) => [x.config_id, x])));
        const first = t.find((x) => x.source === "dem") ?? t[0];
        if (first) setTaskId(first.task_id);
      })
      .catch((e) => setError(`Can't reach the API — is uvicorn running on :8000? (${e.message})`));
  }, []);

  useEffect(() => {
    const task = tasks.find((t) => t.task_id === taskId);
    if (!task) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    loadScene(task, "main")
      .then((s) => {
        if (cancelled) return;
        setScene(s);
        setSelectedRunId(null);
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [taskId, tasks]);

  const visibleRunIds = useMemo(() => {
    const ids = new Set<string>();
    for (const r of scene?.runs ?? []) {
      if (!showInvalid && !r.valid) continue;
      if (!hiddenConfigs.has(r.config_id)) ids.add(r.run_id);
    }
    return ids;
  }, [scene, hiddenConfigs, showInvalid]);

  const regionLabel = (name: string | null) => regions.find((r) => r.name === name)?.label ?? name ?? "Synthetic";
  const tasksByRegion = useMemo(() => {
    const m = new Map<string, Task[]>();
    for (const t of tasks) {
      const key = t.source === "dem" ? regionLabel(t.region_name) : "Synthetic";
      m.set(key, [...(m.get(key) ?? []), t]);
    }
    return [...m.entries()];
  }, [tasks, regions]);

  const selectedRun = scene?.runs.find((r) => r.run_id === selectedRunId) ?? null;

  const f = scene?.grid.frame;
  const nValid = scene?.runs.filter((r) => r.valid).length ?? 0;

  return (
    <>
      <aside className="sidebar">
        {header}
        {error && <div className="error">{error}</div>}

        <section className="block">
          <h2 className="label">
            Task
            {health && <span className="num">{health.results_dir.split(/[\\/]/).pop()} · {health.counts.runs} runs</span>}
          </h2>
          <select value={taskId ?? ""} onChange={(e) => setTaskId(e.target.value)} aria-label="Task">
            {tasksByRegion.map(([label, ts]) => (
              <optgroup key={label} label={label}>
                {ts.map((t) => (
                  <option key={t.task_id} value={t.task_id}>
                    {t.grid_size}×{t.grid_size} · seed {t.seed}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          {scene && f && (
            <div className="place">
              <h3>{f.region_label}</h3>
              <div className="stats">
                <div><b>Grid</b><span className="num">{scene.grid.grid_size}²</span></div>
                <div><b>Blocked</b><span className="num">{scene.grid.blocked.length}</span></div>
                <div><b>Optimal</b><span className="num">{scene.task.ground_truth.optimal_value?.toFixed(1) ?? "—"}</span></div>
                <div><b>Valid</b><span className="num">{nValid}/{scene.runs.length}</span></div>
              </div>
            </div>
          )}
        </section>

        <section className="block">
          <h2 className="label">
            Configs
            {scene && scene.runs.length > 0 && (
              <button className="link" onClick={() =>
                setHiddenConfigs(hiddenConfigs.size ? new Set() : new Set(scene.runs.map((r) => r.config_id)))}>
                {hiddenConfigs.size ? "Show all" : "Hide all"}
              </button>
            )}
          </h2>
          {scene && !scene.runs.length && <p className="muted">No runs for this task.</p>}
          {scene && (
            <RunGroups runs={scene.runs} configs={configs} hiddenConfigs={hiddenConfigs}
              setHiddenConfigs={setHiddenConfigs} selectedRunId={selectedRunId} onSelect={setSelectedRunId} />
          )}
          {scene && scene.runs.some((r) => !r.valid) && (
            <label className="check" style={{ marginTop: 10 }}>
              <input type="checkbox" checked={showInvalid} onChange={(e) => setShowInvalid(e.target.checked)} />
              Draw invalid routes ({fmtPct((scene.runs.length - nValid) / scene.runs.length)})
            </label>
          )}
        </section>

        {selectedRun && (
          <RunDetail run={selectedRun} config={configs.get(selectedRun.config_id)}
            optimal={scene?.task.ground_truth.optimal_value ?? null} onClose={() => setSelectedRunId(null)} />
        )}
      </aside>

      <main className="map">
        <Viewer
          scene={scene}
          configs={configs}
          visibleRunIds={visibleRunIds}
          selectedRunId={selectedRunId}
          onSelectRun={setSelectedRunId}
          onError={setError}
          loading={loading ? "Loading terrain…" : null}
        />
      </main>
    </>
  );
}
