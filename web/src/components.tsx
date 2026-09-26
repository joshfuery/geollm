import { useMemo } from "react";
import { configLabel, type Config, type Run } from "./api";
import { runColor } from "./layers";

export const fmtRegret = (x: number | null) => {
  const v = (x ?? 0) * 100;
  return v === 0 ? "optimal" : `+${v < 10 ? v.toFixed(1) : Math.round(v)}%`;
};
export const fmtPct = (x: number) => `${Math.round(x * 100)}%`;

export const median = (xs: number[]) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export function toggled<T>(set: Set<T>, v: T): Set<T> {
  const next = new Set(set);
  if (next.has(v)) next.delete(v);
  else next.add(v);
  return next;
}

export const failureLabel = (f: string) =>
  ({
    infeasible_step: "illegal step",
    hallucinated_coord: "off-grid cell",
    format_violation: "no path given",
    ignored_constraint: "wrong endpoints",
    arithmetic_error: "arithmetic error",
    tool_call_malformed: "bad tool call",
  })[f] ?? f.replace(/_/g, " ");

export function runSummary(r: Run): string {
  if (r.error) return "API error";
  return r.valid ? fmtRegret(r.regret) : failureLabel(r.failure);
}

export function Swatch({ run }: { run: Run }) {
  return (
    <i
      className={`swatch ${run.valid ? "" : "dashed"}`}
      style={{ background: `rgb(${runColor(run).slice(0, 3).join(",")})` }}
    />
  );
}

export function RunList(props: {
  runs: Run[];
  configs: Map<string, Config>;
  hidden: Set<string>;
  setHidden: (s: Set<string>) => void;
  selectedRunId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const runs = [...props.runs].sort((a, b) => b.created_at.localeCompare(a.created_at));
  return (
    <ul className="runs">
      {runs.map((r) => (
        <li key={r.run_id} className={`${r.run_id === props.selectedRunId ? "sel" : ""} ${props.hidden.has(r.run_id) ? "off" : ""}`}>
          <input type="checkbox" aria-label="show on map" checked={!props.hidden.has(r.run_id)}
            onChange={() => props.setHidden(toggled(props.hidden, r.run_id))} />
          <Swatch run={r} />
          <button onClick={() => props.onSelect(r.run_id === props.selectedRunId ? null : r.run_id)}
            title={configLabel(props.configs.get(r.config_id))}>
            {configLabel(props.configs.get(r.config_id))}
          </button>
          <span className="num">{runSummary(r)}</span>
        </li>
      ))}
    </ul>
  );
}

export function RunGroups(props: {
  runs: Run[];
  configs: Map<string, Config>;
  hiddenConfigs: Set<string>;
  setHiddenConfigs: (s: Set<string>) => void;
  selectedRunId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const { runs, configs, hiddenConfigs, setHiddenConfigs, selectedRunId, onSelect } = props;
  const groups = useMemo(() => {
    const m = new Map<string, Run[]>();
    for (const r of runs) m.set(r.config_id, [...(m.get(r.config_id) ?? []), r]);
    return [...m.entries()]
      .map(([cid, rs]) => {
        const valid = rs.filter((r) => r.valid);
        return {
          cid,
          runs: [...rs].sort((a, b) => Number(b.valid) - Number(a.valid) || (a.regret ?? 0) - (b.regret ?? 0)),
          validRate: valid.length / rs.length,
          medRegret: median(valid.map((r) => r.regret ?? 0)),
        };
      })
      .sort((a, b) => b.validRate - a.validRate || (a.medRegret ?? 9) - (b.medRegret ?? 9));
  }, [runs]);

  return (
    <>
      {groups.map((g) => {
        const off = hiddenConfigs.has(g.cid);
        return (
          <div key={g.cid} className={`group ${off ? "off" : ""}`}>
            <label className="group-head">
              <input type="checkbox" checked={!off} onChange={() => setHiddenConfigs(toggled(hiddenConfigs, g.cid))} />
              <span className="cfg">{configLabel(configs.get(g.cid))}</span>
              <span className="num" title="valid runs · median regret of valid runs">
                {fmtPct(g.validRate)}{g.medRegret != null && ` · ${fmtRegret(g.medRegret)}`}
              </span>
            </label>
            <div className="tokens">
              {g.runs.map((r) => (
                <button key={r.run_id} className={r.run_id === selectedRunId ? "sel" : ""}
                  onClick={() => onSelect(r.run_id === selectedRunId ? null : r.run_id)} title={r.run_id}>
                  <Swatch run={r} />
                  {r.valid ? fmtRegret(r.regret).replace("optimal", "0%") : "✕"}
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </>
  );
}

export function RunDetail({ run, config, optimal, onClose }: {
  run: Run; config?: Config; optimal: number | null; onClose: () => void;
}) {
  return (
    <section className="block">
      <h2 className="label">
        {configLabel(config)}
        <button className="link" onClick={onClose}>Close</button>
      </h2>
      <dl className="kv">
        <dt>Result</dt>
        <dd>{run.valid ? fmtRegret(run.regret) : failureLabel(run.failure)}</dd>
        <dt>Cost</dt>
        <dd>{run.agent_value?.toFixed(1) ?? "—"}{optimal != null && ` / ${optimal.toFixed(1)} optimal`}</dd>
        <dt>Tokens</dt>
        <dd>{run.tokens_in.toLocaleString()} in · {run.tokens_out.toLocaleString()} out</dd>
        <dt>Time</dt>
        <dd>{(run.wall_ms / 1000).toFixed(1)} s</dd>
      </dl>
      {run.raw_answer && (
        <details className="raw">
          <summary>Model output</summary>
          <pre>{run.raw_answer}</pre>
        </details>
      )}
    </section>
  );
}
