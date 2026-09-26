export type Cell = [number, number];
export type BBox = [number, number, number, number];

export interface Task {
  task_id: string;
  family: string;
  seed: number;
  params: Record<string, unknown>;
  ground_truth: {
    optimal_value: number | null;
    optimal_solution: Cell[] | null;
    solver: string | null;
    compute_ms: number | null;
  };
  start: Cell | null;
  goal: Cell | null;
  grid_size: number | null;
  source: "dem" | "synthetic" | null;
  region_name: string | null;
  region: BBoxDict;
}

export interface BBoxDict {
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
}

export type StoreName = "main" | "live";

export interface LiveOptions {
  openai: boolean;
  opentopo: boolean;
  regions: { name: string; label: string }[];
  mock_modes: string[];
  topologies: string[];
  tool_access: string[];
  conditions: string[];
}

export interface NewTaskRequest {
  source: "region" | "anywhere" | "synthetic";
  region?: string | null;
  grid_size: number;
}

export interface NewRunRequest {
  task_id: string;
  provider: string;
  model: string;
  topology: string;
  tool_access: string;
  condition: string;
  temperature?: number;
  max_segment_cells?: number | null;
  max_revise_rounds?: number;
  max_tool_steps?: number;
}

export type LiveEvent = { seq: number; t_ms: number } & (
  | { type: "status"; message: string }
  | { type: "thinking"; step: number }
  | {
      type: "llm";
      step: number;
      text: string;
      truncated: boolean;
      tool_calls: { name: string; args: Record<string, unknown> }[];
      draft: Cell[] | null;
      tokens_in: number;
      tokens_out: number;
      wall_ms: number;
    }
  | {
      type: "tool";
      name: string;
      args: Record<string, unknown>;
      ok: boolean;
      cells?: Cell[] | null;
      cost?: number | null;
      error?: string | null;
      issues?: string[];
      n_issues?: number;
      draft?: Cell[] | null;
    }
  | { type: "infra_error"; step: number; message: string }
  | { type: "result"; run: Run }
  | { type: "error"; message: string }
  | { type: "done" }
);

export interface Run {
  run_id: string;
  task_id: string;
  config_id: string;
  valid: boolean;
  regret: number | null;
  agent_value: number | null;
  failure: string;
  notes: string | null;
  tokens_in: number;
  tokens_out: number;
  wall_ms: number;
  error: string | null;
  predicted_path: unknown;
  raw_answer: string | null;
  created_at: string;
}

export interface Config {
  config_id: string;
  model: string;
  temperature: number;
  prompt_version: string;
  condition: string;
  topology: string;
  tool_access: string;
  payload: Record<string, unknown>;
}

export interface TaskGrid {
  task_id: string;
  grid_size: number;
  frame: {
    kind: "dem" | "synthetic";
    region: string | null;
    region_label: string;
    bbox: BBox;
    pixel_size_m: number | null;
    terrain_url: string | null;
  };
  elev_m: number[][];
  cost: (number | null)[][];
  cost_range: [number, number];
  blocked: Cell[];
}

export interface Region {
  name: string;
  label: string;
  description: string;
  has_terrain: boolean;
}

export interface Health {
  ok: boolean;
  results_dir: string;
  counts: { tasks: number; configs: number; runs: number };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}
const get = <T,>(path: string) => request<T>(path);
const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  health: () => get<Health>("/api/health"),
  regions: () => get<Region[]>("/api/regions"),
  tasks: (store: StoreName = "main") => get<Task[]>(`/api/tasks?limit=1000&store=${store}`),
  configs: (store: StoreName = "main") => get<Config[]>(`/api/configs?store=${store}`),
  taskGrid: (id: string, store: StoreName = "main") => get<TaskGrid>(`/api/tasks/${id}/grid?store=${store}`),
  taskRuns: (id: string, store: StoreName = "main") => get<Run[]>(`/api/tasks/${id}/runs?limit=1000&store=${store}`),
  liveOptions: () => get<LiveOptions>("/api/live/options"),
  newLiveTask: (req: NewTaskRequest) => post<Task>("/api/live/tasks", req),
  startLiveRun: (req: NewRunRequest) => post<{ live_id: string; config_id: string }>("/api/live/runs", req),
  liveEventsUrl: (liveId: string) => `/api/live/runs/${liveId}/events`,
};

export function argPoint(args: Record<string, unknown>, prefix: "from" | "to"): Cell | null {
  const n = (v: unknown) => (typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN);
  const alt = prefix === "from" ? ["from", "start", "f"] : ["to", "end", "goal", "t"];
  for (const p of alt) {
    const r = n(args[`${p}_row`]);
    const c = n(args[`${p}_col`]);
    if (Number.isFinite(r) && Number.isFinite(c)) return [r, c];
    const v = args[p];
    if (Array.isArray(v) && v.length === 2 && Number.isFinite(n(v[0])) && Number.isFinite(n(v[1]))) return [n(v[0]), n(v[1])];
  }
  return null;
}

export function cleanPath(p: unknown): Cell[] {
  if (!Array.isArray(p)) return [];
  const out: Cell[] = [];
  for (const c of p) {
    if (Array.isArray(c) && c.length >= 2 && Number.isFinite(c[0]) && Number.isFinite(c[1])) {
      out.push([Number(c[0]), Number(c[1])]);
    }
  }
  return out;
}

const TOPOLOGY_SHORT: Record<string, string> = {
  single_shot: "single shot",
  plan_critique_revise: "self-critique",
  plan_verify_revise: "verify loop",
  tool_loop: "tool loop",
  plan_execute: "plan/execute",
};

export function configLabel(c: Config | undefined): string {
  if (!c) return "unknown config";
  const parts = [c.model, TOPOLOGY_SHORT[c.topology] ?? c.topology.replace(/_/g, " ")];
  if (c.topology === "plan_verify_revise") {
    const r = Number((c.payload as { max_revise_rounds?: number }).max_revise_rounds ?? 1);
    if (r > 1) parts[1] += ` ×${r}`;
  }
  if (c.tool_access === "calculator") parts.push("checker");
  if (c.tool_access === "restricted_solver") {
    const cap = (c.payload as { max_segment_cells?: number | null }).max_segment_cells;
    parts.push(`capped ${cap ?? 5}`);
  }
  if (c.tool_access === "full_solver") parts.push("full solver");
  if (c.condition === "narrative_only") parts.push("prose");
  if (c.condition === "both") parts.push("prose+grid");
  return parts.join(" · ");
}
