import {
  ColumnLayer,
  PathLayer,
  ScatterplotLayer,
  TerrainLayer,
  TextLayer,
} from "deck.gl";
import { PathStyleExtension } from "@deck.gl/extensions";
import { TerrainLoader } from "@loaders.gl/terrain";
import type { Layer } from "deck.gl";
import { cleanPath, type BBox, type Cell, type Run, type Task, type TaskGrid } from "./api";
import {
  bboxWidthMeters,
  cellToLonLat,
  sampleElevation,
  TERRARIUM_OFFSET,
  type Elevation,
} from "./terrain";

export interface Scene {
  task: Task;
  grid: TaskGrid;
  runs: Run[];
  elevation: Elevation;
  terrainUrl: string;
  bbox: BBox;
}

export interface LiveOverlay {
  draft: Cell[] | null;
  legs: Cell[][];
  pending: [Cell, Cell] | null;
}

export const DRAFT_COLOR: RGBA = [29, 27, 23, 255];
export const LEG_COLOR: RGBA = [31, 79, 209, 255];

export interface ViewOptions {
  overlay?: LiveOverlay | null;
  exaggeration: number;
  textureUrl: string | null;
  visibleRunIds: Set<string>;
  selectedRunId: string | null;
  showOptimal: boolean;
  showBlocked: boolean;
}

type RGBA = [number, number, number, number];

export interface PathDatum {
  id: string;
  path: [number, number, number][];
  color: RGBA;
  width: number;
  run?: Run;
  optimal?: boolean;
}

export const OPTIMAL_COLOR: RGBA = [255, 253, 246, 255];
export const INVALID_COLOR: RGBA = [155, 44, 111, 235];
export const INK: RGBA = [29, 27, 23, 255];

export const REGRET_STOPS: [number, [number, number, number]][] = [
  [0, [47, 111, 58]],
  [0.25, [120, 150, 40]],
  [0.6, [201, 162, 39]],
  [1.2, [196, 71, 29]],
  [2.5, [139, 26, 26]],
];

export function regretColor(regret: number | null, alpha = 235): RGBA {
  const r = Math.max(0, regret ?? 0);
  const stops = REGRET_STOPS;
  for (let i = 1; i < stops.length; i++) {
    if (r <= stops[i][0]) {
      const [t0, a] = stops[i - 1];
      const [t1, b] = stops[i];
      const f = (r - t0) / (t1 - t0);
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, alpha];
    }
  }
  return [...stops[stops.length - 1][1], alpha];
}

export function runColor(run: Run, alpha = 235): RGBA {
  return run.valid ? regretColor(run.regret, alpha) : [...INVALID_COLOR.slice(0, 3), alpha] as RGBA;
}

export function sceneMetrics(scene: Scene, exaggeration: number) {
  const widthM = bboxWidthMeters(scene.bbox);
  const relief = Math.max(1, scene.elevation.max - scene.elevation.min);
  const meshMaxError = Math.max(0.5, relief * exaggeration * 0.002);
  const lift = Math.max(3 * meshMaxError, widthM * 0.0025);
  const cellM = widthM / scene.grid.grid_size;
  return { widthM, relief, meshMaxError, lift, cellM };
}

export function defaultExaggeration(scene: Scene): number {
  if (scene.grid.frame.kind === "dem") return 1;
  const { widthM, relief } = sceneMetrics(scene, 1);
  return Math.round(Math.min(3, Math.max(0.05, (0.2 * widthM) / relief)) * 100) / 100;
}

function makeZ(scene: Scene, k: number, lift: number) {
  const { elevation, bbox } = scene;
  return (lon: number, lat: number) =>
    (sampleElevation(elevation, bbox, lon, lat) - elevation.min) * k + lift;
}

function drapePath(
  cells: Cell[],
  scene: Scene,
  z: (lon: number, lat: number) => number,
  steps = 8,
): [number, number, number][] {
  const n = scene.grid.grid_size;
  const out: [number, number, number][] = [];
  const push = (r: number, c: number) => {
    const [lon, lat] = cellToLonLat([r, c], n, scene.bbox);
    out.push([lon, lat, z(lon, lat)]);
  };
  cells.forEach((cell, i) => {
    if (i === 0) return push(cell[0], cell[1]);
    const prev = cells[i - 1];
    for (let s = 1; s <= steps; s++) {
      const t = s / steps;
      push(prev[0] + (cell[0] - prev[0]) * t, prev[1] + (cell[1] - prev[1]) * t);
    }
  });
  return out;
}

export function buildLayers(scene: Scene, o: ViewOptions): Layer[] {
  const k = o.exaggeration;
  const { meshMaxError, lift, cellM, relief } = sceneMetrics(scene, k);
  const z = makeZ(scene, k, lift);
  const zGround = makeZ(scene, k, 0);
  const n = scene.grid.grid_size;
  const base = scene.elevation.min;

  const terrain = new TerrainLayer({
    id: `terrain-${scene.task.task_id}`,
    elevationData: scene.terrainUrl,
    bounds: scene.bbox,
    texture: o.textureUrl,
    elevationDecoder: {
      rScaler: 256 * k,
      gScaler: k,
      bScaler: k / 256,
      offset: (-TERRARIUM_OFFSET - base) * k,
    },
    meshMaxError,
    material: false,
    color: [180, 180, 180],
    // main-thread loader: the default worker is fetched from a CDN
    loaders: [TerrainLoader],
    tesselator: "martini",
    loadOptions: { worker: false, terrain: { skirtHeight: 0 } },
  });

  const selected = o.selectedRunId;
  const runPaths: PathDatum[] = [];
  const invalidPaths: PathDatum[] = [];
  for (const run of scene.runs) {
    if (!o.visibleRunIds.has(run.run_id)) continue;
    const cells = cleanPath(run.predicted_path);
    if (cells.length < 2) continue;
    const isSel = run.run_id === selected;
    const dim = selected != null && !isSel;
    const d: PathDatum = {
      id: run.run_id,
      path: drapePath(cells, scene, z),
      color: runColor(run, dim ? 70 : 235),
      width: isSel ? 7 : 3,
      run,
    };
    (run.valid ? runPaths : invalidPaths).push(d);
  }
  const bySel = (a: PathDatum, b: PathDatum) => Number(a.id === selected) - Number(b.id === selected);
  runPaths.sort(bySel);
  invalidPaths.sort(bySel);

  const gt = scene.task.ground_truth.optimal_solution;
  const optimal: PathDatum[] =
    o.showOptimal && gt && gt.length > 1
      ? [{ id: "optimal", path: drapePath(gt, scene, z), color: OPTIMAL_COLOR, width: 5, optimal: true }]
      : [];

  const pathProps = {
    getPath: (d: PathDatum) => d.path,
    getColor: (d: PathDatum) => d.color,
    getWidth: (d: PathDatum) => d.width,
    widthUnits: "pixels" as const,
    capRounded: true,
    jointRounded: true,
    pickable: true,
    autoHighlight: true,
    highlightColor: [194, 65, 12, 160] as RGBA,
    updateTriggers: { getColor: [selected], getWidth: [selected] },
    parameters: { depthCompare: "less-equal" as const },
  };

  const layers: Layer[] = [terrain];

  if (o.showBlocked && scene.grid.blocked.length) {
    layers.push(
      new ColumnLayer<Cell>({
        id: "blocked",
        data: scene.grid.blocked,
        diskResolution: 4,
        angle: 45,
        radius: cellM * 0.45 * Math.SQRT2,
        extruded: true,
        getPosition: (c) => {
          const [lon, lat] = cellToLonLat(c, n, scene.bbox);
          return [lon, lat, zGround(lon, lat)];
        },
        getElevation: Math.max(cellM * 0.15, relief * k * 0.02),
        getFillColor: [29, 27, 23, 235],
        getLineColor: [29, 27, 23, 255],
        stroked: true,
        lineWidthUnits: "pixels",
        getLineWidth: 1,
        pickable: true,
        updateTriggers: { getPosition: [k] },
      }),
    );
  }

  layers.push(
    new PathLayer<PathDatum>({
      id: "optimal-casing",
      data: optimal,
      ...pathProps,
      pickable: false,
      getColor: INK,
      getWidth: 9,
    }),
    new PathLayer<PathDatum>({ id: "optimal", data: optimal, ...pathProps }),
    new PathLayer<PathDatum>({ id: "runs-valid", data: runPaths, ...pathProps }),
    new PathLayer<PathDatum>({
      id: "runs-invalid",
      data: invalidPaths,
      ...pathProps,
      getDashArray: [2.5, 2],
      extensions: [new PathStyleExtension({ dash: true })],
    } as never),
  );

  const ov = o.overlay;
  if (ov) {
    const legs: PathDatum[] = ov.legs
      .filter((l) => l.length > 1)
      .map((l, i) => ({ id: `leg-${i}`, path: drapePath(l, scene, z), color: LEG_COLOR, width: 5 }));
    const draft: PathDatum[] =
      ov.draft && ov.draft.length > 1
        ? [{ id: "draft", path: drapePath(ov.draft, scene, z), color: DRAFT_COLOR, width: 4 }]
        : [];
    const pending: PathDatum[] = ov.pending
      ? [{ id: "pending", path: drapePath([ov.pending[0], ov.pending[1]], scene, z, 24), color: [...LEG_COLOR.slice(0, 3), 170] as RGBA, width: 2 }]
      : [];
    const livePathProps = { ...pathProps, pickable: false, autoHighlight: false };
    layers.push(
      new PathLayer<PathDatum>({ id: "live-legs-casing", data: legs, ...livePathProps, getColor: [246, 241, 228, 255], getWidth: 8 }),
      new PathLayer<PathDatum>({ id: "live-legs", data: legs, ...livePathProps }),
      new PathLayer<PathDatum>({
        id: "live-draft",
        data: draft,
        ...livePathProps,
        getDashArray: [2, 1.5],
        extensions: [new PathStyleExtension({ dash: true })],
      } as never),
      new PathLayer<PathDatum>({
        id: "live-pending",
        data: pending,
        ...livePathProps,
        getDashArray: [1, 2],
        extensions: [new PathStyleExtension({ dash: true })],
      } as never),
      new ScatterplotLayer<Cell>({
        id: "live-waypoints",
        data: ov.pending ? [ov.pending[0], ov.pending[1]] : [],
        getPosition: (c) => {
          const [lon, lat] = cellToLonLat(c, n, scene.bbox);
          return [lon, lat, z(lon, lat) + lift];
        },
        getFillColor: [246, 241, 228, 255],
        getLineColor: LEG_COLOR,
        stroked: true,
        lineWidthUnits: "pixels",
        getLineWidth: 2,
        radiusUnits: "pixels",
        getRadius: 5,
        billboard: true,
        parameters: { depthCompare: "always" },
      }),
    );
  }

  const ends: { cell: Cell; label: string; color: RGBA }[] = [];
  if (scene.task.start) ends.push({ cell: scene.task.start, label: "START", color: [246, 241, 228, 255] });
  if (scene.task.goal) ends.push({ cell: scene.task.goal, label: "GOAL", color: [194, 65, 12, 255] });
  const endPos = (d: { cell: Cell }) => {
    const [lon, lat] = cellToLonLat(d.cell, n, scene.bbox);
    return [lon, lat, z(lon, lat) + lift] as [number, number, number];
  };
  layers.push(
    new ScatterplotLayer({
      id: "endpoints",
      data: ends,
      getPosition: endPos,
      getFillColor: (d) => d.color,
      getLineColor: INK,
      stroked: true,
      lineWidthUnits: "pixels",
      getLineWidth: 2.5,
      radiusUnits: "pixels",
      getRadius: 7,
      billboard: true,
      parameters: { depthCompare: "always" },
      updateTriggers: { getPosition: [k] },
    }),
    new TextLayer({
      id: "endpoint-labels",
      data: ends,
      getPosition: endPos,
      getText: (d) => d.label,
      getSize: 14,
      getColor: INK,
      getPixelOffset: [0, -18],
      fontFamily: '"Barlow Condensed", "Arial Narrow", sans-serif',
      fontWeight: 600,
      outlineWidth: 4,
      outlineColor: [246, 241, 228, 255],
      fontSettings: { sdf: true },
      billboard: true,
      parameters: { depthCompare: "always" },
      updateTriggers: { getPosition: [k] },
    }),
  );

  return layers;
}
