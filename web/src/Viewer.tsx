import { useEffect, useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import type { MapViewState, PickingInfo } from "deck.gl";
import { cleanPath, configLabel, type Config } from "./api";
import { failureLabel, fmtRegret } from "./components";
import { buildLayers, defaultExaggeration, REGRET_STOPS, type LiveOverlay, type PathDatum, type Scene } from "./layers";
import { fitView } from "./scene";
import { loadImage, paintTexture, satelliteUrl, type SurfaceMode } from "./terrain";

type Surface = SurfaceMode | "satellite";

const tip = (text: string) => ({
  text,
  style: {
    background: "#f6f1e4", color: "#1d1b17", border: "1px solid #1d1b17", borderRadius: "0",
    padding: "6px 8px", fontSize: "12px", lineHeight: "1.45", whiteSpace: "pre",
    fontFamily: '"JetBrains Mono", ui-monospace, monospace',
  },
});

export function Viewer(props: {
  scene: Scene | null;
  configs: Map<string, Config>;
  visibleRunIds: Set<string>;
  selectedRunId: string | null;
  onSelectRun: (id: string | null) => void;
  overlay?: LiveOverlay | null;
  onError: (msg: string) => void;
  loading?: string | null;
}) {
  const { scene, configs, visibleRunIds, selectedRunId, onSelectRun, overlay, onError } = props;
  const [viewState, setViewState] = useState<MapViewState>({ longitude: 0, latitude: 20, zoom: 1.5, pitch: 0, bearing: 0 });
  const [exaggeration, setExaggeration] = useState(1);
  const [surface, setSurface] = useState<Surface>("relief");
  const [showOptimal, setShowOptimal] = useState(true);
  const [showBlocked, setShowBlocked] = useState(true);

  const taskId = scene?.task.task_id;
  useEffect(() => {
    if (!scene) return;
    setExaggeration(defaultExaggeration(scene));
    if (scene.grid.frame.kind !== "dem") setSurface((m) => (m === "satellite" ? "relief" : m));
    setViewState(fitView(scene));
  }, [taskId]);

  const textureUrl = useMemo(() => {
    if (!scene) return null;
    if (surface === "satellite") return satelliteUrl(scene.bbox, scene.elevation.width, scene.elevation.height);
    return paintTexture(surface, scene.elevation, scene.bbox, scene.grid);
  }, [scene?.task.task_id, scene?.elevation, surface]);

  useEffect(() => {
    if (surface !== "satellite" || !textureUrl) return;
    let cancelled = false;
    loadImage(textureUrl).catch(() => {
      if (cancelled) return;
      onError("Satellite imagery couldn't load (it needs internet access). Showing relief instead.");
      setSurface("relief");
    });
    return () => {
      cancelled = true;
    };
  }, [surface, textureUrl, onError]);

  const layers = useMemo(
    () =>
      scene
        ? buildLayers(scene, { exaggeration, textureUrl, visibleRunIds, selectedRunId, showOptimal, showBlocked, overlay })
        : [],
    [scene, exaggeration, textureUrl, visibleRunIds, selectedRunId, showOptimal, showBlocked, overlay],
  );

  const getTooltip = ({ object, layer }: PickingInfo) => {
    if (!object) return null;
    if (layer?.id === "blocked") return tip("impassable");
    const d = object as PathDatum;
    if (d.optimal) return tip(`optimal · cost ${scene?.task.ground_truth.optimal_value?.toFixed(1)}`);
    if (!d.run) return null;
    const r = d.run;
    const head = r.valid ? fmtRegret(r.regret) : failureLabel(r.failure);
    return tip(`${configLabel(configs.get(r.config_id))}\n${head} · ${cleanPath(r.predicted_path).length} cells`);
  };

  return (
    <>
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: v }) => setViewState(v as MapViewState)}
        controller={{ dragRotate: true, touchRotate: true, inertia: 250 }}
        layers={layers}
        getTooltip={getTooltip}
        onClick={({ object }) => {
          const d = object as PathDatum | undefined;
          onSelectRun(d?.run ? (d.run.run_id === selectedRunId ? null : d.run.run_id) : null);
        }}
        getCursor={({ isHovering, isDragging }) => (isDragging ? "grabbing" : isHovering ? "pointer" : "grab")}
      />

      <div className="toolbar">
        <div className="seg" role="group" aria-label="Surface">
          {(["relief", "cost", "satellite"] as Surface[]).map((m) => (
            <button
              key={m}
              className={surface === m ? "on" : ""}
              disabled={m === "satellite" && scene?.grid.frame.kind !== "dem"}
              onClick={() => setSurface(m)}
              title={m === "cost" ? "The cost grid the agent was given" : m === "satellite" ? "Needs internet" : "Elevation with contours"}
            >
              {m === "cost" ? "Cost" : m === "relief" ? "Topo" : "Sat"}
            </button>
          ))}
        </div>
        <label title="Vertical exaggeration">
          <span className="muted">Z</span>
          <input type="range" min={0.05} max={4} step={0.05} value={exaggeration}
            onChange={(e) => setExaggeration(Number(e.target.value))} />
          <span className="num">{exaggeration.toFixed(1)}×</span>
        </label>
        <label className="check"><input type="checkbox" checked={showOptimal} onChange={(e) => setShowOptimal(e.target.checked)} />Optimal</label>
        <label className="check"><input type="checkbox" checked={showBlocked} onChange={(e) => setShowBlocked(e.target.checked)} />Blocked</label>
        {scene && <button className="icon" onClick={() => setViewState(fitView(scene))} title="Reset camera">Reset</button>}
      </div>

      <div className="legend">
        <div><i className="swatch" style={{ background: "#fffdf6", outline: "1.5px solid #1d1b17" }} /> Optimal</div>
        <div className="ramp">
          <span className="num">0%</span>
          <i style={{ background: `linear-gradient(90deg, ${REGRET_STOPS.map(([, c]) => `rgb(${c.join(",")})`).join(", ")})` }} />
          <span className="num">+250%</span>
        </div>
        <div><i className="swatch dashed" style={{ background: "rgb(155,44,111)" }} /> Invalid</div>
        {overlay && (
          <>
            <div><i className="swatch dashed" style={{ background: "#1d1b17" }} /> Draft</div>
            <div><i className="swatch" style={{ background: "rgb(31,79,209)" }} /> Solver legs</div>
          </>
        )}
      </div>

      {props.loading && <div className="loading">{props.loading}</div>}
    </>
  );
}
