import { WebMercatorViewport, type MapViewState } from "deck.gl";
import { api, type StoreName, type Task } from "./api";
import type { Scene } from "./layers";
import { decodeTerrarium, elevationFromGrid, loadImage, toTerrainMeshImage } from "./terrain";

export const SIDEBAR_W = 340;

export async function loadScene(task: Task, store: StoreName): Promise<Scene> {
  const [grid, runs] = await Promise.all([api.taskGrid(task.task_id, store), api.taskRuns(task.task_id, store)]);
  const elevation = grid.frame.terrain_url
    ? decodeTerrarium(await loadImage(grid.frame.terrain_url))
    : elevationFromGrid(grid.elev_m);
  return { task, grid, runs, elevation, terrainUrl: toTerrainMeshImage(elevation), bbox: grid.frame.bbox };
}

export function fitView(scene: Scene): MapViewState {
  const [w, s, e, n] = scene.bbox;
  const wide = window.innerWidth > 800;
  const width = wide ? window.innerWidth - SIDEBAR_W : window.innerWidth;
  const height = wide ? window.innerHeight : Math.round(window.innerHeight * 0.55);
  const vp = new WebMercatorViewport({ width, height });
  const pad = Math.min(70, height / 8);
  const { longitude, latitude, zoom } = vp.fitBounds([[w, s], [e, n]], {
    padding: { left: pad, right: pad, top: pad + 30, bottom: pad },
  });
  return { longitude, latitude, zoom: zoom - 0.25, pitch: 55, bearing: -20, maxPitch: 85, maxZoom: 22 };
}
