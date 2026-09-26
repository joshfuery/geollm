import type { BBox, Cell, TaskGrid } from "./api";

export const TERRARIUM_OFFSET = 32768;

export interface Elevation {
  width: number;
  height: number;
  data: Float32Array;
  min: number;
  max: number;
}

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load image ${url}`));
    img.src = url;
  });
}

function canvas2d(w: number, h: number) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
  if (!ctx) throw new Error("2D canvas unavailable");
  return { c, ctx };
}

export function decodeTerrarium(img: HTMLImageElement): Elevation {
  const { ctx } = canvas2d(img.naturalWidth, img.naturalHeight);
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, img.naturalWidth, img.naturalHeight).data;
  const n = img.naturalWidth * img.naturalHeight;
  const data = new Float32Array(n);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const z = px[i * 4] * 256 + px[i * 4 + 1] + px[i * 4 + 2] / 256 - TERRARIUM_OFFSET;
    data[i] = z;
    if (z < min) min = z;
    if (z > max) max = z;
  }
  return { width: img.naturalWidth, height: img.naturalHeight, data, min, max };
}

export const nextPow2 = (x: number) => 2 ** Math.ceil(Math.log2(Math.max(2, x)));

function bilinearResize(
  get: (r: number, c: number) => number,
  srcW: number,
  srcH: number,
  outW: number,
  outH: number,
): Elevation {
  const data = new Float32Array(outW * outH);
  let min = Infinity;
  let max = -Infinity;
  const at = (r: number, c: number) =>
    get(Math.max(0, Math.min(srcH - 1, r)), Math.max(0, Math.min(srcW - 1, c)));
  for (let y = 0; y < outH; y++) {
    const v = ((y + 0.5) / outH) * srcH - 0.5;
    const r0 = Math.floor(v);
    const fy = v - r0;
    for (let x = 0; x < outW; x++) {
      const u = ((x + 0.5) / outW) * srcW - 0.5;
      const c0 = Math.floor(u);
      const fx = u - c0;
      const z =
        at(r0, c0) * (1 - fx) * (1 - fy) +
        at(r0, c0 + 1) * fx * (1 - fy) +
        at(r0 + 1, c0) * (1 - fx) * fy +
        at(r0 + 1, c0 + 1) * fx * fy;
      data[y * outW + x] = z;
      if (z < min) min = z;
      if (z > max) max = z;
    }
  }
  return { width: outW, height: outH, data, min, max };
}

export function elevationFromGrid(grid: number[][], pxPerCell = 16): Elevation {
  const n = grid.length;
  const size = Math.min(1024, nextPow2(n * pxPerCell));
  return bilinearResize((r, c) => grid[r][c], n, n, size, size);
}

// loaders.gl leaves z=0 padding on non-square images (walls on two edges); square 2^n avoids it.
export function toTerrainMeshImage(e: Elevation, maxSize = 1024): string {
  const size = Math.min(maxSize, nextPow2(Math.max(e.width, e.height)));
  const sq =
    e.width === size && e.height === size
      ? e
      : bilinearResize((r, c) => e.data[r * e.width + c], e.width, e.height, size, size);
  const { c, ctx } = canvas2d(size, size);
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < sq.data.length; i++) {
    const fixed = Math.round((sq.data[i] + TERRARIUM_OFFSET) * 256);
    img.data[i * 4] = (fixed >> 16) & 0xff;
    img.data[i * 4 + 1] = (fixed >> 8) & 0xff;
    img.data[i * 4 + 2] = fixed & 0xff;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL("image/png");
}

export function sampleElevation(e: Elevation, bbox: BBox, lon: number, lat: number): number {
  const [w, s, east, n] = bbox;
  const x = Math.max(0, Math.min(e.width - 1, ((lon - w) / (east - w)) * e.width - 0.5));
  const y = Math.max(0, Math.min(e.height - 1, ((n - lat) / (n - s)) * e.height - 0.5));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(e.width - 1, x0 + 1);
  const y1 = Math.min(e.height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const d = e.data;
  return (
    d[y0 * e.width + x0] * (1 - fx) * (1 - fy) +
    d[y0 * e.width + x1] * fx * (1 - fy) +
    d[y1 * e.width + x0] * (1 - fx) * fy +
    d[y1 * e.width + x1] * fx * fy
  );
}

export function cellToLonLat(cell: Cell, n: number, bbox: BBox): [number, number] {
  const [w, s, e, north] = bbox;
  return [w + ((cell[1] + 0.5) / n) * (e - w), north - ((cell[0] + 0.5) / n) * (north - s)];
}

export function bboxWidthMeters(bbox: BBox): number {
  const midLat = (bbox[1] + bbox[3]) / 2;
  return (bbox[2] - bbox[0]) * 111_320 * Math.cos((midLat * Math.PI) / 180);
}

type RGB = [number, number, number];
type Stop = [number, RGB];

function ramp(stops: Stop[], t: number): RGB {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, a] = stops[i - 1];
      const [t1, b] = stops[i];
      const f = (t - t0) / (t1 - t0 || 1);
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}

const RELIEF: Stop[] = [
  [0.0, [150, 172, 128]],
  [0.25, [188, 196, 146]],
  [0.5, [214, 200, 156]],
  [0.72, [196, 170, 132]],
  [0.88, [210, 196, 172]],
  [1.0, [232, 224, 206]],
];

export function contourInterval(range: number, target = 14): number {
  const raw = range / target;
  const mag = 10 ** Math.floor(Math.log10(Math.max(raw, 1e-6)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * mag >= raw) return m * mag;
  return 10 * mag;
}

const COST: Stop[] = [
  [0.0, [242, 236, 218]],
  [0.35, [226, 204, 150]],
  [0.6, [206, 150, 92]],
  [0.8, [160, 92, 60]],
  [1.0, [92, 50, 38]],
];
const BLOCKED: RGB = [29, 27, 23];

function hillshade(e: Elevation, bbox: BBox): Float32Array {
  const { width: w, height: h, data } = e;
  const dx = bboxWidthMeters(bbox) / w;
  const dy = (((bbox[3] - bbox[1]) * 111_320) / h) || dx;
  const L = [-0.5, 0.5, Math.SQRT1_2];
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const yu = Math.max(0, y - 1);
    const yd = Math.min(h - 1, y + 1);
    for (let x = 0; x < w; x++) {
      const xl = Math.max(0, x - 1);
      const xr = Math.min(w - 1, x + 1);
      const dzdx = (data[y * w + xr] - data[y * w + xl]) / ((xr - xl) * dx);
      const dzdy = -(data[yd * w + x] - data[yu * w + x]) / ((yd - yu) * dy);
      const norm = Math.hypot(dzdx, dzdy, 1);
      const v = (-dzdx * L[0] - dzdy * L[1] + L[2]) / norm;
      out[y * w + x] = Math.max(0, v);
    }
  }
  return out;
}

export type SurfaceMode = "relief" | "cost";

export function paintTexture(
  mode: SurfaceMode,
  e: Elevation,
  bbox: BBox,
  grid: TaskGrid,
): string {
  const n = grid.grid_size;
  const scale = mode === "cost" ? Math.max(1, Math.ceil((n * 12) / Math.min(e.width, e.height))) : 1;
  const W = e.width * scale;
  const H = e.height * scale;
  const shade = hillshade(e, bbox);
  const { c, ctx } = canvas2d(W, H);
  const img = ctx.createImageData(W, H);
  const range = e.max - e.min || 1;
  const [cMin, cMax] = grid.cost_range;
  const logMin = Math.log(cMin);
  const logSpan = Math.log(cMax) - logMin || 1;
  const interval = contourInterval(range);
  const band = (i: number) => Math.floor(e.data[i] / interval);

  for (let Y = 0; Y < H; Y++) {
    const y = Math.floor(Y / scale);
    const row = Math.min(n - 1, Math.floor((Y / H) * n));
    const rowEdge = Math.abs(((Y + 0.5) / H) * n - Math.round(((Y + 0.5) / H) * n)) * (H / n) < 0.6;
    for (let X = 0; X < W; X++) {
      const x = Math.floor(X / scale);
      const i = y * e.width + x;
      const sh = 0.62 + 0.45 * shade[i];
      let rgb: RGB;
      if (mode === "relief") {
        rgb = ramp(RELIEF, (e.data[i] - e.min) / range);
        const b = band(i);
        const bx = x + 1 < e.width ? band(i + 1) : b;
        const by = y + 1 < e.height ? band(i + e.width) : b;
        if (b !== bx || b !== by) {
          const index = Math.max(b, bx, by) % 5 === 0;
          const k = index ? 0.5 : 0.72;
          rgb = [rgb[0] * k, rgb[1] * k * 0.96, rgb[2] * k * 0.9];
        }
      } else {
        const col = Math.min(n - 1, Math.floor((X / W) * n));
        const v = grid.cost[row][col];
        rgb = v == null ? BLOCKED : ramp(COST, (Math.log(v) - logMin) / logSpan);
        const colEdge =
          Math.abs(((X + 0.5) / W) * n - Math.round(((X + 0.5) / W) * n)) * (W / n) < 0.6;
        if (rowEdge || colEdge) rgb = [rgb[0] * 0.55, rgb[1] * 0.55, rgb[2] * 0.55];
      }
      const o = (Y * W + X) * 4;
      img.data[o] = Math.min(255, rgb[0] * sh);
      img.data[o + 1] = Math.min(255, rgb[1] * sh);
      img.data[o + 2] = Math.min(255, rgb[2] * sh);
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL("image/png");
}

export function satelliteUrl(bbox: BBox, width: number, height: number): string {
  const scale = Math.min(2048 / width, 2048 / height, 4);
  const w = Math.round(width * scale);
  const h = Math.round(height * scale);
  const params = new URLSearchParams({
    bbox: bbox.join(","),
    bboxSR: "4326",
    imageSR: "4326",
    size: `${w},${h}`,
    format: "jpg",
    f: "image",
  });
  return `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?${params}`;
}
