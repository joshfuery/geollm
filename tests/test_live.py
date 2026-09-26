from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from geoagentbench.api import app as api_mod  # noqa: E402
from geoagentbench.core import BBox, RouteParams  # noqa: E402
from geoagentbench.runner.config import AgentConfig, RunConfig, ToolAccess, Topology  # noqa: E402
from geoagentbench.runner.executor import execute_one  # noqa: E402
from geoagentbench.runner.observe import ComposerResponder, ObservedProvider  # noqa: E402
from geoagentbench.runner.providers import MockProvider  # noqa: E402
from geoagentbench.solvers.route_solver import solve_and_attach  # noqa: E402
from geoagentbench.tasks.route import generate_synthetic_route_task  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setattr(api_mod, "RESULTS_DIR", tmp_path / "main")
    return TestClient(api_mod.app)


def _events(client: TestClient, live_id: str) -> list[dict]:
    with client.stream("GET", f"/api/live/runs/{live_id}/events") as s:
        return [json.loads(line[6:]) for line in s.iter_lines() if line.startswith("data: ")]


def test_composer_through_observed_provider_streams_legs_and_scores():
    task, cost, _ = generate_synthetic_route_task(3, RouteParams(grid_size=16))
    task = solve_and_attach(task, cost)
    events: list[dict] = []
    provider = ObservedProvider(MockProvider(responder=ComposerResponder(max_leg=4), latency_ms=0), events.append)
    cfg = RunConfig(agent=AgentConfig(model="mock-composer", provider="mock:composer",
                                      topology=Topology.TOOL_LOOP, tool_access=ToolAccess.RESTRICTED_SOLVER,
                                      max_segment_cells=5))
    out = execute_one(task, cost, provider, cfg)

    assert out.record.score.valid, out.record.score
    legs = [e for e in events if e["type"] == "tool" and e["name"] == "solve_between" and e["ok"]]
    assert legs and all(len(e["cells"]) <= 5 for e in legs)
    assert any(e["type"] == "tool" and e["name"] == "check_path" for e in events)
    final = [e for e in events if e["type"] == "llm"][-1]
    assert final["draft"][0] == task.params["start"] and final["draft"][-1] == task.params["goal"]


def test_live_synthetic_task_and_mock_run_end_to_end(client):
    t = client.post("/api/live/tasks", json={"source": "synthetic", "grid_size": 12}).json()
    assert t["ground_truth"]["optimal_solution"][0] == t["start"]

    grid = client.get(f"/api/tasks/{t['task_id']}/grid?store=live").json()
    assert grid["grid_size"] == 12 and grid["frame"]["terrain_url"] is None
    assert client.get(f"/api/tasks/{t['task_id']}/grid").status_code == 404

    r = client.post("/api/live/runs", json={
        "task_id": t["task_id"], "provider": "mock:oracle", "topology": "single_shot"}).json()
    evs = _events(client, r["live_id"])
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "status" and kinds[-1] == "done" and "llm" in kinds
    result = next(e for e in evs if e["type"] == "result")["run"]
    assert result["valid"] and result["regret"] == pytest.approx(0.0)

    runs = client.get(f"/api/tasks/{t['task_id']}/runs?store=live").json()
    assert [x["run_id"] for x in runs] == [result["run_id"]]


def test_live_run_rejects_bad_requests(client):
    t = client.post("/api/live/tasks", json={"source": "synthetic", "grid_size": 10}).json()
    bad_topology = client.post("/api/live/runs", json={
        "task_id": t["task_id"], "provider": "mock:composer", "topology": "single_shot"})
    assert bad_topology.status_code == 400
    assert client.post("/api/live/runs", json={"task_id": "nope", "provider": "mock:oracle"}).status_code == 404


def _write_dem(path: Path, west=-119.7, south=37.7, east=-119.6, north=37.78, n=120) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    yy, xx = np.mgrid[0:n, 0:n]
    z = 1500 + 400 * np.sin(xx / 15.0) * np.cos(yy / 20.0) + 3 * xx
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_bounds(west, south, east, north, n, n)) as dst:
        dst.write(z.astype("float32"), 1)


def test_region_window_task_regenerates_and_serves_terrain(client, tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_dem(data / "yosemite" / "dem.tif")
    monkeypatch.setattr(api_mod, "DATA_DIR", data)

    t = client.post("/api/live/tasks", json={"source": "region", "region": "yosemite", "grid_size": 16}).json()
    w = t["params"]["window"]
    assert -119.7 <= w[0] < w[2] <= -119.6 and 37.7 <= w[1] < w[3] <= 37.78

    grid = client.get(f"/api/tasks/{t['task_id']}/grid?store=live").json()
    assert grid["frame"]["bbox"] == w
    png = client.get(grid["frame"]["terrain_url"])
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"


def test_anywhere_fetches_once_and_resolves_region(client, tmp_path, monkeypatch):
    import geoagentbench.places as places

    data = tmp_path / "data"
    monkeypatch.setattr(api_mod, "DATA_DIR", data)
    monkeypatch.setenv("OPENTOPO_API_KEY", "test-key")
    calls: list[dict] = []

    class FakeResp:
        status_code = 200

        def __init__(self, content: bytes):
            self.content = content
            self.text = ""

        def raise_for_status(self):
            pass

    def fake_get(url, params, timeout):
        calls.append(params)
        tif = tmp_path / "fake.tif"
        _write_dem(tif, params["west"], params["south"], params["east"], params["north"], n=90)
        return FakeResp(tif.read_bytes())

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    spec = places.fetch_random_place(data, rng=random.Random(1))
    assert spec.name.startswith("places/") and (data / spec.name / "dem.tif").exists()
    places.fetch_random_place(data, rng=random.Random(1))
    assert len(calls) == 1 and calls[0]["API_Key"] == "test-key"

    t = client.post("/api/live/tasks", json={"source": "anywhere", "grid_size": 12}).json()
    grid = client.get(f"/api/tasks/{t['task_id']}/grid?store=live").json()
    assert grid["frame"]["region"].startswith("places/")
    assert grid["frame"]["region_label"].startswith("Near ")


def test_anywhere_without_key_is_a_clear_400(client, monkeypatch):
    monkeypatch.delenv("OPENTOPO_API_KEY", raising=False)
    r = client.post("/api/live/tasks", json={"source": "anywhere"})
    assert r.status_code == 400 and "OPENTOPO_API_KEY" in r.json()["detail"]


def test_window_param_keeps_whole_region_ids_unchanged(tmp_path):
    from geoagentbench.tasks.route import generate_route_from_dem

    _write_dem(tmp_path / "yosemite" / "dem.tif")
    whole, _, _ = generate_route_from_dem("yosemite", 5, data_dir=tmp_path, grid_size=12)
    assert "window" not in whole.params
    win = BBox(min_x=-119.68, min_y=37.71, max_x=-119.62, max_y=37.76)
    a, _, _ = generate_route_from_dem("yosemite", 5, data_dir=tmp_path, grid_size=12, window=win)
    b, _, _ = generate_route_from_dem("yosemite", 5, data_dir=tmp_path, grid_size=12, window=win)
    assert a.task_id == b.task_id != whole.task_id
