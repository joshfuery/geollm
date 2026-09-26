from __future__ import annotations

import math

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from geoagentbench.api import app as api_mod  # noqa: E402
from geoagentbench.core import RouteParams  # noqa: E402
from geoagentbench.runner.store import Store  # noqa: E402
from geoagentbench.tasks.route import generate_synthetic_route_task  # noqa: E402


def test_grid_endpoint_regenerates_synthetic_task(tmp_path, monkeypatch):
    task, cost, elev = generate_synthetic_route_task(
        seed=7, params=RouteParams(grid_size=16, impassable_frac=0.1)
    )
    Store.open(tmp_path).upsert_task(task)
    monkeypatch.setattr(api_mod, "RESULTS_DIR", tmp_path)

    res = TestClient(api_mod.app).get(f"/api/tasks/{task.task_id}/grid")
    assert res.status_code == 200
    body = res.json()

    assert body["grid_size"] == 16
    assert body["frame"]["kind"] == "synthetic"
    w, s, e, n = body["frame"]["bbox"]
    assert w < 0 < e and s < 0 < n

    blocked = {tuple(c) for c in body["blocked"]}
    assert blocked == {(r, c) for r in range(16) for c in range(16) if math.isinf(cost[r, c])}
    assert all(body["cost"][r][c] is None for r, c in blocked)
    assert body["elev_m"][3][5] == pytest.approx(elev[3, 5], abs=0.01)


def test_grid_endpoint_404s_unknown_task(tmp_path, monkeypatch):
    Store.open(tmp_path)
    monkeypatch.setattr(api_mod, "RESULTS_DIR", tmp_path)
    assert TestClient(api_mod.app).get("/api/tasks/nope/grid").status_code == 404
