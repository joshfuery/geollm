from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..core import NodeTrace, RunRecord, TaskInstance
from .config import AgentConfig


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id           TEXT PRIMARY KEY,
    family            TEXT NOT NULL,
    seed              INTEGER NOT NULL,
    params_json       TEXT NOT NULL,
    ground_truth_json TEXT NOT NULL,
    region_json       TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS configs (
    config_id      TEXT PRIMARY KEY,
    model          TEXT NOT NULL,
    temperature    REAL NOT NULL,
    prompt_version TEXT NOT NULL,
    condition      TEXT NOT NULL,
    topology       TEXT NOT NULL,
    tool_access    TEXT NOT NULL,
    payload_json   TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL REFERENCES tasks(task_id),
    config_id    TEXT NOT NULL REFERENCES configs(config_id),
    valid        INTEGER NOT NULL,
    regret       REAL,
    agent_value  REAL,
    failure      TEXT NOT NULL,
    notes        TEXT,
    tokens_in    INTEGER NOT NULL DEFAULT 0,
    tokens_out   INTEGER NOT NULL DEFAULT 0,
    wall_ms      INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    answer_json  TEXT NOT NULL,
    score_json   TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_task_config ON runs(task_id, config_id);
CREATE INDEX IF NOT EXISTS idx_runs_config      ON runs(config_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Store:
    db_path: Path
    traces_dir: Path

    @classmethod
    def open(cls, results_dir: Path | str = "results") -> "Store":
        rd = Path(results_dir)
        db = rd / "geoab.db"
        tr = rd / "traces"
        rd.mkdir(parents=True, exist_ok=True)
        tr.mkdir(parents=True, exist_ok=True)
        store = cls(db_path=db, traces_dir=tr)
        with store._conn() as c:
            c.executescript(SCHEMA)
        return store

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


    def upsert_task(self, task: TaskInstance) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO tasks "
                "(task_id, family, seed, params_json, ground_truth_json, region_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task.task_id,
                    task.family.value,
                    task.seed,
                    json.dumps(task.params, default=str),
                    json.dumps(task.ground_truth.model_dump(), default=str),
                    json.dumps(task.region.model_dump()),
                    _now(),
                ),
            )

    def upsert_config(self, cfg: AgentConfig) -> str:
        cid = cfg.config_id
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO configs "
                "(config_id, model, temperature, prompt_version, condition, topology, tool_access, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    cfg.model,
                    cfg.temperature,
                    cfg.prompt_version,
                    cfg.condition.value,
                    cfg.topology.value,
                    cfg.tool_access.value,
                    json.dumps(cfg.model_dump(mode="json")),
                    _now(),
                ),
            )
        return cid

    def write_run(self, run: RunRecord, traces: list[NodeTrace]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, task_id, config_id, valid, regret, agent_value, failure, notes, "
                " tokens_in, tokens_out, wall_ms, error, answer_json, score_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.task_id,
                    run.config_id,
                    int(run.score.valid),
                    run.score.regret,
                    run.score.agent_value,
                    run.score.failure.value,
                    run.score.notes,
                    run.tokens_in,
                    run.tokens_out,
                    run.wall_ms,
                    run.error,
                    json.dumps(run.answer.model_dump(), default=str),
                    json.dumps(run.score.model_dump()),
                    _now(),
                ),
            )
        trace_path = self.traces_dir / f"{run.run_id}.jsonl"
        with trace_path.open("w") as f:
            for t in traces:
                f.write(json.dumps(t.model_dump()) + "\n")


    def list_runs(
        self,
        config_id: str | None = None,
        task_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM runs"
        clauses, args = [], []
        if config_id:
            clauses.append("config_id = ?"); args.append(config_id)
        if task_id:
            clauses.append("task_id = ?"); args.append(task_id)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def read_traces(self, run_id: str) -> list[dict[str, Any]]:
        p = self.traces_dir / f"{run_id}.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def list_tasks(self, family: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as c:
            if family:
                rows = c.execute(
                    "SELECT * FROM tasks WHERE family = ? ORDER BY created_at DESC LIMIT ?",
                    (family, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    def list_configs(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM configs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
