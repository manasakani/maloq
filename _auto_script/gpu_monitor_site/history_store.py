#!/usr/bin/env python3
"""Persistent SQLite history for the private SC26 GPU monitor."""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sampled_at TEXT NOT NULL,
    servers_online INTEGER NOT NULL,
    servers_total INTEGER NOT NULL,
    gpus_total INTEGER NOT NULL,
    gpus_idle INTEGER NOT NULL,
    gpus_busy INTEGER NOT NULL,
    gpus_warning INTEGER NOT NULL,
    gpus_offline INTEGER NOT NULL,
    gpu_memory_used_mib INTEGER NOT NULL,
    gpu_memory_total_mib INTEGER NOT NULL,
    power_draw_w REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_sampled_at
ON snapshots(sampled_at);

CREATE TABLE IF NOT EXISTS server_samples (
    snapshot_id INTEGER NOT NULL,
    server_id TEXT NOT NULL,
    hostname TEXT,
    online INTEGER NOT NULL,
    cached INTEGER NOT NULL,
    latency_ms INTEGER,
    uptime_seconds REAL,
    error TEXT,
    PRIMARY KEY (snapshot_id, server_id)
);

CREATE TABLE IF NOT EXISTS storage_samples (
    snapshot_id INTEGER NOT NULL,
    server_id TEXT NOT NULL,
    mountpoint TEXT NOT NULL,
    kind TEXT NOT NULL,
    filesystem_type TEXT,
    total_bytes INTEGER,
    used_bytes INTEGER,
    available_bytes INTEGER,
    used_percent REAL,
    policy_limit_bytes INTEGER,
    policy_remaining_bytes INTEGER,
    policy_used_percent REAL,
    policy_exceeded INTEGER,
    PRIMARY KEY (snapshot_id, server_id, mountpoint)
);

CREATE INDEX IF NOT EXISTS idx_storage_server_mount
ON storage_samples(server_id, mountpoint, snapshot_id);

CREATE TABLE IF NOT EXISTS gpu_samples (
    snapshot_id INTEGER NOT NULL,
    server_id TEXT NOT NULL,
    gpu_index INTEGER NOT NULL,
    gpu_uuid TEXT,
    state TEXT,
    utilization_percent REAL,
    memory_used_mib INTEGER,
    memory_total_mib INTEGER,
    temperature_c REAL,
    power_draw_w REAL,
    power_limit_w REAL,
    pstate TEXT,
    process_count INTEGER NOT NULL,
    PRIMARY KEY (snapshot_id, server_id, gpu_index)
);

CREATE INDEX IF NOT EXISTS idx_gpu_server_index
ON gpu_samples(server_id, gpu_index, snapshot_id);

CREATE TABLE IF NOT EXISTS process_samples (
    snapshot_id INTEGER NOT NULL,
    server_id TEXT NOT NULL,
    gpu_index INTEGER NOT NULL,
    pid INTEGER NOT NULL,
    user TEXT,
    command TEXT,
    process_name TEXT,
    elapsed TEXT,
    memory_used_mib INTEGER,
    PRIMARY KEY (snapshot_id, server_id, gpu_index, pid)
);

CREATE INDEX IF NOT EXISTS idx_process_user_pid
ON process_samples(user, pid, snapshot_id);
"""


class HistoryStore:
    """Append-only monitor history with one transaction per saved snapshot."""

    def __init__(self, path: Path, sample_interval_seconds: float) -> None:
        self.path = path
        self.sample_interval_seconds = sample_interval_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def record(
        self,
        sampled_at: str,
        servers: list[dict[str, object]],
        fleet: dict[str, object],
    ) -> dict[str, object]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO snapshots (
                    sampled_at, servers_online, servers_total, gpus_total,
                    gpus_idle, gpus_busy, gpus_warning, gpus_offline,
                    gpu_memory_used_mib, gpu_memory_total_mib, power_draw_w
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sampled_at,
                    int(fleet.get("servers_online") or 0),
                    int(fleet.get("servers_total") or 0),
                    int(fleet.get("gpus_total") or 0),
                    int(fleet.get("gpus_idle") or 0),
                    int(fleet.get("gpus_busy") or 0),
                    int(fleet.get("gpus_warning") or 0),
                    int(fleet.get("gpus_offline") or 0),
                    int(fleet.get("memory_used_mib") or 0),
                    int(fleet.get("memory_total_mib") or 0),
                    float(fleet.get("power_draw_w") or 0),
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            for server in servers:
                server_id = str(server.get("id") or "unknown")
                connection.execute(
                    """
                    INSERT INTO server_samples (
                        snapshot_id, server_id, hostname, online, cached,
                        latency_ms, uptime_seconds, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        server_id,
                        server.get("hostname"),
                        int(bool(server.get("online"))),
                        int(bool(server.get("cached"))),
                        server.get("latency_ms"),
                        server.get("uptime_seconds"),
                        server.get("error"),
                    ),
                )
                for volume in server.get("storage") or []:
                    connection.execute(
                        """
                        INSERT INTO storage_samples (
                            snapshot_id, server_id, mountpoint, kind,
                            filesystem_type, total_bytes, used_bytes,
                            available_bytes, used_percent, policy_limit_bytes,
                            policy_remaining_bytes, policy_used_percent,
                            policy_exceeded
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            server_id,
                            volume.get("mountpoint"),
                            volume.get("kind"),
                            volume.get("filesystem_type"),
                            volume.get("total_bytes"),
                            volume.get("used_bytes"),
                            volume.get("available_bytes"),
                            volume.get("used_percent"),
                            volume.get("policy_limit_bytes"),
                            volume.get("policy_remaining_bytes"),
                            volume.get("policy_used_percent"),
                            (
                                int(bool(volume.get("policy_exceeded")))
                                if volume.get("policy_exceeded") is not None
                                else None
                            ),
                        ),
                    )
                for gpu in server.get("gpus") or []:
                    gpu_index = int(gpu.get("index") or 0)
                    processes = gpu.get("processes") or []
                    connection.execute(
                        """
                        INSERT INTO gpu_samples (
                            snapshot_id, server_id, gpu_index, gpu_uuid, state,
                            utilization_percent, memory_used_mib,
                            memory_total_mib, temperature_c, power_draw_w,
                            power_limit_w, pstate, process_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            server_id,
                            gpu_index,
                            gpu.get("uuid"),
                            gpu.get("state"),
                            gpu.get("utilization_percent"),
                            gpu.get("memory_used_mib"),
                            gpu.get("memory_total_mib"),
                            gpu.get("temperature_c"),
                            gpu.get("power_draw_w"),
                            gpu.get("power_limit_w"),
                            gpu.get("pstate"),
                            len(processes),
                        ),
                    )
                    for process in processes:
                        connection.execute(
                            """
                            INSERT INTO process_samples (
                                snapshot_id, server_id, gpu_index, pid, user,
                                command, process_name, elapsed, memory_used_mib
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                snapshot_id,
                                server_id,
                                gpu_index,
                                process.get("pid"),
                                process.get("user"),
                                process.get("command"),
                                process.get("process_name"),
                                process.get("elapsed"),
                                process.get("memory_used_mib"),
                            ),
                        )
        return self.summary()

    def summary(self) -> dict[str, object]:
        with self._connect() as connection:
            first = connection.execute(
                "SELECT id, sampled_at FROM snapshots ORDER BY id ASC LIMIT 1"
            ).fetchone()
            last = connection.execute(
                "SELECT id, sampled_at FROM snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        database_bytes = sum(
            candidate.stat().st_size
            for candidate in (
                self.path,
                Path(f"{self.path}-wal"),
                Path(f"{self.path}-shm"),
            )
            if candidate.exists()
        )
        return {
            "enabled": True,
            "database": str(self.path),
            "sample_interval_seconds": self.sample_interval_seconds,
            "sample_count": int(last[0]) if last else 0,
            "tracking_since": first[1] if first else None,
            "last_recorded_at": last[1] if last else None,
            "database_bytes": database_bytes,
        }

    def recent_gpu_history(
        self,
        points_per_gpu: int,
    ) -> dict[tuple[str, int], list[dict[str, object]]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT server_id, gpu_index, sampled_at, utilization_percent,
                       memory_used_mib, temperature_c
                FROM (
                    SELECT g.server_id, g.gpu_index, s.sampled_at,
                           g.utilization_percent, g.memory_used_mib,
                           g.temperature_c,
                           ROW_NUMBER() OVER (
                               PARTITION BY g.server_id, g.gpu_index
                               ORDER BY g.snapshot_id DESC
                           ) AS position
                    FROM gpu_samples AS g
                    JOIN snapshots AS s ON s.id = g.snapshot_id
                )
                WHERE position <= ?
                ORDER BY sampled_at ASC
                """,
                (points_per_gpu,),
            ).fetchall()
        history: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
        for server_id, gpu_index, sampled_at, utilization, memory, temperature in rows:
            history[(str(server_id), int(gpu_index))].append(
                {
                    "at": sampled_at,
                    "utilization_percent": utilization,
                    "memory_used_mib": memory,
                    "temperature_c": temperature,
                }
            )
        return dict(history)

    def gpu_history(
        self,
        server_id: str,
        gpu_index: int,
        hours: float,
        max_points: int = 720,
    ) -> dict[str, object]:
        """Return bounded GPU metric and process history for one device."""
        range_end = datetime.now(timezone.utc)
        range_start = range_end - timedelta(hours=hours)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.sampled_at, g.utilization_percent, g.memory_used_mib,
                       g.memory_total_mib, g.temperature_c, g.power_draw_w,
                       g.power_limit_w, g.process_count, g.state
                FROM gpu_samples AS g
                JOIN snapshots AS s ON s.id = g.snapshot_id
                WHERE g.server_id = ? AND g.gpu_index = ?
                  AND s.sampled_at >= ?
                ORDER BY s.sampled_at ASC
                """,
                (server_id, gpu_index, range_start.isoformat()),
            ).fetchall()
            process_rows = connection.execute(
                """
                SELECT p.user, p.pid, COALESCE(p.command, p.process_name),
                       MIN(s.sampled_at), MAX(s.sampled_at),
                       MAX(p.memory_used_mib), COUNT(*)
                FROM process_samples AS p
                JOIN snapshots AS s ON s.id = p.snapshot_id
                WHERE p.server_id = ? AND p.gpu_index = ?
                  AND s.sampled_at >= ?
                GROUP BY p.user, p.pid, COALESCE(p.command, p.process_name)
                ORDER BY MAX(s.sampled_at) DESC
                LIMIT 50
                """,
                (server_id, gpu_index, range_start.isoformat()),
            ).fetchall()

        fields = (
            "utilization_percent",
            "memory_used_mib",
            "memory_total_mib",
            "temperature_c",
            "power_draw_w",
            "power_limit_w",
            "process_count",
        )
        raw_points = [
            {
                "sampled_at": row[0],
                **dict(zip(fields, row[1:8], strict=True)),
                "state": row[8],
            }
            for row in rows
        ]
        points = self._downsample(raw_points, max_points)
        processes = [
            {
                "user": row[0],
                "pid": row[1],
                "command": row[2],
                "first_seen_at": row[3],
                "last_seen_at": row[4],
                "peak_memory_used_mib": row[5],
                "sample_count": row[6],
            }
            for row in process_rows
        ]
        return {
            "server_id": server_id,
            "gpu_index": gpu_index,
            "hours": hours,
            "range_start": range_start.isoformat(),
            "range_end": range_end.isoformat(),
            "raw_point_count": len(raw_points),
            "point_count": len(points),
            "downsampled": len(points) < len(raw_points),
            "points": points,
            "processes": processes,
        }

    @staticmethod
    def _downsample(
        points: list[dict[str, object]],
        max_points: int,
    ) -> list[dict[str, object]]:
        if len(points) <= max_points:
            return points
        bucket_size = math.ceil(len(points) / max_points)
        numeric_fields = (
            "utilization_percent",
            "memory_used_mib",
            "memory_total_mib",
            "temperature_c",
            "power_draw_w",
            "power_limit_w",
        )
        sampled: list[dict[str, object]] = []
        for offset in range(0, len(points), bucket_size):
            bucket = points[offset : offset + bucket_size]
            point: dict[str, object] = {
                "sampled_at": bucket[-1]["sampled_at"],
                "state": bucket[-1]["state"],
                "process_count": max(
                    int(candidate.get("process_count") or 0) for candidate in bucket
                ),
                "samples": len(bucket),
            }
            for field in numeric_fields:
                values = [
                    float(candidate[field])
                    for candidate in bucket
                    if candidate.get(field) is not None
                ]
                point[field] = sum(values) / len(values) if values else None
            sampled.append(point)
        return sampled
