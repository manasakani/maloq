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
            "scope": "gpu",
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

    def aggregate_history(
        self,
        hours: float,
        server_id: str | None = None,
        max_points: int = 720,
    ) -> dict[str, object]:
        """Return fleet-wide or one-server aggregate metric history."""
        range_end = datetime.now(timezone.utc)
        range_start = range_end - timedelta(hours=hours)
        server_clause = " AND g.server_id = ?" if server_id is not None else ""
        parameters: tuple[object, ...] = (
            (range_start.isoformat(), server_id)
            if server_id is not None
            else (range_start.isoformat(),)
        )
        process_server_clause = (
            " AND p.server_id = ?" if server_id is not None else ""
        )
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT s.sampled_at, COUNT(g.gpu_index),
                       AVG(g.utilization_percent),
                       MAX(g.utilization_percent),
                       SUM(g.memory_used_mib), SUM(g.memory_total_mib),
                       AVG(g.temperature_c), MAX(g.temperature_c),
                       SUM(g.power_draw_w), SUM(g.power_limit_w),
                       SUM(g.process_count)
                FROM gpu_samples AS g
                JOIN snapshots AS s ON s.id = g.snapshot_id
                JOIN server_samples AS ss
                  ON ss.snapshot_id = g.snapshot_id
                 AND ss.server_id = g.server_id
                WHERE s.sampled_at >= ?
                  AND ss.online = 1 AND ss.cached = 0
                  {server_clause}
                GROUP BY g.snapshot_id
                ORDER BY s.sampled_at ASC
                """,
                parameters,
            ).fetchall()
            process_rows = connection.execute(
                f"""
                SELECT p.server_id, p.gpu_index, p.user, p.pid,
                       COALESCE(p.command, p.process_name),
                       MIN(s.sampled_at), MAX(s.sampled_at),
                       MAX(p.memory_used_mib), COUNT(*)
                FROM process_samples AS p
                JOIN snapshots AS s ON s.id = p.snapshot_id
                JOIN server_samples AS ss
                  ON ss.snapshot_id = p.snapshot_id
                 AND ss.server_id = p.server_id
                WHERE s.sampled_at >= ?
                  AND ss.online = 1 AND ss.cached = 0
                  {process_server_clause}
                GROUP BY p.server_id, p.gpu_index, p.user, p.pid,
                         COALESCE(p.command, p.process_name)
                ORDER BY MAX(s.sampled_at) DESC
                LIMIT 100
                """,
                parameters,
            ).fetchall()

        raw_points: list[dict[str, object]] = []
        for row in rows:
            memory_total = float(row[5] or 0)
            memory_used = float(row[4] or 0)
            raw_points.append(
                {
                    "sampled_at": row[0],
                    "reporting_gpus": row[1],
                    "utilization_average_percent": row[2],
                    "utilization_peak_percent": row[3],
                    "memory_used_mib": row[4],
                    "memory_total_mib": row[5],
                    "memory_utilization_percent": (
                        memory_used / memory_total * 100 if memory_total else 0
                    ),
                    "temperature_average_c": row[6],
                    "temperature_peak_c": row[7],
                    "power_draw_w": row[8],
                    "power_limit_w": row[9],
                    "process_count": row[10],
                }
            )
        points = self._downsample_aggregate(raw_points, max_points)
        latest = raw_points[-1] if raw_points else {}
        range_summary = {
            "latest": latest,
            "peak_utilization_percent": max(
                (
                    float(point.get("utilization_peak_percent") or 0)
                    for point in raw_points
                ),
                default=0,
            ),
            "peak_memory_utilization_percent": max(
                (
                    float(point.get("memory_utilization_percent") or 0)
                    for point in raw_points
                ),
                default=0,
            ),
            "peak_temperature_c": max(
                (
                    float(point.get("temperature_peak_c") or 0)
                    for point in raw_points
                ),
                default=0,
            ),
            "peak_power_draw_w": max(
                (float(point.get("power_draw_w") or 0) for point in raw_points),
                default=0,
            ),
            "max_process_count": max(
                (int(point.get("process_count") or 0) for point in raw_points),
                default=0,
            ),
        }
        processes = [
            {
                "server_id": row[0],
                "gpu_index": row[1],
                "user": row[2],
                "pid": row[3],
                "command": row[4],
                "first_seen_at": row[5],
                "last_seen_at": row[6],
                "peak_memory_used_mib": row[7],
                "sample_count": row[8],
            }
            for row in process_rows
        ]
        return {
            "scope": "server" if server_id is not None else "fleet",
            "server_id": server_id,
            "hours": hours,
            "range_start": range_start.isoformat(),
            "range_end": range_end.isoformat(),
            "raw_point_count": len(raw_points),
            "point_count": len(points),
            "downsampled": len(points) < len(raw_points),
            "points": points,
            "summary": range_summary,
            "processes": processes,
        }

    def storage_history(
        self,
        server_id: str,
        mountpoint: str,
        hours: float,
        max_points: int = 720,
    ) -> dict[str, object]:
        """Return bounded capacity history for one server mount."""
        range_end = datetime.now(timezone.utc)
        range_start = range_end - timedelta(hours=hours)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.sampled_at, st.kind, st.filesystem_type,
                       st.total_bytes, st.used_bytes, st.available_bytes,
                       st.used_percent, st.policy_limit_bytes,
                       st.policy_remaining_bytes, st.policy_used_percent,
                       st.policy_exceeded
                FROM storage_samples AS st
                JOIN snapshots AS s ON s.id = st.snapshot_id
                JOIN server_samples AS ss
                  ON ss.snapshot_id = st.snapshot_id
                 AND ss.server_id = st.server_id
                WHERE st.server_id = ? AND st.mountpoint = ?
                  AND s.sampled_at >= ?
                  AND ss.online = 1 AND ss.cached = 0
                ORDER BY s.sampled_at ASC
                """,
                (server_id, mountpoint, range_start.isoformat()),
            ).fetchall()
        raw_points: list[dict[str, object]] = []
        for row in rows:
            has_policy = row[7] is not None
            raw_points.append(
                {
                    "sampled_at": row[0],
                    "kind": row[1],
                    "filesystem_type": row[2],
                    "total_bytes": row[3],
                    "used_bytes": row[4],
                    "available_bytes": row[5],
                    "used_percent": row[6],
                    "policy_limit_bytes": row[7],
                    "policy_remaining_bytes": row[8],
                    "policy_used_percent": row[9],
                    "policy_exceeded": bool(row[10]) if row[10] is not None else None,
                    "effective_remaining_bytes": row[8] if has_policy else row[5],
                    "effective_used_percent": row[9] if has_policy else row[6],
                }
            )
        points = self._downsample_storage(raw_points, max_points)
        latest = raw_points[-1] if raw_points else {}
        first = raw_points[0] if raw_points else {}
        used_values = [
            int(point.get("used_bytes") or 0) for point in raw_points
        ]
        percent_values = [
            float(point.get("effective_used_percent") or 0)
            for point in raw_points
        ]
        return {
            "scope": "storage",
            "server_id": server_id,
            "mountpoint": mountpoint,
            "kind": latest.get("kind"),
            "hours": hours,
            "range_start": range_start.isoformat(),
            "range_end": range_end.isoformat(),
            "raw_point_count": len(raw_points),
            "point_count": len(points),
            "downsampled": len(points) < len(raw_points),
            "points": points,
            "summary": {
                "latest": latest,
                "change_bytes": (
                    int(latest.get("used_bytes") or 0)
                    - int(first.get("used_bytes") or 0)
                    if raw_points
                    else 0
                ),
                "minimum_used_bytes": min(used_values, default=0),
                "maximum_used_bytes": max(used_values, default=0),
                "peak_used_percent": max(percent_values, default=0),
            },
            "processes": [],
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

    @staticmethod
    def _downsample_aggregate(
        points: list[dict[str, object]],
        max_points: int,
    ) -> list[dict[str, object]]:
        if len(points) <= max_points:
            return points
        bucket_size = math.ceil(len(points) / max_points)
        average_fields = (
            "utilization_average_percent",
            "memory_used_mib",
            "memory_total_mib",
            "memory_utilization_percent",
            "temperature_average_c",
            "power_draw_w",
            "power_limit_w",
        )
        peak_fields = (
            "reporting_gpus",
            "utilization_peak_percent",
            "temperature_peak_c",
            "process_count",
        )
        sampled: list[dict[str, object]] = []
        for offset in range(0, len(points), bucket_size):
            bucket = points[offset : offset + bucket_size]
            point: dict[str, object] = {
                "sampled_at": bucket[-1]["sampled_at"],
                "samples": len(bucket),
            }
            for field in average_fields:
                values = [
                    float(candidate[field])
                    for candidate in bucket
                    if candidate.get(field) is not None
                ]
                point[field] = sum(values) / len(values) if values else None
            for field in peak_fields:
                point[field] = max(
                    (float(candidate.get(field) or 0) for candidate in bucket),
                    default=0,
                )
            sampled.append(point)
        return sampled

    @staticmethod
    def _downsample_storage(
        points: list[dict[str, object]],
        max_points: int,
    ) -> list[dict[str, object]]:
        if len(points) <= max_points:
            return points
        bucket_size = math.ceil(len(points) / max_points)
        return [
            points[min(offset + bucket_size, len(points)) - 1]
            for offset in range(0, len(points), bucket_size)
        ]
