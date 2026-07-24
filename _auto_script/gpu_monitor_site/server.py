#!/usr/bin/env python3
"""Serve a private, read-only dashboard for the two SC26 GPU servers."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path("/dataset/seongsu/shared-home/workspace/project")
SITE_ROOT = PROJECT_ROOT / "_auto_script/gpu_monitor_site"
STATIC_ROOT = SITE_ROOT / "static"
COLLECTOR = SITE_ROOT / "collect_gpu_status.py"
PYTHON = Path("/usr/bin/python3")
HISTORY_LENGTH = 120

if str(SITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SITE_ROOT))

from history_store import HistoryStore  # noqa: E402

SERVER_CONFIGS = (
    {
        "id": "server-1",
        "label": "GPU Server 1",
        "transport": "local",
        "ssh_target": None,
    },
    {
        "id": "server-2",
        "label": "GPU Server 2",
        "transport": "ssh",
        "ssh_target": "scp-gpu-2",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_state(gpu: dict[str, object]) -> str:
    memory = int(gpu.get("memory_used_mib") or 0)
    utilization = int(gpu.get("utilization_percent") or 0)
    processes = gpu.get("processes") or []
    temperature = int(gpu.get("temperature_c") or 0)
    if temperature >= 80:
        return "warning"
    if memory > 1024 or utilization > 5 or processes:
        return "busy"
    return "idle"


def fleet_summary(servers: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, float | int] = {
        "servers_online": 0,
        "servers_total": len(servers),
        "gpus_total": 0,
        "gpus_idle": 0,
        "gpus_busy": 0,
        "gpus_warning": 0,
        "gpus_offline": 0,
        "memory_used_mib": 0,
        "memory_total_mib": 0,
        "power_draw_w": 0.0,
    }
    for server in servers:
        gpus = server.get("gpus") or []
        gpu_slots = len(gpus) or 8
        summary["gpus_total"] += gpu_slots
        if server.get("online"):
            summary["servers_online"] += 1
        else:
            summary["gpus_offline"] += gpu_slots
            continue
        for gpu in gpus:
            state = gpu_state(gpu)
            summary[f"gpus_{state}"] += 1
            summary["memory_used_mib"] += int(gpu.get("memory_used_mib") or 0)
            summary["memory_total_mib"] += int(gpu.get("memory_total_mib") or 0)
            summary["power_draw_w"] += float(gpu.get("power_draw_w") or 0)
    return summary


class MonitorState:
    def __init__(
        self,
        refresh_seconds: float,
        history_store: HistoryStore | None = None,
        history_seconds: float = 60,
    ) -> None:
        self.refresh_seconds = refresh_seconds
        self.history_store = history_store
        self.history_seconds = history_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._servers: dict[str, dict[str, object]] = {}
        self._last_good: dict[str, dict[str, object]] = {}
        self._history: dict[tuple[str, int], deque[dict[str, object]]] = defaultdict(
            lambda: deque(maxlen=HISTORY_LENGTH)
        )
        self._last_history_write = 0.0
        self._tracking = (
            history_store.summary()
            if history_store is not None
            else {"enabled": False}
        )
        if history_store is not None:
            persisted = history_store.recent_gpu_history(HISTORY_LENGTH)
            for key, points in persisted.items():
                self._history[key].extend(points)
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="gpu-poller",
            daemon=True,
        )

    def start(self) -> None:
        self.poll_once()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=5)

    def trigger(self) -> None:
        self._wake.set()

    def _collector_command(self, config: dict[str, object]) -> list[str]:
        collector_args = [
            str(PYTHON),
            str(COLLECTOR),
            "--id",
            str(config["id"]),
            "--label",
            str(config["label"]),
        ]
        if config["transport"] == "local":
            return collector_args
        return [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            "-o",
            "ServerAliveInterval=3",
            str(config["ssh_target"]),
            shlex.join(collector_args),
        ]

    def _collect_one(self, config: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        try:
            result = subprocess.run(
                self._collector_command(config),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            payload = json.loads(result.stdout)
            payload.update(
                {
                    "online": True,
                    "error": None,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "last_success_at": utc_now(),
                }
            )
            return payload
        except (
            OSError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
        ) as exc:
            message = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                stderr = (exc.stderr or "").strip().splitlines()
                if stderr:
                    message = stderr[-1]
            return {
                "id": config["id"],
                "label": config["label"],
                "hostname": None,
                "sampled_at": utc_now(),
                "gpu_count": 0,
                "gpus": [],
                "online": False,
                "error": message[:400],
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "last_success_at": None,
            }

    def poll_once(self) -> None:
        collected: dict[str, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=len(SERVER_CONFIGS)) as executor:
            futures = {
                executor.submit(self._collect_one, config): config
                for config in SERVER_CONFIGS
            }
            for future in as_completed(futures):
                payload = future.result()
                collected[str(payload["id"])] = payload

        sampled_at = utc_now()
        history_servers: list[dict[str, object]] = []
        with self._lock:
            for config in SERVER_CONFIGS:
                server_id = str(config["id"])
                payload = collected[server_id]
                if payload["online"]:
                    self._last_good[server_id] = deepcopy(payload)
                elif server_id in self._last_good:
                    cached = deepcopy(self._last_good[server_id])
                    cached.update(
                        {
                            "online": False,
                            "error": payload["error"],
                            "latency_ms": payload["latency_ms"],
                            "sampled_at": payload["sampled_at"],
                            "cached": True,
                        }
                    )
                    payload = cached
                for gpu in payload.get("gpus") or []:
                    index = int(gpu["index"])
                    state = gpu_state(gpu)
                    gpu["state"] = state
                    history_point = {
                        "at": sampled_at,
                        "utilization_percent": gpu.get("utilization_percent"),
                        "memory_used_mib": gpu.get("memory_used_mib"),
                        "temperature_c": gpu.get("temperature_c"),
                    }
                    self._history[(server_id, index)].append(history_point)
                    gpu["history"] = list(self._history[(server_id, index)])
                self._servers[server_id] = payload
            history_servers = [
                deepcopy(self._servers.get(str(config["id"]), {}))
                for config in SERVER_CONFIGS
            ]

        history_due = (
            self.history_store is not None
            and time.monotonic() - self._last_history_write >= self.history_seconds
        )
        if history_due and self.history_store is not None:
            try:
                tracking = self.history_store.record(
                    sampled_at,
                    history_servers,
                    fleet_summary(history_servers),
                )
                with self._lock:
                    self._tracking = tracking
                self._last_history_write = time.monotonic()
            except Exception:
                logging.exception("Could not persist GPU monitor history.")

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self.refresh_seconds)
            self._wake.clear()
            if self._stop.is_set():
                return
            self.poll_once()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            servers = [
                deepcopy(self._servers.get(str(config["id"]), {}))
                for config in SERVER_CONFIGS
            ]
        return {
            "generated_at": utc_now(),
            "refresh_seconds": self.refresh_seconds,
            "fleet": fleet_summary(servers),
            "servers": servers,
            "tracking": deepcopy(self._tracking),
        }

    def gpu_history(
        self,
        server_id: str,
        gpu_index: int,
        hours: float,
    ) -> dict[str, object] | None:
        if self.history_store is None:
            return None
        return self.history_store.gpu_history(server_id, gpu_index, hours)


class DashboardHandler(BaseHTTPRequestHandler):
    state: MonitorState
    server_version = "SC26Monitor/1.0"
    sys_version = ""

    def log_message(self, format_string: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), format_string % args)

    def _headers(
        self,
        status: HTTPStatus,
        content_type: str,
        content_length: int,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'",
        )
        self.end_headers()

    def _send_json(self, payload: dict[str, object], status=HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_static(self, relative_path: str, *, include_body: bool = True) -> None:
        target = (STATIC_ROOT / relative_path).resolve()
        static_root = STATIC_ROOT.resolve()
        if target != static_root and static_root not in target.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "image/svg+xml",
        }:
            content_type += "; charset=utf-8"
        self._headers(HTTPStatus.OK, content_type, len(body))
        if include_body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            self._send_json(self.state.snapshot())
            return
        if path == "/api/history":
            query = parse_qs(parsed.query)
            server_id = query.get("server_id", [""])[0]
            gpu_text = query.get("gpu_index", [""])[0]
            hours_text = query.get("hours", ["24"])[0]
            valid_servers = {str(config["id"]) for config in SERVER_CONFIGS}
            try:
                gpu_index = int(gpu_text)
                hours = float(hours_text)
            except ValueError:
                self._send_json(
                    {"error": "gpu_index and hours must be numeric"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if server_id not in valid_servers or not 0 <= gpu_index < 8:
                self._send_json(
                    {"error": "unknown server_id or GPU index"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if hours not in {1, 6, 24, 168}:
                self._send_json(
                    {"error": "hours must be one of 1, 6, 24, or 168"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            history = self.state.gpu_history(server_id, gpu_index, hours)
            if history is None:
                self._send_json(
                    {"error": "history tracking is unavailable"},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            self._send_json(history)
            return
        if path == "/healthz":
            snapshot = self.state.snapshot()
            self._send_json(
                {
                    "status": "ok",
                    "servers_online": snapshot["fleet"]["servers_online"],
                    "servers_total": snapshot["fleet"]["servers_total"],
                    "history_samples": snapshot["tracking"].get("sample_count", 0),
                    "history_last_recorded_at": snapshot["tracking"].get(
                        "last_recorded_at"
                    ),
                }
            )
            return
        if path in {"/", "/index.html"}:
            self._send_static("index.html")
            return
        if path.startswith("/assets/"):
            self._send_static(path.removeprefix("/assets/"))
            return
        if path == "/favicon.svg":
            self._send_static("favicon.svg")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_static("index.html", include_body=False)
            return
        if path.startswith("/assets/"):
            self._send_static(path.removeprefix("/assets/"), include_body=False)
            return
        if path == "/favicon.svg":
            self._send_static("favicon.svg", include_body=False)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/refresh":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.state.trigger()
        self._send_json({"accepted": True}, status=HTTPStatus.ACCEPTED)


def configure_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument("--history-seconds", type=float, default=60.0)
    parser.add_argument("--history-db", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.refresh_seconds < 2:
        parser.error("--refresh-seconds must be at least 2")
    if args.history_seconds < args.refresh_seconds:
        parser.error("--history-seconds must be at least --refresh-seconds")

    configure_logging(args.log_file)
    history_store = (
        HistoryStore(args.history_db, args.history_seconds)
        if args.history_db is not None
        else None
    )
    state = MonitorState(
        args.refresh_seconds,
        history_store=history_store,
        history_seconds=args.history_seconds,
    )
    state.start()
    DashboardHandler.state = state
    server = ThreadingHTTPServer((args.bind, args.port), DashboardHandler)

    def stop_server(signum: int, _frame: object) -> None:
        logging.info("Received signal %s; stopping.", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    logging.info(
        "SC26 GPU monitor listening on http://%s:%s", args.bind, args.port
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        state.stop()
        logging.info("SC26 GPU monitor stopped.")


if __name__ == "__main__":
    main()
