const ui = {
  servers: document.querySelector("#servers"),
  liveState: document.querySelector("#live-state"),
  liveLabel: document.querySelector("#live-label"),
  lastUpdated: document.querySelector("#last-updated"),
  refreshCaption: document.querySelector("#refresh-caption"),
  refreshButton: document.querySelector("#refresh-button"),
  viewToggle: document.querySelector("#view-toggle"),
  available: document.querySelector("#summary-available"),
  serverState: document.querySelector("#summary-server-state"),
  gpus: document.querySelector("#summary-gpus"),
  gpusMeter: document.querySelector("#summary-gpus-meter"),
  memory: document.querySelector("#summary-memory"),
  memoryTotal: document.querySelector("#summary-memory-total"),
  memoryProgress: document.querySelector("#summary-memory-progress"),
  memoryPercent: document.querySelector("#summary-memory-percent"),
  power: document.querySelector("#summary-power"),
  busy: document.querySelector("#summary-busy"),
  fleetHistoryButton: document.querySelector("#fleet-history-button"),
  fleetStatisticsCaption: document.querySelector("#fleet-statistics-caption"),
  fleetUtilizationAverage: document.querySelector("#fleet-utilization-average"),
  fleetReportingGpus: document.querySelector("#fleet-reporting-gpus"),
  fleetUtilizationPeak: document.querySelector("#fleet-utilization-peak"),
  fleetUtilizationPeakGpu: document.querySelector("#fleet-utilization-peak-gpu"),
  fleetMemoryRatio: document.querySelector("#fleet-memory-ratio"),
  fleetMemoryDetail: document.querySelector("#fleet-memory-detail"),
  fleetTemperatureAverage: document.querySelector("#fleet-temperature-average"),
  fleetTemperaturePeak: document.querySelector("#fleet-temperature-peak"),
  fleetProcesses: document.querySelector("#fleet-processes"),
  fleetUsers: document.querySelector("#fleet-users"),
  fleetUserNames: document.querySelector("#fleet-user-names"),
  filterGroup: document.querySelector("#filter-group"),
  footerVersion: document.querySelector("#footer-version"),
  historyModal: document.querySelector("#history-modal"),
  historyEyebrow: document.querySelector("#history-eyebrow"),
  historyTitle: document.querySelector("#history-title"),
  historySubtitle: document.querySelector("#history-subtitle"),
  historyRange: document.querySelector("#history-range"),
  historyState: document.querySelector("#history-state"),
  historySummary: document.querySelector("#history-summary"),
  historyCharts: document.querySelector("#history-charts"),
  historyProcessCount: document.querySelector("#history-process-count"),
  historyProcessList: document.querySelector("#history-process-list"),
  processHistorySection: document.querySelector("#process-history-section"),
  historyRefresh: document.querySelector("#history-refresh"),
  toast: document.querySelector("#toast"),
};

let activeFilter = "all";
let refreshTimer = null;
let toastTimer = null;
let historySelection = null;
let historyHours = 24;
let historyRequestId = 0;
let viewTransitionTimer = null;
const openProcessPanels = new Set();
const viewStorageKey = "sc26-gpu-view";
let cleanView = false;

try {
  cleanView = window.localStorage.getItem(viewStorageKey) === "clean";
} catch (_error) {
  cleanView = false;
}

const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const clamp = (value, min = 0, max = 100) =>
  Math.min(max, Math.max(min, Number(value) || 0));

const formatMemory = (mib) => {
  const value = Number(mib) || 0;
  return value >= 1024 ? `${(value / 1024).toFixed(value >= 10240 ? 0 : 1)} GB` : `${value} MB`;
};

const formatPower = (watts) =>
  watts == null ? "—" : `${Math.round(Number(watts))} W`;

const formatBytes = (bytes) => {
  let value = Number(bytes) || 0;
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = value >= 100 || unit === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unit]}`;
};

const formatDecimalTerabytes = (bytes) => {
  const value = (Number(bytes) || 0) / 1_000_000_000_000;
  return `${value.toFixed(value >= 10 ? 1 : 2)} TB`;
};

const formatTime = (iso) => {
  if (!iso) return "No sample";
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    month: "short",
    day: "numeric",
  }).format(new Date(iso));
};

const relativeAge = (iso) => {
  if (!iso) return "never";
  const seconds = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (seconds < 4) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  return `${Math.round(seconds / 60)}m ago`;
};

function sparkline(history = []) {
  const values = history.slice(-36).map((point) => clamp(point.utilization_percent));
  if (!values.length) values.push(0);
  const width = 68;
  const height = 36;
  const points = values
    .map((value, index) => {
      const x = values.length === 1 ? width : (index / (values.length - 1)) * width;
      const y = height - (value / 100) * (height - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const area = `0,${height} ${points} ${width},${height}`;
  return `
    <svg class="sparkline" viewBox="0 0 ${width} ${height}" aria-label="Recent utilization">
      <polyline class="area" points="${area}"></polyline>
      <polyline class="line" points="${points}"></polyline>
    </svg>`;
}

function processList(processes = []) {
  if (!processes.length) {
    return `<div class="empty-process">No compute process</div>`;
  }
  return processes
    .map((process) => {
      const name = process.command || process.process_name || "process";
      const user = process.user || "unknown";
      const elapsed = process.elapsed || "—";
      return `
        <div class="process" title="PID ${escapeHtml(process.pid)}">
          <div class="process-identity">
            <strong>${escapeHtml(user)}</strong>
            <span>${escapeHtml(name)}</span>
          </div>
          <small>PID ${escapeHtml(process.pid)} · ${escapeHtml(elapsed)} · ${formatMemory(process.memory_used_mib)}</small>
        </div>`;
    })
    .join("");
}

function gpuCard(gpu, serverId) {
  const state = gpu.state || "idle";
  const utilization = clamp(gpu.utilization_percent);
  const memoryPercent = gpu.memory_total_mib
    ? clamp((gpu.memory_used_mib / gpu.memory_total_mib) * 100)
    : 0;
  const processCount = (gpu.processes || []).length;
  const processKey = `${serverId}:${gpu.index}`;
  const processOpen = openProcessPanels.has(processKey) ? " open" : "";
  const processEmpty = processCount ? "" : " empty";
  return `
    <article class="gpu-card ${escapeHtml(state)}" data-state="${escapeHtml(state)}">
      <div class="gpu-title">
        <h4>GPU ${escapeHtml(gpu.index)}</h4>
        <div class="gpu-actions">
          <button
            class="history-button"
            type="button"
            data-history-scope="gpu"
            data-server-id="${escapeHtml(serverId)}"
            data-gpu-index="${escapeHtml(gpu.index)}"
            data-gpu-name="${escapeHtml(gpu.name)}"
            title="Open persistent GPU history"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 18V9m5 9V5m5 13v-7m5 7V3"></path>
            </svg>
            <span>History</span>
          </button>
          <div class="gpu-status">
            ${processCount ? `<span class="compact-process-count">${processCount} proc</span>` : ""}
            <i class="dot ${escapeHtml(state)}"></i>${escapeHtml(state)}
          </div>
        </div>
      </div>
      <p class="gpu-model">${escapeHtml(gpu.name)}</p>
      <div class="util-row">
        <div class="util-value">
          <strong>${Math.round(utilization)}%</strong>
          <span>GPU utilization</span>
        </div>
        ${sparkline(gpu.history)}
      </div>
      <div class="metric-stack">
        <div class="metric-row memory-row">
          <span>Memory</span>
          <strong>${formatMemory(gpu.memory_used_mib)} / ${formatMemory(gpu.memory_total_mib)}</strong>
        </div>
        <div class="memory-progress-row">
          <progress
            class="memory-progress gpu-memory-progress"
            max="100"
            value="${memoryPercent.toFixed(1)}"
            aria-label="GPU ${escapeHtml(gpu.index)} memory utilization"
          ></progress>
          <strong>${memoryPercent.toFixed(1)}%</strong>
        </div>
        <div class="metric-row thermal-row">
          <span>Thermal</span>
          <strong>${gpu.temperature_c ?? "—"}°C</strong>
        </div>
        <div class="metric-row power-row">
          <span>Power</span>
          <strong>${formatPower(gpu.power_draw_w)} / ${formatPower(gpu.power_limit_w)}</strong>
        </div>
        <div class="metric-row pstate-row">
          <span>Performance state</span>
          <strong>${escapeHtml(gpu.pstate || "—")}</strong>
        </div>
      </div>
      <details class="processes${processEmpty}" data-process-key="${escapeHtml(processKey)}"${processOpen}>
        <summary>
          <span>Compute processes</span>
          <span class="process-count">${processCount}</span>
        </summary>
        <div class="process-list">${processList(gpu.processes)}</div>
      </details>
    </article>`;
}

function storagePanel(storage = [], serverId = "", serverLabel = "") {
  if (!storage.length) {
    return `<div class="storage-unavailable">Storage status unavailable</div>`;
  }
  return `
    <div class="server-storage" aria-label="Server storage">
      ${storage
        .map((volume) => {
          const hasPolicy = volume.policy_limit_bytes != null;
          const rawPercent = hasPolicy
            ? Number(volume.policy_used_percent) || 0
            : Number(volume.used_percent) || 0;
          const meterPercent = clamp(rawPercent);
          const percentLabel = rawPercent < 10
            ? `${rawPercent.toFixed(1)}%`
            : `${Math.round(rawPercent)}%`;
          const policyState = volume.policy_exceeded
            ? " exceeded"
            : hasPolicy && rawPercent >= 80
              ? " attention"
              : "";
          const badge = hasPolicy
            ? `<span class="storage-kind">40 TB limit</span>`
            : `<span class="storage-kind local">physical</span>`;
          const capacityLine = hasPolicy
            ? `<span>${formatDecimalTerabytes(volume.policy_remaining_bytes)} budget left · ${formatDecimalTerabytes(volume.policy_limit_bytes)} limit</span>`
            : `<span>${formatBytes(volume.available_bytes)} free · ${formatBytes(volume.total_bytes)} total</span>`;
          const detailLine = hasPolicy
            ? `Physical ${formatBytes(volume.total_bytes)} · ${volume.mountpoint} · ${volume.filesystem_type}`
            : `${volume.mountpoint} · ${volume.filesystem_type}`;
          const usedLabel = hasPolicy
            ? formatDecimalTerabytes(volume.used_bytes)
            : formatBytes(volume.used_bytes);
          return `
            <div class="storage-volume${policyState}">
              <div class="storage-topline">
                <span>${escapeHtml(volume.label)} ${badge}</span>
                <div class="storage-topline-actions">
                  <strong>${percentLabel}</strong>
                  <button
                    class="storage-history-button"
                    type="button"
                    data-history-scope="storage"
                    data-server-id="${escapeHtml(serverId)}"
                    data-server-label="${escapeHtml(serverLabel)}"
                    data-mountpoint="${escapeHtml(volume.mountpoint)}"
                    data-storage-kind="${escapeHtml(volume.kind)}"
                    data-volume-label="${escapeHtml(volume.label)}"
                    title="Open persistent storage history"
                  >
                    History
                  </button>
                </div>
              </div>
              <div class="storage-values">
                <span>${usedLabel} used</span>
                ${capacityLine}
              </div>
              <progress
                class="memory-progress storage-progress"
                max="100"
                value="${meterPercent.toFixed(1)}"
                aria-label="${escapeHtml(volume.label)} utilization"
              ></progress>
              <small>${escapeHtml(detailLine)}</small>
            </div>`;
        })
        .join("")}
    </div>`;
}

function serverPanel(server, index) {
  const gpus = server.gpus || [];
  const idle = gpus.filter((gpu) => gpu.state === "idle").length;
  const onlineClass = server.online ? "idle" : "offline";
  const error = server.error
    ? `<div class="server-error">${escapeHtml(server.error)}${server.cached ? " · showing last good sample" : ""}</div>`
    : "";
  const cards = gpus.length
    ? gpus.map((gpu) => gpuCard(gpu, server.id)).join("")
    : `<div class="server-error">No GPU data is available for this server.</div>`;
  return `
    <article class="server-panel">
      <header class="server-header">
        <div class="server-identity">
          <div class="server-number">0${index + 1}</div>
          <div>
            <p class="server-overline">${escapeHtml(server.id || `server-${index + 1}`)}</p>
            <h3>${escapeHtml(server.label || "GPU Server")}</h3>
          </div>
        </div>
        <div class="server-meta">
          <span><i class="dot ${onlineClass}"></i> ${server.online ? "Online" : "Offline"}</span>
          <span><strong>${idle}</strong> idle / ${gpus.length} GPUs</span>
          <span>${escapeHtml(server.hostname || "unreachable")}</span>
          <span>${server.latency_ms ?? "—"} ms</span>
          <button
            class="scope-history-button"
            type="button"
            data-history-scope="server"
            data-server-id="${escapeHtml(server.id)}"
            data-server-label="${escapeHtml(server.label || "GPU Server")}"
            title="Open history aggregated across this server"
          >
            Server history
          </button>
        </div>
      </header>
      ${error}
      ${storagePanel(server.storage, server.id, server.label)}
      <div class="gpu-grid">${cards}</div>
    </article>`;
}

function applyFilter() {
  document.querySelectorAll(".gpu-card").forEach((card) => {
    card.classList.toggle(
      "hidden",
      activeFilter !== "all" && card.dataset.state !== activeFilter,
    );
  });
}

function syncViewMode() {
  document.body.classList.toggle("clean-view", cleanView);
  ui.viewToggle.setAttribute("aria-pressed", String(cleanView));
  ui.viewToggle.title = cleanView
    ? "Show GPU history and process details"
    : "Show a compact GPU overview";
  ui.viewToggle.querySelector("span").textContent = cleanView
    ? "Detailed view"
    : "Clean view";
}

function applyViewMode({ animate = false } = {}) {
  if (!animate || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    syncViewMode();
    return;
  }
  if (document.startViewTransition) {
    document.startViewTransition(syncViewMode);
    return;
  }
  window.clearTimeout(viewTransitionTimer);
  document.body.classList.add("view-changing");
  viewTransitionTimer = window.setTimeout(() => {
    syncViewMode();
    window.requestAnimationFrame(() => {
      document.body.classList.remove("view-changing");
    });
  }, 120);
}

function render(payload) {
  const fleet = payload.fleet || {};
  const servers = payload.servers || [];
  const online = Number(fleet.servers_online) || 0;
  const serverTotal = Number(fleet.servers_total) || servers.length;
  const totalGpus = Number(fleet.gpus_total) || 0;
  const idle = Number(fleet.gpus_idle) || 0;
  const busy = (Number(fleet.gpus_busy) || 0) + (Number(fleet.gpus_warning) || 0);
  const memoryUsed = Number(fleet.memory_used_mib) || 0;
  const memoryTotal = Number(fleet.memory_total_mib) || 0;
  const memoryPercent = memoryTotal ? clamp((memoryUsed / memoryTotal) * 100) : 0;
  const activeUsers = Array.isArray(fleet.active_users) ? fleet.active_users : [];

  ui.liveState.classList.toggle("online", online === serverTotal && serverTotal > 0);
  ui.liveLabel.textContent =
    online === serverTotal ? "All systems live" : `${online}/${serverTotal} servers online`;
  ui.lastUpdated.textContent = `${formatTime(payload.generated_at)} · ${relativeAge(payload.generated_at)}`;
  ui.refreshCaption.textContent = `Automatic refresh every ${payload.refresh_seconds || 5} seconds`;
  ui.available.textContent = idle;
  ui.serverState.textContent = `${online} of ${serverTotal} servers responding`;
  ui.gpus.textContent = totalGpus;
  ui.gpusMeter.style.width = `${serverTotal ? clamp((totalGpus / (serverTotal * 8)) * 100) : 0}%`;
  ui.memory.textContent = formatMemory(memoryUsed);
  ui.memoryTotal.textContent = `of ${formatMemory(memoryTotal)}`;
  ui.memoryProgress.value = memoryPercent;
  ui.memoryPercent.textContent = `${memoryPercent.toFixed(1)}%`;
  ui.power.textContent = Math.round(Number(fleet.power_draw_w) || 0).toLocaleString();
  ui.busy.textContent = busy ? `${busy} GPU${busy === 1 ? "" : "s"} carrying work` : "No active jobs detected";
  ui.fleetStatisticsCaption.textContent = `${Number(fleet.gpus_reporting) || 0} reporting GPUs · live 5-second sample`;
  ui.fleetUtilizationAverage.textContent = `${(Number(fleet.utilization_average_percent) || 0).toFixed(1)}%`;
  ui.fleetReportingGpus.textContent = `${Number(fleet.gpus_reporting) || 0} GPUs reporting`;
  ui.fleetUtilizationPeak.textContent = `${Math.round(Number(fleet.utilization_peak_percent) || 0)}%`;
  ui.fleetUtilizationPeakGpu.textContent = fleet.utilization_peak_gpu || "No sample";
  ui.fleetMemoryRatio.textContent = `${memoryPercent.toFixed(1)}%`;
  ui.fleetMemoryDetail.textContent = `${formatMemory(memoryUsed)} / ${formatMemory(memoryTotal)}`;
  ui.fleetTemperatureAverage.textContent = `${(Number(fleet.temperature_average_c) || 0).toFixed(1)}°C`;
  ui.fleetTemperaturePeak.textContent = `Peak ${Math.round(Number(fleet.temperature_peak_c) || 0)}°C`;
  ui.fleetProcesses.textContent = Number(fleet.processes_active) || 0;
  ui.fleetUsers.textContent = activeUsers.length;
  ui.fleetUserNames.textContent = activeUsers.length
    ? activeUsers.join(", ")
    : "No active user";
  const tracking = payload.tracking || {};
  ui.footerVersion.textContent = tracking.enabled
    ? `${Number(tracking.sample_count || 0).toLocaleString()} saved snapshots · every ${tracking.sample_interval_seconds || 60}s`
    : "History tracking unavailable";
  openProcessPanels.clear();
  ui.servers.querySelectorAll("details.processes[open]").forEach((panel) => {
    openProcessPanels.add(panel.dataset.processKey);
  });
  ui.servers.innerHTML = servers.map(serverPanel).join("");
  applyFilter();
}

function showToast(message) {
  clearTimeout(toastTimer);
  ui.toast.textContent = message;
  ui.toast.classList.add("show");
  toastTimer = setTimeout(() => ui.toast.classList.remove("show"), 2400);
}

const formatHistoryTimestamp = (iso, includeDate = true) => {
  if (!iso) return "—";
  const options = includeDate
    ? { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
    : { hour: "2-digit", minute: "2-digit" };
  return new Intl.DateTimeFormat(undefined, options).format(new Date(iso));
};

function historyChart(points, config) {
  const width = 760;
  const height = 208;
  const plot = { left: 45, right: 14, top: 15, bottom: 29 };
  const plotWidth = width - plot.left - plot.right;
  const plotHeight = height - plot.top - plot.bottom;
  const samples = points
    .map((point) => ({
      at: new Date(point.sampled_at).getTime(),
      value: point[config.field] == null
        ? null
        : config.transform(Number(point[config.field])),
    }))
    .filter((point) => Number.isFinite(point.at) && Number.isFinite(point.value));
  if (!samples.length) {
    return `
      <article class="history-chart">
        <div class="history-chart-heading"><h3>${config.title}</h3><span>—</span></div>
        <div class="history-chart-empty">No saved samples in this range</div>
      </article>`;
  }

  const values = samples.map((sample) => sample.value);
  const minimum = config.minimum ?? Math.min(...values);
  const rawMaximum = config.maximum ?? Math.max(...values);
  const maximum = rawMaximum > minimum ? rawMaximum : minimum + 1;
  const firstTime = samples[0].at;
  const lastTime = samples[samples.length - 1].at;
  const timeSpan = Math.max(1, lastTime - firstTime);
  const coordinates = samples.map((sample) => {
    const x = plot.left + ((sample.at - firstTime) / timeSpan) * plotWidth;
    const y = plot.top + (1 - (sample.value - minimum) / (maximum - minimum)) * plotHeight;
    return { x, y };
  });
  const line = coordinates
    .map((point, index) => `${index ? "L" : "M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
    .join(" ");
  const area = `${line} L${coordinates.at(-1).x.toFixed(1)},${(plot.top + plotHeight).toFixed(1)} L${coordinates[0].x.toFixed(1)},${(plot.top + plotHeight).toFixed(1)} Z`;
  const grid = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    const y = plot.top + ratio * plotHeight;
    const value = maximum - ratio * (maximum - minimum);
    return `
      <line x1="${plot.left}" y1="${y.toFixed(1)}" x2="${width - plot.right}" y2="${y.toFixed(1)}"></line>
      <text x="${plot.left - 8}" y="${(y + 3).toFixed(1)}">${config.axis(value)}</text>`;
  }).join("");
  const latest = values.at(-1);
  const peak = Math.max(...values);
  return `
    <article class="history-chart ${config.className}">
      <div class="history-chart-heading">
        <h3>${config.title}</h3>
        <span>${config.value(latest)} latest · ${config.value(peak)} peak</span>
      </div>
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${config.title} history">
        <g class="history-grid">${grid}</g>
        <path class="history-area" d="${area}"></path>
        <path class="history-line" d="${line}"></path>
      </svg>
      <div class="history-chart-time">
        <span>${formatHistoryTimestamp(samples[0].at)}</span>
        <span>${formatHistoryTimestamp(samples.at(-1).at)}</span>
      </div>
    </article>`;
}

function renderProcessHistory(processes = [], emptyScope = "this GPU") {
  ui.historyProcessCount.textContent = `${processes.length} process record${processes.length === 1 ? "" : "s"}`;
  ui.historyProcessList.innerHTML = processes.length
    ? `
      <div class="history-process-row heading">
        <span>User / command</span><span>PID</span><span>First seen</span><span>Last seen</span><span>Peak GPU memory</span>
      </div>
      ${processes.map((process) => {
        const location = process.server_id != null
          ? `${process.server_id} / GPU ${process.gpu_index}`
          : "";
        const detail = [process.command || "process", location]
          .filter(Boolean)
          .join(" · ");
        return `
          <div class="history-process-row">
            <span><strong>${escapeHtml(process.user || "unknown")}</strong><small>${escapeHtml(detail)}</small></span>
            <span>${escapeHtml(process.pid)}</span>
            <span>${formatHistoryTimestamp(process.first_seen_at)}</span>
            <span>${formatHistoryTimestamp(process.last_seen_at)}</span>
            <span>${formatMemory(process.peak_memory_used_mib)}</span>
          </div>`;
      }).join("")}`
    : `<div class="history-process-empty">No compute process was recorded across ${escapeHtml(emptyScope)} in the selected range.</div>`;
}

function renderGpuHistory(payload) {
  const points = payload.points || [];
  ui.processHistorySection.hidden = false;
  ui.historyState.hidden = true;
  const latest = points.at(-1) || {};
  const utilizationValues = points.map((point) => Number(point.utilization_percent) || 0);
  const busySamples = points.filter((point) => (Number(point.process_count) || 0) > 0).length;
  const busyPercent = points.length ? (busySamples / points.length) * 100 : 0;
  const peakUtilization = utilizationValues.length ? Math.max(...utilizationValues) : 0;
  ui.historySubtitle.textContent = payload.raw_point_count
    ? `${payload.raw_point_count.toLocaleString()} saved samples · ${formatHistoryTimestamp(points[0]?.sampled_at)} to ${formatHistoryTimestamp(points.at(-1)?.sampled_at)}${payload.downsampled ? " · chart downsampled" : ""}`
    : `No snapshots saved in the selected ${historyHours}-hour range`;
  ui.historySummary.innerHTML = `
    <article>
      <span>Latest utilization</span>
      <strong>${Math.round(Number(latest.utilization_percent) || 0)}%</strong>
    </article>
    <article>
      <span>Peak utilization</span>
      <strong>${Math.round(peakUtilization)}%</strong>
    </article>
    <article>
      <span>Latest memory</span>
      <strong>${formatMemory(latest.memory_used_mib)}</strong>
    </article>
    <article>
      <span>Process-active samples</span>
      <strong>${busyPercent.toFixed(1)}%</strong>
    </article>`;
  const memoryMaximum = Math.max(
    1,
    ...points.map((point) => (Number(point.memory_total_mib) || 0) / 1024),
  );
  const powerMaximum = Math.max(
    1,
    ...points.map((point) => Number(point.power_limit_w) || 0),
  );
  ui.historyCharts.innerHTML = [
    {
      title: "GPU utilization",
      field: "utilization_percent",
      className: "utilization-history",
      transform: (value) => value,
      minimum: 0,
      maximum: 100,
      axis: (value) => `${Math.round(value)}%`,
      value: (value) => `${Math.round(value)}%`,
    },
    {
      title: "GPU memory",
      field: "memory_used_mib",
      className: "memory-history",
      transform: (value) => value / 1024,
      minimum: 0,
      maximum: memoryMaximum,
      axis: (value) => `${Math.round(value)}G`,
      value: (value) => `${value.toFixed(1)} GB`,
    },
    {
      title: "Temperature",
      field: "temperature_c",
      className: "temperature-history",
      transform: (value) => value,
      minimum: 20,
      maximum: Math.max(90, ...points.map((point) => Number(point.temperature_c) || 0)),
      axis: (value) => `${Math.round(value)}°`,
      value: (value) => `${Math.round(value)}°C`,
    },
    {
      title: "Power draw",
      field: "power_draw_w",
      className: "power-history",
      transform: (value) => value,
      minimum: 0,
      maximum: powerMaximum,
      axis: (value) => `${Math.round(value)}W`,
      value: (value) => `${Math.round(value)} W`,
    },
  ].map((config) => historyChart(points, config)).join("");

  renderProcessHistory(payload.processes, "this GPU");
}

function renderAggregateHistory(payload) {
  const points = payload.points || [];
  ui.processHistorySection.hidden = false;
  const latest = payload.summary?.latest || points.at(-1) || {};
  const rangeSummary = payload.summary || {};
  const scopeLabel = payload.scope === "fleet"
    ? "the whole fleet"
    : payload.server_id;
  ui.historyState.hidden = true;
  ui.historySubtitle.textContent = payload.raw_point_count
    ? `${payload.raw_point_count.toLocaleString()} aggregate samples · ${formatHistoryTimestamp(points[0]?.sampled_at)} to ${formatHistoryTimestamp(points.at(-1)?.sampled_at)}${payload.downsampled ? " · chart downsampled" : ""}`
    : `No aggregate snapshots saved in the selected ${historyHours}-hour range`;
  ui.historySummary.innerHTML = `
    <article>
      <span>Latest average utilization</span>
      <strong>${(Number(latest.utilization_average_percent) || 0).toFixed(1)}%</strong>
    </article>
    <article>
      <span>Range peak GPU utilization</span>
      <strong>${Math.round(Number(rangeSummary.peak_utilization_percent) || 0)}%</strong>
    </article>
    <article>
      <span>Latest memory ratio</span>
      <strong>${(Number(latest.memory_utilization_percent) || 0).toFixed(1)}%</strong>
    </article>
    <article>
      <span>Latest active processes</span>
      <strong>${Math.round(Number(latest.process_count) || 0)}</strong>
    </article>`;
  const powerMaximum = Math.max(
    1,
    ...points.map((point) => Number(point.power_limit_w) || 0),
  );
  ui.historyCharts.innerHTML = [
    {
      title: "Average GPU utilization",
      field: "utilization_average_percent",
      className: "utilization-history",
      transform: (value) => value,
      minimum: 0,
      maximum: 100,
      axis: (value) => `${Math.round(value)}%`,
      value: (value) => `${value.toFixed(1)}%`,
    },
    {
      title: "Aggregate GPU memory ratio",
      field: "memory_utilization_percent",
      className: "memory-history",
      transform: (value) => value,
      minimum: 0,
      maximum: 100,
      axis: (value) => `${Math.round(value)}%`,
      value: (value) => `${value.toFixed(1)}%`,
    },
    {
      title: "Average GPU temperature",
      field: "temperature_average_c",
      className: "temperature-history",
      transform: (value) => value,
      minimum: 20,
      maximum: Math.max(
        90,
        ...points.map((point) => Number(point.temperature_peak_c) || 0),
      ),
      axis: (value) => `${Math.round(value)}°`,
      value: (value) => `${value.toFixed(1)}°C`,
    },
    {
      title: "Total GPU power draw",
      field: "power_draw_w",
      className: "power-history",
      transform: (value) => value,
      minimum: 0,
      maximum: powerMaximum,
      axis: (value) => `${Math.round(value)}W`,
      value: (value) => `${Math.round(value)} W`,
    },
  ].map((config) => historyChart(points, config)).join("");
  renderProcessHistory(payload.processes, scopeLabel);
}

function renderStorageHistory(payload) {
  const points = payload.points || [];
  const latest = payload.summary?.latest || points.at(-1) || {};
  const shared = payload.kind === "shared";
  const scale = shared ? 1_000_000_000_000 : 1024 ** 3;
  const unit = shared ? "TB" : "GiB";
  const formatCapacity = (bytes) => shared
    ? formatDecimalTerabytes(bytes)
    : formatBytes(bytes);
  const changeBytes = Number(payload.summary?.change_bytes) || 0;
  const changePrefix = changeBytes > 0 ? "+" : changeBytes < 0 ? "−" : "";
  const changeLabel = `${changePrefix}${formatCapacity(Math.abs(changeBytes))}`;
  const capacityMaximum = shared
    ? Number(latest.policy_limit_bytes) || 40_000_000_000_000
    : Number(latest.total_bytes) || 1;
  ui.processHistorySection.hidden = true;
  ui.historyState.hidden = true;
  ui.historySubtitle.textContent = payload.raw_point_count
    ? `${payload.raw_point_count.toLocaleString()} storage samples · ${formatHistoryTimestamp(points[0]?.sampled_at)} to ${formatHistoryTimestamp(points.at(-1)?.sampled_at)}${payload.downsampled ? " · chart downsampled" : ""}`
    : `No storage snapshots saved in the selected ${historyHours}-hour range`;
  ui.historySummary.innerHTML = `
    <article>
      <span>Latest used</span>
      <strong>${formatCapacity(latest.used_bytes)}</strong>
    </article>
    <article>
      <span>${shared ? "40 TB budget remaining" : "Latest available"}</span>
      <strong>${formatCapacity(latest.effective_remaining_bytes)}</strong>
    </article>
    <article>
      <span>${shared ? "Budget utilization" : "Filesystem utilization"}</span>
      <strong>${(Number(latest.effective_used_percent) || 0).toFixed(1)}%</strong>
    </article>
    <article>
      <span>Change in selected range</span>
      <strong>${changeLabel}</strong>
    </article>`;
  ui.historyCharts.innerHTML = [
    {
      title: shared ? "Shared storage used" : "System disk used",
      field: "used_bytes",
      className: "memory-history",
      transform: (value) => value / scale,
      minimum: 0,
      maximum: capacityMaximum / scale,
      axis: (value) => `${value.toFixed(value >= 10 ? 0 : 1)}${unit}`,
      value: (value) => `${value.toFixed(value >= 10 ? 1 : 2)} ${unit}`,
    },
    {
      title: shared ? "40 TB budget utilization" : "Filesystem utilization",
      field: "effective_used_percent",
      className: "utilization-history",
      transform: (value) => value,
      minimum: 0,
      maximum: 100,
      axis: (value) => `${Math.round(value)}%`,
      value: (value) => `${value.toFixed(1)}%`,
    },
    {
      title: shared ? "40 TB budget remaining" : "Available capacity",
      field: "effective_remaining_bytes",
      className: "temperature-history",
      transform: (value) => value / scale,
      minimum: 0,
      maximum: capacityMaximum / scale,
      axis: (value) => `${value.toFixed(value >= 10 ? 0 : 1)}${unit}`,
      value: (value) => `${value.toFixed(value >= 10 ? 1 : 2)} ${unit}`,
    },
  ].map((config) => historyChart(points, config)).join("");
}

async function loadHistory() {
  if (!historySelection) return;
  const requestId = ++historyRequestId;
  ui.historyState.hidden = false;
  ui.historyState.textContent = "Loading saved history…";
  ui.historySummary.innerHTML = "";
  ui.historyCharts.innerHTML = "";
  ui.historyProcessList.innerHTML = "";
  ui.historyProcessCount.textContent = "—";
  ui.historyRefresh.disabled = true;
  try {
    const query = new URLSearchParams({
      scope: historySelection.scope,
      hours: String(historyHours),
    });
    if (historySelection.serverId) {
      query.set("server_id", historySelection.serverId);
    }
    if (historySelection.scope === "gpu") {
      query.set("gpu_index", String(historySelection.gpuIndex));
    }
    if (historySelection.scope === "storage") {
      query.set("mountpoint", historySelection.mountpoint);
    }
    const response = await fetch(`/api/history?${query}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (requestId === historyRequestId) {
      if (payload.scope === "gpu") renderGpuHistory(payload);
      else if (payload.scope === "storage") renderStorageHistory(payload);
      else renderAggregateHistory(payload);
    }
  } catch (error) {
    if (requestId === historyRequestId) {
      ui.historyState.hidden = false;
      ui.historyState.textContent = `Could not load history: ${error.message}`;
    }
  } finally {
    if (requestId === historyRequestId) ui.historyRefresh.disabled = false;
  }
}

function openHistory(button) {
  const scope = button.dataset.historyScope || "gpu";
  historySelection = {
    scope,
    serverId: button.dataset.serverId,
    gpuIndex: scope === "gpu" ? Number(button.dataset.gpuIndex) : null,
    gpuName: button.dataset.gpuName,
    serverLabel: button.dataset.serverLabel,
    mountpoint: button.dataset.mountpoint,
    storageKind: button.dataset.storageKind,
    volumeLabel: button.dataset.volumeLabel,
  };
  historyHours = 24;
  ui.historyRange.querySelectorAll("button[data-hours]").forEach((candidate) => {
    candidate.classList.toggle("active", candidate.dataset.hours === "24");
  });
  if (scope === "fleet") {
    ui.historyEyebrow.textContent = "PERSISTENT AGGREGATE HISTORY";
    ui.historyTitle.textContent = "Whole-fleet history";
    ui.historySubtitle.textContent = "Aggregate across both GPU servers";
  } else if (scope === "server") {
    ui.historyEyebrow.textContent = "PERSISTENT AGGREGATE HISTORY";
    ui.historyTitle.textContent = `${historySelection.serverLabel} history`;
    ui.historySubtitle.textContent = `Aggregate across 8 GPUs on ${historySelection.serverId}`;
  } else if (scope === "storage") {
    ui.historyEyebrow.textContent = "PERSISTENT STORAGE HISTORY";
    ui.historyTitle.textContent = `${historySelection.serverLabel} · ${historySelection.volumeLabel}`;
    ui.historySubtitle.textContent = historySelection.storageKind === "shared"
      ? `${historySelection.mountpoint} · shared NFS · 40 TB operating limit`
      : `${historySelection.mountpoint} · local filesystem`;
    ui.processHistorySection.hidden = true;
  } else {
    ui.historyEyebrow.textContent = "PERSISTENT GPU HISTORY";
    ui.historyTitle.textContent = `${historySelection.serverId} · GPU ${historySelection.gpuIndex}`;
    ui.historySubtitle.textContent = historySelection.gpuName;
  }
  ui.historyModal.hidden = false;
  document.body.classList.add("modal-open");
  loadHistory();
}

function closeGpuHistory() {
  historyRequestId += 1;
  ui.historyModal.hidden = true;
  document.body.classList.remove("modal-open");
  historySelection = null;
}

async function loadStatus({ quiet = false } = {}) {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    if (!quiet) showToast("Fleet status synchronized");
  } catch (error) {
    ui.liveState.classList.remove("online");
    ui.liveLabel.textContent = "Monitor unavailable";
    if (!quiet) showToast(`Could not refresh: ${error.message}`);
  }
}

async function manualRefresh() {
  ui.refreshButton.disabled = true;
  ui.refreshButton.classList.add("loading");
  try {
    await fetch("/api/refresh", { method: "POST" });
    await new Promise((resolve) => setTimeout(resolve, 700));
    await loadStatus();
  } finally {
    ui.refreshButton.disabled = false;
    ui.refreshButton.classList.remove("loading");
  }
}

ui.refreshButton.addEventListener("click", manualRefresh);
ui.servers.addEventListener("click", (event) => {
  const button = event.target.closest(
    "button.history-button, button.scope-history-button, button.storage-history-button",
  );
  if (button) openHistory(button);
});
ui.fleetHistoryButton.addEventListener("click", () => {
  openHistory(ui.fleetHistoryButton);
});
ui.historyRange.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-hours]");
  if (!button) return;
  historyHours = Number(button.dataset.hours);
  ui.historyRange.querySelectorAll("button[data-hours]").forEach((candidate) => {
    candidate.classList.toggle("active", candidate === button);
  });
  loadHistory();
});
ui.historyRefresh.addEventListener("click", loadHistory);
ui.historyModal.addEventListener("click", (event) => {
  if (event.target.closest("[data-history-close]")) closeGpuHistory();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !ui.historyModal.hidden) closeGpuHistory();
});
ui.viewToggle.addEventListener("click", () => {
  cleanView = !cleanView;
  try {
    window.localStorage.setItem(viewStorageKey, cleanView ? "clean" : "detailed");
  } catch (_error) {
    // The mode still works when browser storage is unavailable.
  }
  applyViewMode({ animate: true });
});
ui.filterGroup.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  activeFilter = button.dataset.filter;
  ui.filterGroup.querySelectorAll("button").forEach((candidate) => {
    candidate.classList.toggle("active", candidate === button);
  });
  applyFilter();
});

applyViewMode();
loadStatus({ quiet: true });
refreshTimer = setInterval(() => loadStatus({ quiet: true }), 5000);

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) loadStatus({ quiet: true });
});

window.addEventListener("beforeunload", () => clearInterval(refreshTimer));
