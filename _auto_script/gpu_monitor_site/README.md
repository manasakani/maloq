# SC26 GPU fleet monitor

This is a private, read-only dashboard for the two SC26 GPU servers. Server 1
collects its local NVIDIA status and reads server 2 through the existing
`scp-gpu-2` SSH configuration every five seconds.

## Start and stop

Use the canonical absolute launcher path:

```bash
/dataset/seongsu/shared-home/workspace/project/_auto_script/gpu_monitor_site/run_gpu_monitor_site.sh start
/dataset/seongsu/shared-home/workspace/project/_auto_script/gpu_monitor_site/run_gpu_monitor_site.sh status
/dataset/seongsu/shared-home/workspace/project/_auto_script/gpu_monitor_site/run_gpu_monitor_site.sh logs
/dataset/seongsu/shared-home/workspace/project/_auto_script/gpu_monitor_site/run_gpu_monitor_site.sh stop
```

The monitor runs in the persistent tmux session `sc26-gpu-monitor-site`.
Runtime logs are stored at the absolute path:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/gpu-monitor-site/server.log
```

Persistent monitoring history is stored in:

```text
/dataset/seongsu/shared-home/workspace/project/outputs/gpu-monitor-site/history.sqlite3
```

The live dashboard still refreshes every five seconds. A complete snapshot is
committed to SQLite every 60 seconds so history remains useful without producing
excessive NFS writes. The append-only history includes server reachability,
GPU metrics, GPU process users and PIDs, local/shared storage, and the 40 TB
policy calculation. There is no automatic retention deletion. Recent GPU
history is restored from SQLite when the monitor restarts.

Select **History** on any GPU card to open its persistent metrics. The viewer
supports 1-hour, 6-hour, 24-hour, and 7-day ranges and plots utilization, GPU
memory, temperature, and power. It also lists the users, commands, PIDs,
first/last-seen times, and peak GPU memory for processes observed on that GPU in
the selected range. Long ranges are bounded to 720 chart points while the raw
one-minute records remain unchanged in SQLite.

Use **Server history** in either server header to aggregate that server's eight
GPUs, or **Fleet history** beside Current aggregate to combine both servers.
These views plot mean GPU utilization, aggregate memory ratio, mean
temperature, and total power, while their range summaries retain peak
utilization and process activity. Offline cached samples are excluded so they
are not mistaken for live measurements.

Each **System disk** and **Shared dataset** card also has a **History** button.
Storage history plots used capacity, utilization percentage, and remaining
capacity, and reports the net change in the selected range. Local disks use
their physical filesystem capacity; shared `/dataset` uses the 40 TB decimal
operating budget for percentage and remaining capacity while retaining the
larger physical NFS total as source metadata.

## Open the site

The HTTP server intentionally listens only on server 1 localhost. From a
machine with the `scp-gpu-1` SSH alias, create a private tunnel:

```bash
ssh -N -L 8787:127.0.0.1:8787 scp-gpu-1
```

Then open:

```text
http://127.0.0.1:8787
```

This avoids exposing process and utilization metadata on an unauthenticated
network port. The dashboard refreshes automatically and provides a manual
refresh button.

Use **Clean view** in the top bar to fit each server's eight GPUs into a compact
overview. It keeps utilization, memory, temperature, power, state, and active
process counts visible while hiding the large introduction, history graphs, and
process details. **Detailed view** restores the full cards. The selected mode is
remembered in that browser. Supported browsers use a native View Transition;
the fallback uses a short fade and the introductory section collapses smoothly.

Every GPU memory bar shows the exact used/total ratio and percentage. The
**Current aggregate** section summarizes the whole live fleet: mean and peak GPU
utilization, total memory ratio, mean/peak temperature, active compute-process
count, and unique active users. These are current five-second samples, not
historical averages.

Each server panel also shows local system-disk and shared `/dataset` capacity.
The shared volume refers to the same NFS storage on both servers and is measured
against the 40 TB decimal operating limit rather than the larger physical NFS
capacity. The card shows budget used and remaining, changes to attention at 80%,
and changes to exceeded at 100%. Select **Compute processes** on any GPU card to
reveal every compute process currently reported by NVIDIA, including its user,
command, PID, elapsed time, and GPU memory. Open process panels remain open
across automatic refreshes.

## Collected data

- GPU utilization and memory use
- local system-disk capacity and shared-dataset usage against the 40 TB limit
- temperature, power draw/limit, P-state, and compute mode
- compute PID, owning user, elapsed time, process name, and GPU memory
- server reachability, collection latency, and last successful sample
- short live utilization graphs backed by persistent restart history
- append-only one-minute SQLite snapshots for long-term tracking

All operations are read-only. If server 2 becomes unreachable, the dashboard
marks it offline and retains its last successful sample until connectivity
returns.
