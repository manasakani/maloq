# OMol transfer monitor

This durable SC26 monitor watches and, when necessary, restarts the three
resumable data jobs:

- OMol25 open-shell MALOQ ASE synchronization;
- OMol25 electrolyte completed-index LMDB synchronization;
- OMol_CSH public HDF5 download.

It polls every five minutes, writes a compact snapshot and append-only history
below `outputs/omol-transfer-monitor`, and exits only after every dataset's
exact verification succeeds. A stopped incomplete tmux job is restarted up to
20 times; the underlying download helpers also retain partial files and retry
transient transport failures.

```bash
/dataset/seongsu/shared-home/workspace/project/_auto_script/omol_transfer_monitor/monitor_omol_transfers.sh status
```

Completion is recorded at:

`/dataset/seongsu/shared-home/workspace/project/outputs/omol-transfer-monitor/COMPLETE`
