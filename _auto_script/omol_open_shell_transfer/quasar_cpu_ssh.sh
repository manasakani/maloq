#!/usr/bin/env bash
set -euo pipefail

exec ssh \
  -T \
  -o BatchMode=yes \
  -o ConnectTimeout=30 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=4 \
  -o StrictHostKeyChecking=accept-new \
  -o "ProxyCommand=env KUBECONFIG=/dataset/kubeconfig.yaml /home/gpuuser/.local/bin/kubectl exec -i --namespace=p-material-foundation pod/quasar-cpu-0 -- nc 127.0.0.1 2222" \
  "$@"
