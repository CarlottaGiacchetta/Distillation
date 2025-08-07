#!/bin/bash

GPU_ID=4;

mkdir -p logs

# Timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
logfile="logs/run_${timestamp}.log"
smilog="logs/nvidia_smi_${timestamp}.log"


echo "Logging to $logfile"
echo "Logging nvidia-smi to $smilog"

# Logging nvidia-smi ogni 5s
(
  while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S')" >> "$smilog"
    nvidia-smi >> "$smilog"
    echo "" >> "$smilog"
    sleep 5
  done
) &
SMI_PID=$!
SMI_PID=$!


docker run --rm --runtime=nvidia --name='class-cgiacchetta-rsde-'${GPU_ID} -e CUDA_VISIBLE_DEVICES=$GPU_ID --ipc=host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
--ulimit memlock=-1 --ulimit stack=67108864 -t --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) class-cgiacchetta-rsde4:latest \
2>&1 | tee "$logfile"

# Ferma logging nvidia-smi
kill $SMI_PID

echo "Finished. Logs saved to:"
echo "  - Docker log: $logfile"
echo "  - NVIDIA-SMI log: $smilog"