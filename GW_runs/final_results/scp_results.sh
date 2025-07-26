#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <GW_EVENT_ID>"
  exit 1
fi

gw="$1"

for id in bns nsbh default; do
  scp twouters@nikhef:/data/gravwav/twouters/projects/eos_source_classification/eos_source_classification/GW_runs/final_results/$gw/$id/${id}_result.json $gw/$id/${id}_result.json
done
