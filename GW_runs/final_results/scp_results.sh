for gw in GW170817 GW190425 GW230529; do
  for id in bns nsbh default; do
    scp twouters@nikhef:/data/gravwav/twouters/projects/eos_source_classification/eos_source_classification/GW_runs/final_results/$gw/$id/${id}_result.json $gw/$id/${id}_result.json
  done
done
