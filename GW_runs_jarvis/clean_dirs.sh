#!/bin/bash
# Clean the runs directory

if [ -d "/work/wouters/neural_priors_paper_runs" ]; then
    find /work/wouters/neural_priors_paper_runs/* -maxdepth 0 -type d -exec rm -rf {} +
    echo "Cleaned subdirectories in runs directory"
else
    echo "runs directory does not exist"
fi