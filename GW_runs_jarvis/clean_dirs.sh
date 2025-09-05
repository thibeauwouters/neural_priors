#!/bin/bash
# Clean the runs directory

if [ -d "runs" ]; then
    rm -rf runs/*
    echo "Cleaned runs directory"
else
    echo "runs directory does not exist"
fi