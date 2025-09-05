#!/bin/bash

# Submit all bilby_pipe configurations for EOS source classification runs
# 
# This script finds all config.ini files in the generated directory structure
# and runs the complete bilby_pipe workflow for each configuration:
# 1. bilby_pipe config.ini
# 2. fix_submit_files outdir/submit  
# 3. sbatch outdir/submit/slurm_pe_master.sh

set -e  # Exit on any error

# Configuration
BASE_DIR="/work/wouters/neural_priors_paper_runs"
LOG_FILE="bilby_pipe_submission_$(date +%Y%m%d_%H%M%S).log"

# Function to run a command 
run_command() {
    local cmd="$1"
    local cwd="$2"
    
    echo "Running: $cmd in $cwd"
    if [[ -n "$cwd" ]]; then
        cd "$cwd"
    fi
    
    if timeout 300 bash -c "$cmd" >> "$LOG_FILE" 2>&1; then
        echo "✓ Completed: $cmd"
        return 0
    else
        local exit_code=$?
        echo "✗ Failed: $cmd (exit code: $exit_code)"
        return $exit_code
    fi
}

# Function to extract outdir from config.ini
get_outdir() {
    local config_file="$1"
    
    if [[ -f "$config_file" ]]; then
        grep "^outdir" "$config_file" | head -1 | cut -d'=' -f2 | tr -d ' '
    else
        echo ""
    fi
}

# Function to process a single config file
process_config() {
    local config_file="$1"
    local config_dir=$(dirname "$config_file")
    local config_name=${config_file#$BASE_DIR/}
    
    echo ""
    echo "================================================================================"
    echo "Processing: $config_name"
    echo "Directory: $config_dir"
    echo "================================================================================"
    
    # Step 1: Run bilby_pipe config.ini
    if ! run_command "bilby_pipe config.ini" "$config_dir"; then
        echo "bilby_pipe failed for $config_name"
        return 1
    fi
    
    # Step 2: Get output directory and fix submit files
    local outdir=$(get_outdir "$config_file")
    
    if [[ -z "$outdir" ]]; then
        echo "Could not determine outdir for $config_name"
        return 1
    fi
    
    local submit_dir="$config_dir/$outdir/submit"
    
    if ! run_command "fix_submit_files $submit_dir" "$config_dir"; then
        echo "fix_submit_files failed for $config_name"
        return 1
    fi
    
    # Step 3: Submit to SLURM
    local slurm_script="$submit_dir/slurm_pe_master.sh"
    
    if ! run_command "sbatch $slurm_script" "$config_dir"; then
        echo "sbatch submission failed for $config_name"
        return 1
    fi
    
    echo "✓ Successfully processed $config_name"
    return 0
}

# Main execution
echo "Starting bilby_pipe submission script"
echo "Base directory: $BASE_DIR"
echo "Log file: $LOG_FILE"

# Check if base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Base directory does not exist: $BASE_DIR"
    exit 1
fi

# Find all config.ini files
echo "Searching for config.ini files..."

config_files=()
while IFS= read -r -d '' file; do
    config_files+=("$file")
done < <(find "$BASE_DIR" -name "config.ini" -type f -print0 | sort -z)

total_configs=${#config_files[@]}

if [[ $total_configs -eq 0 ]]; then
    echo "No config.ini files found!"
    exit 1
fi

echo "Found $total_configs config.ini files"

# Process each config file
successful=0
failed=0
current=0

for config_file in "${config_files[@]}"; do
    current=$((current + 1))
    echo ""
    echo "Progress: $current/$total_configs"
    
    if process_config "$config_file"; then
        successful=$((successful + 1))
    else
        failed=$((failed + 1))
        echo "Stopping due to failure"
        break
    fi
done

# Summary
echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Total configs found: $total_configs"
echo "Successfully processed: $successful"
echo "Failed: $failed"
echo "Log file: $LOG_FILE"

if [[ $failed -gt 0 ]]; then
    echo "Some configurations failed. Check the log for details."
    exit 1
else
    echo "All configurations processed successfully!"
    exit 0
fi