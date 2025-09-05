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
DRY_RUN=false
FILTER=""
SKIP_ERRORS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --base-dir DIR     Base directory containing config files (default: $BASE_DIR)"
    echo "  --dry-run          Show what would be run without executing commands"
    echo "  --filter STRING    Only process configs matching this substring"
    echo "  --skip-errors      Continue processing other configs if one fails"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dry-run                    # Show what would be done"
    echo "  $0 --filter GW170817           # Only process GW170817 configs"
    echo "  $0 --filter default             # Only process default runs"
    echo "  $0 --skip-errors                # Continue on failures"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --skip-errors)
            SKIP_ERRORS=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${BLUE}[$timestamp INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        SUCCESS)
            echo -e "${GREEN}[$timestamp SUCCESS]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        WARN)
            echo -e "${YELLOW}[$timestamp WARN]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        ERROR)
            echo -e "${RED}[$timestamp ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Function to run a command with logging
run_command() {
    local cmd="$1"
    local cwd="$2"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "[DRY RUN] Would run: $cmd"
        if [[ -n "$cwd" ]]; then
            log INFO "[DRY RUN] Working directory: $cwd"
        fi
        return 0
    fi
    
    log INFO "Running: $cmd"
    if [[ -n "$cwd" ]]; then
        log INFO "Working directory: $cwd"
        cd "$cwd"
    fi
    
    if timeout 300 bash -c "$cmd" >> "$LOG_FILE" 2>&1; then
        log SUCCESS "Command completed: $cmd"
        return 0
    else
        local exit_code=$?
        log ERROR "Command failed (exit code: $exit_code): $cmd"
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
    
    log INFO ""
    log INFO "================================================================================"
    log INFO "Processing: $config_name"
    log INFO "Directory: $config_dir"
    log INFO "================================================================================"
    
    # Step 1: Run bilby_pipe config.ini
    if ! run_command "bilby_pipe config.ini" "$config_dir"; then
        log ERROR "bilby_pipe failed for $config_name"
        return 1
    fi
    
    # Step 2: Get output directory and fix submit files
    local outdir
    if [[ "$DRY_RUN" == "true" ]]; then
        outdir="outdir_${config_name//\//_}"
        outdir=${outdir//config.ini/}
        outdir=${outdir%_}
    else
        outdir=$(get_outdir "$config_file")
    fi
    
    if [[ -z "$outdir" ]]; then
        log ERROR "Could not determine outdir for $config_name"
        return 1
    fi
    
    local submit_dir="$config_dir/$outdir/submit"
    
    if ! run_command "fix_submit_files $submit_dir" "$config_dir"; then
        log ERROR "fix_submit_files failed for $config_name"
        return 1
    fi
    
    # Step 3: Submit to SLURM
    local slurm_script="$submit_dir/slurm_pe_master.sh"
    
    if ! run_command "sbatch $slurm_script" "$config_dir"; then
        log ERROR "sbatch submission failed for $config_name"
        return 1
    fi
    
    log SUCCESS "Successfully processed $config_name"
    return 0
}

# Main execution
main() {
    log INFO "Starting bilby_pipe submission script"
    log INFO "Base directory: $BASE_DIR"
    log INFO "Log file: $LOG_FILE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log WARN "DRY RUN MODE - No commands will be executed"
    fi
    
    if [[ -n "$FILTER" ]]; then
        log INFO "Filter: $FILTER"
    fi
    
    # Check if base directory exists
    if [[ ! -d "$BASE_DIR" ]]; then
        log ERROR "Base directory does not exist: $BASE_DIR"
        exit 1
    fi
    
    # Find all config.ini files
    log INFO "Searching for config.ini files..."
    
    local config_files=()
    while IFS= read -r -d '' file; do
        if [[ -z "$FILTER" ]] || [[ "$file" == *"$FILTER"* ]]; then
            config_files+=("$file")
        fi
    done < <(find "$BASE_DIR" -name "config.ini" -type f -print0 | sort -z)
    
    local total_configs=${#config_files[@]}
    
    if [[ $total_configs -eq 0 ]]; then
        log ERROR "No config.ini files found!"
        if [[ -n "$FILTER" ]]; then
            log ERROR "Filter '$FILTER' may be too restrictive"
        fi
        exit 1
    fi
    
    log INFO "Found $total_configs config.ini files"
    
    # Process each config file
    local successful=0
    local failed=0
    local current=0
    
    for config_file in "${config_files[@]}"; do
        current=$((current + 1))
        log INFO ""
        log INFO "Progress: $current/$total_configs"
        
        if process_config "$config_file"; then
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
            
            if [[ "$SKIP_ERRORS" != "true" ]]; then
                log ERROR "Stopping due to failure (use --skip-errors to continue)"
                break
            fi
        fi
    done
    
    # Summary
    log INFO ""
    log INFO "================================================================================"
    log INFO "SUMMARY"
    log INFO "================================================================================"
    log INFO "Total configs found: $total_configs"
    log INFO "Successfully processed: $successful"
    log INFO "Failed: $failed"
    log INFO "Log file: $LOG_FILE"
    
    if [[ $failed -gt 0 ]]; then
        log WARN "Some configurations failed. Check the log for details."
        exit 1
    else
        log SUCCESS "All configurations processed successfully!"
        exit 0
    fi
}

# Run main function
main "$@"