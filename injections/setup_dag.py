#!/usr/bin/env python3
"""
Script to automate creating DAG files for given rundirs

Usage:
    python setup_and_submit_injection.py --run-dir GW170817_bns_jester --eos-samples-name radio
"""

import os
import argparse
import subprocess
import sys


def create_subdirectories(run_dir, priors=['bns', 'nsbh', 'default']):
    """Create subdirectories for each prior in the injection directory."""
    for prior in priors:
        subdir = os.path.join(run_dir, prior)
        os.makedirs(subdir, exist_ok=True)
        print(f"Created directory: {subdir}")


def modify_dag_file(run_dir, eos_samples_name='radio'):
    """
    Modifies and saves the template DAG file to use the correct template and paths.
    """
    
    # Put it in a separate dir, to separate all aux dag files from our run_dir
    dag_dir = os.path.join(run_dir, 'dag')
    os.makedirs(dag_dir, exist_ok=True)
    dag_file = os.path.join(dag_dir, 'run.dag')
    
    # Get absolute path to analysis.sub
    sub_path = os.path.abspath('analysis.sub')
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Template file not found: {sub_path}")
    
    # Create new DAG content
    dag_content = f"""JOB run_a {sub_path}
VARS run_a run_dir="{run_dir}" prior_name="bns" eos_samples_name="{eos_samples_name}"
JOB run_b {sub_path}
VARS run_b run_dir="{run_dir}" prior_name="nsbh" eos_samples_name="{eos_samples_name}"
JOB run_c {sub_path}
VARS run_c run_dir="{run_dir}" prior_name="default" eos_samples_name="{eos_samples_name}"
"""
    
    # Write the modified DAG file
    with open(dag_file, 'w') as f:
        f.write(dag_content)
    
    print(f"Written DAG file: {dag_file}")
    return dag_file


# def submit_dag(dag_file, dry_run=False):
#     """Submit the DAG file using condor_submit_dag."""
#     if dry_run:
#         print(f"DRY RUN: Would submit DAG file: {dag_file}")
#         return
    
#     try:
#         cmd = ['condor_submit_dag', dag_file]
#         print(f"Submitting DAG with command: {' '.join(cmd)}")
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         print("DAG submission successful!")
#         print(result.stdout)
#     except subprocess.CalledProcessError as e:
#         print(f"Error submitting DAG: {e}")
#         print(f"stderr: {e.stderr}")
#         sys.exit(1)
#     except FileNotFoundError:
#         print("Error: condor_submit_dag command not found. Make sure HTCondor is installed and in PATH.")
#         sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Setup and submit injection analysis DAG')
    parser.add_argument('--run-dir', required=True,
                        help='Injection directory name (e.g., GW170817_bns_jester)')
    parser.add_argument('--eos-samples-name', default='radio',
                        help='EOS samples name (default: radio)')
    parser.add_argument('--seed', default='1234',
                        help='Random seed (default: 1234)')
    parser.add_argument('--relative-binning-delta', default='0.0005',
                        help='Relative binning delta (default: 0.0005)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually submitting')
    
    args = parser.parse_args()
    
    # Check if injection directory exists
    if not os.path.exists(args.run_dir):
        raise FileNotFoundError(f"Injection directory not found: {args.run_dir}")
    
    print(f"Setting up injection analysis for: {args.run_dir}")
    
    # Create subdirectories
    create_subdirectories(args.run_dir)
    
    # Modify DAG file
    dag_file = modify_dag_file(
        args.run_dir,
        args.eos_samples_name,
    )
    
    # # Submit DAG
    # submit_dag(dag_file, args.dry_run)
    # print("Setup and submission complete!")

if __name__ == '__main__':
    main()