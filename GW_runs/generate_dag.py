#!/usr/bin/env python3
"""
Script to generate DAG files for gravitational wave parameter estimation runs.
Loops over population types, prior types, and source types as specified.
"""

def generate_dag(gw_event, output_file, relative_binning_delta=1e-3):
    """Generate DAG file for the specified GW event."""
    
    population_types = ["uniform", "gaussian", "double_gaussian"]
    eos_samples_names = ["radio", "radio_chiEFT", "radio_chiEFT_NICER", gw_event]
    prior_names = ["bns", "nsbh"]
    
    job_counter = 0
    job_letters = []
    
    # Generate job letters (a, b, c, ..., z, aa, bb, etc.)
    def get_job_name(counter):
        if counter < 26:
            return chr(ord('a') + counter)
        else:
            base_letter = chr(ord('a') + (counter % 26))
            return base_letter * ((counter // 26) + 1)
    
    lines = []
    
    # Main loops over population_type, eos_samples_name, and prior_name
    for population_type in population_types:
        for eos_samples_name in eos_samples_names:
            for prior_name in prior_names:
                job_name = f"run_{get_job_name(job_counter)}"
                
                lines.append(f"JOB {job_name} /data/gravwav/twouters/projects/eos_source_classification/eos_source_classification/GW_runs/analysis.sub")
                lines.append(f"VARS {job_name} GW_event=\"{gw_event}\" population_type=\"{population_type}\" prior_name=\"{prior_name}\" eos_samples_name=\"{eos_samples_name}\" relative_binning_delta=\"{relative_binning_delta}\" seed=\"1234\"")
                
                job_counter += 1
    
    # Add the single run with uniform, radio, default
    job_name = f"run_{get_job_name(job_counter)}"
    lines.append(f"JOB {job_name} /data/gravwav/twouters/projects/eos_source_classification/eos_source_classification/GW_runs/analysis.sub")
    lines.append(f"VARS {job_name} GW_event=\"{gw_event}\" population_type=\"uniform\" prior_name=\"default\" eos_samples_name=\"radio\" relative_binning_delta=\"{relative_binning_delta}\" seed=\"1234\"")
    
    # Write to file
    with open(output_file, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"Generated DAG file: {output_file}")
    print(f"Total jobs: {job_counter + 1}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python generate_dag.py <GW_event> <output_file> [relative_binning_delta]")
        print("Example: python generate_dag.py GW170817 dag_files/run_GW170817_generated.dag")
        print("Example: python generate_dag.py GW170817 dag_files/run_GW170817_generated.dag 0.005")
        print("Default relative_binning_delta: 1e-3")
        sys.exit(1)
    
    gw_event = sys.argv[1]
    output_file = sys.argv[2]
    relative_binning_delta = float(sys.argv[3]) if len(sys.argv) == 4 else 1e-3
    
    generate_dag(gw_event, output_file, relative_binning_delta)