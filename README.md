# eos_source_classification

## Bilby code

We adapted the bilby source code for this project, the branch used can be found [here](https://github.com/ThibeauWouters/bilby/tree/eos_source_classification). 

## data

Some datasets (GWF files, EOSs) used in the inferences and training.

- GW170817: same files as used in [TurboPE-BNS paper](https://github.com/ThibeauWouters/TurboPE-BNS/tree/main/real_events), see [Zenodo](https://zenodo.org/records/10991918)
- GW190425: same files as used in [TurboPE-BNS paper](https://github.com/ThibeauWouters/TurboPE-BNS/tree/main/real_events), see [Zenodo](https://zenodo.org/records/10991918)
- GW190814: data is fetched as follows: TODO:
- GW230529: data is fetched as follows: TODO:

Maximum likelihood parameters for relative binning are taken from other PE samples:
- GW1701817: [here](https://dcc.ligo.org/LIGO-P1800061/public). 
- GW190245: Previous Jim PE run
- GW230529: [Zenodo release](https://zenodo.org/records/10845779)

## NFprior

Data to store training scripts and models for the NF priors.

## GW_runs

Gravitational wave runs, without ROQ. 

- `final_results`: This directory stores copies of the GW runs, and these samples will be accessed by the postprocessing scripts. Use `copy_results.py` there to create a new backup (e.g. after some reruns with new priors are done etc). 