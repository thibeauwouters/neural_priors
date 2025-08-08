# Cosmology Settings for Bilby Priors

## Current Status

The default bilby cosmology already matches Adrian's and Hauke's cosmology exactly:
- **H0**: 67.74 km/(Mpc·s)
- **Om0**: 0.3075  
- **Tcmb0**: 2.7255 K
- **Neff**: 3.046
- **m_nu**: [0.0, 0.0, 0.06] eV
- **Ob0**: 0.0486
- **Cosmology class**: FlatLambdaCDM

This is the **Planck15** cosmology (not Planck18). All runs use consistent cosmology settings.

## How to Explicitly Specify Cosmology in Prior Files

To guarantee consistency and make the cosmology explicit in your prior files, modify the `UniformComovingVolume` prior as follows:

```python
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Planck 2018 cosmology as used in arXiv:2311.07456
# H0: Hubble constant at z=0
# Om0: Matter density parameter at z=0  
# Tcmb0: Temperature of the CMB at z=0
# Neff: Number of effective neutrino species
# m_nu: Neutrino masses
# Ob0: Baryon density parameter at z=0
planck_cosmology = FlatLambdaCDM(
    H0=67.74 * u.km / u.s / u.Mpc,
    Om0=0.3075,
    Tcmb0=2.7255 * u.K,
    Neff=3.046,
    m_nu=[0.0, 0.0, 0.06] * u.eV,
    Ob0=0.0486
)

luminosity_distance = UniformComovingVolume(
    minimum=1.0, 
    maximum=75.0, 
    name='luminosity_distance', 
    latex_label='$D_L$',
    cosmology=planck_cosmology
)
```

## Reference

The cosmological parameters correspond to Planck 2018 results as used in the paper:
- arXiv:2311.07456 - "Constraining the neutron star equation of state using multi-band independent measurements of radii"

## Verification

Use `check_default_cosmology.py` and `report_priors.py` to verify that cosmologies match between default bilby settings and Adrian's analysis results.