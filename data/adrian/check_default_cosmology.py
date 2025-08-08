#!/usr/bin/env python3
"""
Check the default cosmology used by bilby's UniformComovingVolume prior
"""
import bilby
import numpy as np

# Create a default UniformComovingVolume prior
default_prior = bilby.gw.prior.UniformComovingVolume(minimum=1.0, maximum=75.0, name='luminosity_distance')

print("Default bilby cosmology:")
print(f"  H0: {default_prior.cosmology.H0}")
print(f"  Om0: {default_prior.cosmology.Om0}")
print(f"  Tcmb0: {default_prior.cosmology.Tcmb0}")
print(f"  Neff: {default_prior.cosmology.Neff}")
print(f"  m_nu: {default_prior.cosmology.m_nu}")
print(f"  Ob0: {default_prior.cosmology.Ob0}")
print(f"  Cosmology class: {type(default_prior.cosmology).__name__}")

print("\nAdrian's cosmology (from report):")
print("  H0: 67.74 km / (Mpc s)")
print("  Om0: 0.3075")
print("  Tcmb0: 2.7255 K")
print("  Neff: 3.046")
print("  m_nu: [0.0, 0.0, 0.06] eV")
print("  Ob0: 0.0486")
print("  Cosmology class: FlatLambdaCDM")