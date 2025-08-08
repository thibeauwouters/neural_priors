#!/usr/bin/env python3
"""
Check what bilby's actual default cosmology is and compare with Planck15/Planck18
"""
import bilby
from astropy.cosmology import Planck15, Planck18

print("=== BILBY DEFAULT COSMOLOGY INVESTIGATION ===\n")

# Create a default UniformComovingVolume prior
default_prior = bilby.gw.prior.UniformComovingVolume(minimum=1.0, maximum=75.0, name='luminosity_distance')

print("Bilby default cosmology:")
print(f"  H0: {default_prior.cosmology.H0}")
print(f"  Om0: {default_prior.cosmology.Om0}")
print(f"  Tcmb0: {default_prior.cosmology.Tcmb0}")
print(f"  Neff: {default_prior.cosmology.Neff}")
print(f"  m_nu: {default_prior.cosmology.m_nu}")
print(f"  Ob0: {default_prior.cosmology.Ob0}")
print(f"  Class: {type(default_prior.cosmology).__name__}")
print(f"  String representation: {default_prior.cosmology}")
print()

print("Planck15 cosmology:")
print(f"  H0: {Planck15.H0}")
print(f"  Om0: {Planck15.Om0}")
print(f"  Tcmb0: {Planck15.Tcmb0}")
print(f"  Neff: {Planck15.Neff}")
print(f"  m_nu: {Planck15.m_nu}")
print(f"  Ob0: {Planck15.Ob0}")
print(f"  Class: {type(Planck15).__name__}")
print(f"  String representation: {Planck15}")
print()

print("Planck18 cosmology:")
print(f"  H0: {Planck18.H0}")
print(f"  Om0: {Planck18.Om0}")
print(f"  Tcmb0: {Planck18.Tcmb0}")
print(f"  Neff: {Planck18.Neff}")
print(f"  m_nu: {Planck18.m_nu}")
print(f"  Ob0: {Planck18.Ob0}")
print(f"  Class: {type(Planck18).__name__}")
print(f"  String representation: {Planck18}")
print()

print("=== COMPARISON ===")
print("Does bilby default match Planck15?")
print(f"  H0: {abs(default_prior.cosmology.H0.value - Planck15.H0.value) < 0.01}")
print(f"  Om0: {abs(default_prior.cosmology.Om0 - Planck15.Om0) < 0.001}")
print(f"  Ob0: {abs(default_prior.cosmology.Ob0 - Planck15.Ob0) < 0.001}")
print()

print("Does bilby default match Planck18?")
print(f"  H0: {abs(default_prior.cosmology.H0.value - Planck18.H0.value) < 0.01}")
print(f"  Om0: {abs(default_prior.cosmology.Om0 - Planck18.Om0) < 0.001}")
print(f"  Ob0: {abs(default_prior.cosmology.Ob0 - Planck18.Ob0) < 0.001}")