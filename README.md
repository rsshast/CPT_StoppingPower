# Charged Particle Transport Models

## Comparison point for the Jaybenne CPT Code Suite
Analytic models for alpha particle continuous slowing-down transport models

## Three models implimented:
-  Fourier: stopping power modeled as a fourier series, isotropic emission in the geometry
-  BC: constant stopping power, isotropic emission on the boundary
-  Li-Petrasso: Time-dependent Li-Petrasso stopping power model, isotropic emission in the geometry

## Usage
After cloning the repository...
-  mkdir fluxes
-  cd fluxes
-  mkdir bc fourier lipetrasso
-  sbatch batch_spower.sh
    - (note: make sure you can load ffmpeg)
