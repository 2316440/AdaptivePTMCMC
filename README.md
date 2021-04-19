## Adaptive PTMCMC algorithm (Honours Project)

This repository contains the code written for the purposes of my Honours Project, whose aim was to design an adaptive PTMCMC algorithm 
for compact binary gravitational wave parameter estimation.

ptmcmc.py contains the adaptive (AP) PTMCMC algorithm designed as part of the project - this can be used on any distribution. 
It passed the K-S test on a unimodal correlated Gaussian (9D), bimodal uncorrelated Gaussian (9D), Rosenbrock's banana function (3D) at 5% significance level.

The rest of the files include routines and classes used for simulaing BBH injections and recovering the data likelihood function. These include:
- gw.py - simulating frequency domain waveforms, applying antenna patterns and time delays, astrophysically motivated priors, etc.
- interferometer.py - classes used for storing frequency domain strain, noise generation, signal injection, and likelihood calculation 
  
...and a Jupyter notebook with a demo parameter estimation on a simulated BBH injection (parameter estimation.ipynb) which uses:
- data (folder) - H1 and L1 O3 noise ASDs (source: Soni, S. (LIGO Scientific Collaboration), LIGO-G1900992 and LIGO-G1900993 documents (2019))
- distance_given_snr.p - interpolated function that finds distance corresponding to a chosen SNR (for the fixed set of injection parameters used in the demo)
  
