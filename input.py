# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.integrate import quad
from numba import njit, prange
import time

#model = 'fourier'
#model = 'bc'
model = 'lipetrasso'

# model agnostic params
np.random.seed(1)
nparts = 100
verbose = True
track_particles = True

# geometry
length, width = 10, 10 #cm
dx = .01
Area = dx * dx
eps = 1e-8
E0 = 3540 #keV
Eb = 200  #keV

# direction
num_ords = 64
theta_bins = np.linspace(0,2 * np.pi,num_ords+1)
d_theta = 2*np.pi / num_ords
mu, w = np.polynomial.legendre.leggauss(num_ords)

# energy groups
groups = 8
E_grid = np.exp(np.linspace(np.log(E0),np.log(Eb),groups+1))

# Fourier Series
if model == 'fourier':
    # Define a stopping power S, calculate the trajectory of the transported alpha
    swarm = 1e3 # particles per swarm
    K = 4 # source order

elif model == 'bc':
    mu = np.abs(mu[:num_ords//2])
    swarm = 3.5e2 # particles per swarm

elif model == 'lipetrasso':
    swarm = 1e22
    t_f = 1e-9 # s
    length, width = .01, .01 #m
    dx = .00005
    Area = dx * dx
    
