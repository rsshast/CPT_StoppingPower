import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.integrate import quad
from numba import njit, prange
import time
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os
from numba import njit

np.random.seed(42) # seed

@njit(cache=True)
def _jit_generate_batch(i, j, k, dx, n_particles):
    """JIT-compiled function to generate particles for a single cell."""
    p0 = np.empty((n_particles, 3), dtype=np.float64)
    dirs = np.empty((n_particles, 3), dtype=np.float64)

    for p in range(n_particles):
        p0[p, 0] = np.random.uniform(i * dx, (i + 1) * dx)
        p0[p, 1] = np.random.uniform(j * dx, (j + 1) * dx)
        p0[p, 2] = np.random.uniform(k * dx, (k + 1) * dx)

        mu = np.random.uniform(-1.0, 1.0)
        phi = np.random.uniform(0.0, 2 * np.pi)

        sin_theta = np.sqrt(1.0 - mu**2)
        dirs[p, 0] = sin_theta * np.cos(phi) # u
        dirs[p, 1] = sin_theta * np.sin(phi) # v
        dirs[p, 2] = mu                      # w

    return p0, dirs

@njit(cache=True)
def jit_min_d_boundary(x, y, z, u, v, w, I, J, K, dx):
    """JIT-compiled 3D boundary distance calculation."""
    x_min, x_max = I * dx, (I + 1) * dx
    y_min, y_max = J * dx, (J + 1) * dx
    z_min, z_max = K * dx, (K + 1) * dx

    if u > 0:   d_x = (x_max - x) / u
    elif u < 0: d_x = (x_min - x) / u
    else:       d_x = np.inf

    if v > 0:   d_y = (y_max - y) / v
    elif v < 0: d_y = (y_min - y) / v
    else:       d_y = np.inf

    if w > 0:   d_z = (z_max - z) / w
    elif w < 0: d_z = (z_min - z) / w
    else:       d_z = np.inf

    return min(d_x, d_y, d_z) + 1e-12

class LPSTP_3D():
    def __init__(self, nparts_initial, nparts_per_pulse, weight, pulses,
                    E0, Eb, Tbeta, length, width, height, dx, eps,
                    tf, dt, track_particles, verbose):
        self.hbar = 1.0546e-27
        self.kb = 1.380649e-16
        self.MeVtoERG = 1.60218e-6
        self.ERGtoMeV = 1 / self.MeVtoERG

        self.nparts_initial = nparts_initial
        self.nparts_per_pulse = nparts_per_pulse
        self.weight = weight
        self.track_particles = track_particles
        self.verbose = verbose

        self.length = length
        self.width = width
        self.height = height
        self.dx = dx
        self.Volume = self.length * self.width * self.height
        self.dV = dx * dx * dx
        self.eps = eps
        self.cmap = 'plasma'

        self.Z = 2
        self.m_alpha = 6.644657230e-24 
        self.m_beta = 9.109e-28 
        self.rho = 1 
        self.n_a  = self.rho / self.m_alpha
        self.n_e = self.Z * self.n_a

        self.e = 4.803204e-10
        self.erg = 1e7

        self.E0 = E0 * self.MeVtoERG  
        self.Eb = Eb * self.MeVtoERG 
        
        # Prevent Zeno's paradox with a 1 keV death threshold when Eb = 0.0
        self.E_cutoff = max(self.Eb, 1e-3 * self.MeVtoERG) 

        self.T_beta_0 = Tbeta
        self.E_beta = 1.5 * self.kb * self.T_beta_0 
        self.w2 = self.OmegaBeta2()
        self.cv = 1e8 
        self.sie = self.cv * (self.rho * self.dV)

        self.tf = tf
        self.dt = dt
        self.timesteps = np.linspace(0,self.tf,int(self.tf / self.dt)+1)

        self.S = np.zeros((int(self.length/self.dx), int(self.width/self.dx), int(self.height/self.dx)),dtype=float)
        self.phi = np.zeros((int(self.length/self.dx), int(self.width/self.dx), int(self.height/self.dx), 
                        self.timesteps.size),dtype=float)
        self.E_dep = np.zeros_like(self.phi)
        self.S_const = 1e-5

        self.npulses = pulses
        self.emis_time = self.tf / self.npulses
        n_cells_1d = int(self.length / self.dx)
        self.n_cells_total = n_cells_1d ** 3
        self.deaths_per_step = np.zeros(len(self.timesteps))

    def source_alphas(self, total_particles):
        if self.verbose:
            print(f"--> Sourcing {total_particles} simulated histories for this pulse.")

        n = int(self.length // self.dx)
        base_per_cell = total_particles // self.n_cells_total
        remainder = total_particles % self.n_cells_total
        
        counts = np.full(self.n_cells_total, base_per_cell)
        
        if remainder > 0:
            idx = np.random.choice(self.n_cells_total, remainder, replace=False)
            counts[idx] += 1

        count_idx = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    c = counts[count_idx]
                    count_idx += 1
                    if c > 0:
                        yield _jit_generate_batch(i, j, k, self.dx, c)

    def OmegaBeta2(self):
        return (4 * np.pi * self.n_e * self.e * self.e) / self.m_beta

    def tally_E_dep(self, I, J, K, t_idx, actual_E_loss):
        self.E_dep[I, J, K, t_idx] += actual_E_loss * self.weight

    def tally_phi(self, I, J, K, t_idx, ds):
        self.phi[I, J, K, t_idx] += ds * self.weight / self.dV

    def TransportAlphas(self, p0_batch, dir_batch, base_history_id, birth_time, lp=True):
        for p_idx in range(len(p0_batch)):
            particle = np.array([
                p0_batch[p_idx, 0], p0_batch[p_idx, 1], p0_batch[p_idx, 2],
                self.E0, birth_time,
            ])
            u, v, w = dir_batch[p_idx]
            history_id = base_history_id + p_idx

            if self.track_particles:
                self.particle_path.append([history_id, particle[0], particle[1], particle[2], particle[3], particle[4]])

            Transport = True
            while Transport:
                particle[0] %= self.length
                particle[1] %= self.width
                particle[2] %= self.height

                I, J, K = [min(int(particle[idx] / self.dx), int(dim / self.dx) - 1) 
                           for idx, dim in zip([0, 1, 2], [self.length, self.width, self.height])]

                #only comment out dmin for quadratic model
                dmin = jit_min_d_boundary(particle[0], particle[1], particle[2], u, v, w, I, J, K, self.dx)

                t_idx = min(int(particle[4] / self.dt + self.eps), len(self.timesteps) - 1)

                cumulative_heat = np.sum(self.E_dep[I, J, K, :t_idx + 1])
                local_T_beta = self.T_beta_0 + (cumulative_heat / self.sie)

                ## --- CONSTANT --- ##
                S_val = self.S_const
                ## --- QUADRATIC --- ##
                #tq = particle[4] - birth_time
                #S_val = self.S_const * (tq * tq) + 1e-30
                #S_val = max(self.S_const * (tq * tq), 1e-30)

                self.S[I, J, K] = S_val
                v_cell = np.sqrt(2 * particle[3] / self.m_alpha)
                current_cycle = int(particle[4] / self.dt + self.eps)
                time_remaining_in_step = ((current_cycle + 1) * self.dt) - particle[4]

                max_time_dist = v_cell * time_remaining_in_step
                max_dist_before_death = (particle[3] - self.E_cutoff) / S_val

                #actual_dist = min(max_dist_before_death, max_time_dist)
                actual_dist = min(dmin, max_dist_before_death, max_time_dist)
                actual_loss = S_val * actual_dist

                self.tally_E_dep(I, J, K, t_idx, actual_loss)
                #self.tally_phi(I, J, K, t_idx, dmin)

                particle[3] -= actual_loss
                particle[4] += actual_dist / v_cell

                if particle[3] <= self.E_cutoff * 1.0001:
                    particle[3] = self.Eb # Drop to exact background once cut off is reached
                    death_idx = min(int(particle[4] / self.dt), len(self.timesteps) - 1)
                    self.deaths_per_step[death_idx] += self.weight
                    Transport = False

                if particle[4] > self.timesteps[-1]:
                    Transport = False

                if Transport:
                    particle[0] += actual_dist * u
                    particle[1] += actual_dist * v
                    particle[2] += actual_dist * w

    def run(self):
        if self.track_particles: self.particle_path = []
        stt = time.time()
        global_history_id = 0

        pulse_target_times = [i * self.emis_time for i in range(self.npulses)]
        snapped_indices = [int(np.round(t / self.dt)) for t in pulse_target_times]

        for t_idx, current_time in enumerate(self.timesteps):
            if self.verbose or t_idx % 10 == 0:
                print(f"CYCLE {t_idx}, t = {(current_time*1e9):5g}ns")

            pulses_this_step = snapped_indices.count(t_idx)

            for p_idx in range(pulses_this_step):
                if t_idx == 0 and p_idx == 0:
                    particles_to_source = self.nparts_initial

            for p_idx in range(pulses_this_step):
                if t_idx == 0 and p_idx == 0:
                    particles_to_source = self.nparts_initial
                else:
                    # Match the discrete analytical array: max 10 particles per pulse
                    val = 10.0 * np.sin(np.pi * current_time / self.tf)
                    particles_to_source = int(np.floor(val))

                # Only source if there are actually particles to inject
                if particles_to_source > 0:
                    ptcl_batches = self.source_alphas(particles_to_source)
                    for _, (p0_batch, dir_batch) in enumerate(ptcl_batches):
                        self.TransportAlphas(p0_batch, dir_batch, global_history_id, current_time)
                        global_history_id += len(p0_batch)
#                else:
#                    particles_to_source = self.nparts_per_pulse
#
#                ptcl_batches = self.source_alphas(particles_to_source)
#                for _, (p0_batch, dir_batch) in enumerate(ptcl_batches):
#                    self.TransportAlphas(p0_batch, dir_batch, global_history_id, current_time)
#                    global_history_id += len(p0_batch)

        print(f"Total Transport Time: {np.round(time.time() - stt, 5)}s")
        print(f"Total Simulated Histories: {global_history_id}")

        #self.plot_average_kinetic_energy()
        self.plot_average_kinetic_energy_sine()

    def plot_average_kinetic_energy(self):
        stt = time.time()
        ns = 1e9
        print("Generating Average Kinetic Energy curve...")

        pulse_target_times = [i * self.emis_time for i in range(self.npulses)]
        snapped_indices = [int(np.round(t / self.dt)) for t in pulse_target_times]

        cumulative_particles = np.zeros(len(self.timesteps))
        current_particle_count = 0

        for t_idx in range(len(self.timesteps)):
            pulses_here = snapped_indices.count(t_idx)
            for p in range(pulses_here):
                if t_idx == 0 and p == 0:
                    current_particle_count += self.nparts_initial * self.weight
                else:
                    current_particle_count += self.nparts_per_pulse * self.weight
            cumulative_particles[t_idx] = current_particle_count

        total_initial_erg_array = cumulative_particles * self.E0
        e_dep_per_step_erg = np.sum(self.E_dep, axis=(0, 1, 2))
        e_dep_cumulative_erg = np.insert(np.cumsum(e_dep_per_step_erg), 0, 0)[:-1]
        
        cumulative_dead = np.cumsum(self.deaths_per_step)
        dead_energy_bank = cumulative_dead * self.E_cutoff

        # Active Energy & Active Particle Count
        e_kin_active_erg = total_initial_erg_array - e_dep_cumulative_erg - dead_energy_bank
        active_particles = cumulative_particles - cumulative_dead

        U_t_over_E0 = e_kin_active_erg / self.E0
        np.savetxt("quad_UT_case3.txt",U_t_over_E0) # save data
        #tau = 0.0013758674831213246
        tau = 8.68177578551766e-10
        t_over_tau = self.timesteps / tau

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Normalized Total KE
        ax.plot(t_over_tau, U_t_over_E0, label=r'$U(t)/E_0$ (Active Alphas)', color='blue', linewidth=2)

        ax.axvline(1.0, label=r'$\tau$', linestyle='-.', color='black')
        ax.set_title(f'Normalized Total Kinetic Energy of Active Population', fontsize=14)
        ax.set_xlabel(r'$t / \tau$', fontsize=12)
        ax.set_ylabel(r'$U(t) / E_0$', fontsize=12)
        ax.set_xlim(left=0, right=2.0)
        ax.set_ylim(bottom=0)

        ax.grid(True, which='major', linestyle='-', color='lightgray', linewidth=1)
        ax.legend(loc='center right', fontsize=11, frameon=True, edgecolor='lightgray')

        os.makedirs('fluxes/npp', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'fluxes/npp/quad_ke_multipulse_dx_{self.dx}_pulses_{self.npulses}.png', dpi=300)

        print(f"Average KE curve saved in {np.round(time.time() - stt, 3)}s.")

    def plot_average_kinetic_energy_sine(self):
        stt = time.time()
        ns = 1e9
        print("Generating Total Active Kinetic Energy curve...")

        pulse_target_times = [i * self.emis_time for i in range(self.npulses)]
        snapped_indices = [int(np.round(t / self.dt)) for t in pulse_target_times]

        cumulative_particles = np.zeros(len(self.timesteps))
        current_particle_count = 0

        # Accurately track the time-dependent injection for our analytic array
        for t_idx in range(len(self.timesteps)):
            pulses_here = snapped_indices.count(t_idx)
            current_time = self.timesteps[t_idx]

            for p in range(pulses_here):
                if t_idx == 0 and p == 0:
                    current_particle_count += self.nparts_initial * self.weight
                else:
                    # Exact match to the analytical discrete math
                    pts = int(np.floor(10.0 * np.sin(np.pi * current_time / self.tf)))
                    current_particle_count += pts * self.weight

            cumulative_particles[t_idx] = current_particle_count

        total_initial_erg_array = cumulative_particles * self.E0
        e_dep_per_step_erg = np.sum(self.E_dep, axis=(0, 1, 2))
        e_dep_cumulative_erg = np.insert(np.cumsum(e_dep_per_step_erg), 0, 0)[:-1]

        cumulative_dead = np.cumsum(self.deaths_per_step)
        dead_energy_bank = cumulative_dead * self.E_cutoff

        # Active Energy
        e_kin_active_erg = total_initial_erg_array - e_dep_cumulative_erg - dead_energy_bank

        # Calculate Total Active KE normalized by E0 (matches U(t)/E0)
        U_t_over_E0 = e_kin_active_erg / self.E0
        np.savetxt("Ut_case4.txt",U_t_over_E0) # save data

        # True tau in seconds
        tau = 8.68177578551766e-10
        t_over_tau = self.timesteps / tau

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Normalized Total KE
        ax.plot(t_over_tau, U_t_over_E0, label=r'$U(t)/E_0$', color='blue', linewidth=2)

        ax.axvline(1.0, label=r'$\tau$', linestyle='-.', color='black')
        ax.set_title(f'Normalized Total Kinetic Energy of Active Population\n(Sinusoidal Source)', fontsize=14)
        ax.set_xlabel(r'$t / \tau$', fontsize=12)
        ax.set_ylabel(r'$U(t) / E_0$', fontsize=12)
        ax.set_xlim(left=0, right=2.0)
        ax.set_ylim(bottom=0)

        ax.grid(True, which='major', linestyle='-', color='lightgray', linewidth=1)
        ax.legend(loc='upper right', fontsize=11, frameon=True, edgecolor='lightgray')

        os.makedirs('fluxes/npp', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'fluxes/npp/ke_sinusoidal_dx_{self.dx}_pulses_{self.npulses}.png', dpi=300)

        print(f"Energy curve saved in {np.round(time.time() - stt, 3)}s.")

# simulation parameters
nparts_initial = 2000
nparts_per_pulse = 10
track_particles = False
verbose = True
ww = 1

length = 1
width  = 1
height = 1
dx = .25
eps = 1e-8

pulses = 2000
#tf = 2.75173497e-03
#dt = 2.75448946e-06
dt = 1.73809325e-12
tf = 1.73635516e-09

E0 = 3.54 # MeV
Eb = 0.001  # Background energy set to 0.0
Tbeta = 1e7 

lp3d = LPSTP_3D(nparts_initial, nparts_per_pulse, ww, pulses, 
                    E0, Eb, Tbeta, length, width, height, dx, eps, 
                    tf, dt, track_particles, verbose)
lp3d.run()
