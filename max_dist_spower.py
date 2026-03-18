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
    # Pre-allocate the arrays (much faster than stacking later)
    p0 = np.empty((n_particles, 3), dtype=np.float64)
    dirs = np.empty((n_particles, 3), dtype=np.float64)

    for p in range(n_particles):
        # 3D Spatial coordinates
        p0[p, 0] = np.random.uniform(i * dx, (i + 1) * dx)
        p0[p, 1] = np.random.uniform(j * dx, (j + 1) * dx)
        p0[p, 2] = np.random.uniform(k * dx, (k + 1) * dx)

        # 3D Isotropic Angular coordinates
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

@njit(cache=True)
def jit_calc_stopping_power(E, Z, e, m_alpha, m_beta, kb, T_beta, hbar, n_e, n_a, w2):
    """JIT-compiled, flattened stopping power calculation."""

    # get_T_alpha
    v2_alpha = (2.0 * E) / m_alpha
    TA = 2.0 * E / (3.0 * kb)
    
    # lambda ab
    inv_lambda_d2 = (4.0 * np.pi / kb) * ((n_e * e**2 / T_beta) + (n_a * (Z * e)**2 / TA))
    lambda_d = np.sqrt(1.0 / inv_lambda_d2)
    m_ab = m_alpha * m_beta / (m_alpha + m_beta)
    u_bar = np.sqrt(2.0 * E / m_alpha)
    r1 = (Z * e * e) / (m_ab * (u_bar * u_bar))
    r2 = hbar / (2.0 * m_ab * u_bar)
    r = max(r1, r2)
    l_ab = np.log(lambda_d / r)
    
    # psi and psi_prime
    x = m_beta * v2_alpha / (2.0 * kb * T_beta)
    psi = math.erf(np.sqrt(x)) - (2.0 / np.sqrt(np.pi)) * np.sqrt(x) * np.exp(-x)
    psi_prime = (2.0 / np.sqrt(np.pi)) * (np.sqrt(x) / np.exp(x))

#    if x <= 0:
#        psi = 0.0
#        psi_prime = 0.0
#    else:
#        psi = math.erf(np.sqrt(x)) - (2.0 / np.sqrt(np.pi)) * np.sqrt(x) * np.exp(-x)
#        psi_prime = (2.0 / np.sqrt(np.pi)) * (np.sqrt(x) / np.exp(x))
        
    # G parameters
    m_ratio = m_beta / m_alpha
    G = psi - m_ratio * (psi_prime - (1.0 / l_ab) * (psi + psi_prime))
    
    # Final calculation
    Theta = 1.0 if x > 1.0 else 0.0
    coef = (Z * Z) * (e * e) * w2 / v2_alpha
    S = coef * (G * (l_ab) + Theta * np.log(1.123 * np.sqrt(x)))
    return S

class LPSTP_3D():
    def __init__(self,nparts, weight, pulses, # particles
                    E0, Eb, Tbeta, # energy
                    length, width, height, dx, eps, # geometry
                    tf, dt, # timing
                    track_particles,verbose # debug
                    ):
        # USE CGS FOR EVERYTHING
        # constants
        self.hbar = 1.0546e-27
        self.kb = 1.380649e-16
        self.MeVtoERG = 1.60218e-6
        self.ERGtoMeV = 1 / self.MeVtoERG

        # setup
        self.nparts = nparts
        self.weight = weight # n super alphas
        self.track_particles = track_particles
        self.verbose = verbose

        # geometry
        self.length = length
        self.width = width
        self.height = height
        self.dx = dx
        self.Volume = self.length * self.width * self.height
        self.dV = dx * dx * dx # volume of the cell
        self.eps = eps
        self.cmap = 'plasma'

        # mass
        self.Z = 2
        self.m_alpha = 6.644657230e-24 # kg
        self.m_beta = 9.109e-28 #kg
        self.rho = 1 # g/cc
        self.n_a  = self.rho / self.m_alpha
        self.n_e = self.Z * self.n_a

        # charge
        self.e = 4.803204e-10
        self.erg = 1e7

        # Energy
        self.E0 = 3.54 * self.MeVtoERG  # ~5.67e-6 ergs
        self.Eb = 0.20 * self.MeVtoERG  # ~3.20e-7 ergs
        self.T_beta_0 = Tbeta
        self.E_beta = 1.5 * self.kb * self.T_beta_0 # ergs
        self.w2 = self.OmegaBeta2()
        self.cv = 1e8 # erg/K/g
        self.sie = self.cv * (self.rho * self.dV)

        # timing
        self.tf = tf
        self.dt = dt
        self.timesteps = np.linspace(0,self.tf,int(self.tf / self.dt)+1)

        # build stopping power mesh
        self.S = np.zeros((int(self.length/self.dx), int(self.width/self.dx), int(self.height/self.dx)),dtype=float)
        self.phi = np.zeros((int(self.length/self.dx), int(self.width/self.dx), int(self.height/self.dx), 
                        self.timesteps.size),dtype=float)
        self.E_dep = np.zeros_like(self.phi)
        self.S_const = 1e-5

        # Sourcing and Weight Logic
        self.npulses = pulses
        self.emis_time = self.tf / self.npulses
        n_cells_1d = int(self.length / self.dx)
        self.n_cells_total = n_cells_1d ** 3
        self.particles_per_cell = max(1, self.nparts // self.n_cells_total)
        self.simulated_particles_per_pulse = self.particles_per_cell * self.n_cells_total
        self.physical_particles_per_pulse = self.simulated_particles_per_pulse * self.weight
        self.deaths_per_step = np.zeros(len(self.timesteps))

    def source_alphas(self, isotropic=True):
        """Yields batches of particles per cell using pre-calculated weights."""
        if self.verbose:
            print(f"Sourcing {self.particles_per_cell} particle(s) per cell ({self.simulated_particles_per_pulse} simulated histories per pulse).")

        n = int(self.length // self.dx)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # _jit_generate_batch handles the 3D isotropic generation instantly
                    yield _jit_generate_batch(i, j, k, self.dx, self.particles_per_cell)

    def OmegaBeta2(self):
        return (4 * np.pi * self.n_e * self.e * self.e) / self.m_beta

    def tally_E_dep(self, I, J, K, t_idx, actual_E_loss):
        self.E_dep[I, J, K, t_idx] += actual_E_loss * self.weight

    def tally_phi(self, I, J, K, t_idx, ds):
        # ds is the distance traveled in this cell during this timestep
        self.phi[I, J, K, t_idx] += ds * self.weight / self.dV

    def TransportAlphas(self, p0_batch, dir_batch, base_history_id, birth_time,lp=True):
        for p_idx in range(len(p0_batch)):
            # [x, y, z, E, t]
            particle = np.array([
                p0_batch[p_idx, 0],
                p0_batch[p_idx, 1],
                p0_batch[p_idx, 2],
                self.E0,
                birth_time,
            ])
            u, v, w = dir_batch[p_idx]

            # Global history ID (useful for plotting later)
            history_id = base_history_id + p_idx

            if self.track_particles:
                self.particle_path.append([history_id, particle[0], particle[1], particle[2], particle[3], particle[4]])

            Transport = True
            while Transport:
                # periodic boundary conditions
                particle[0] %= self.length
                particle[1] %= self.width
                particle[2] %= self.height

                # find cell
                I = int(particle[0] / self.dx)
                J = int(particle[1] / self.dx)
                K = int(particle[2] / self.dx)
                I = min(I, int(self.length / self.dx) - 1)
                J = min(J, int(self.width / self.dx) - 1)
                K = min(K, int(self.height / self.dx) - 1)

                # stopping power, call jit compiled code
                dmin = jit_min_d_boundary(particle[0], particle[1], particle[2],
                                          u, v, w, I, J, K, self.dx)

                t_idx = int(particle[4] / self.dt + self.eps)
                t_idx = min(t_idx, len(self.timesteps) - 1)

                cumulative_heat = np.sum(self.E_dep[I, J, K, :t_idx + 1])
                local_T_beta = self.T_beta_0 + (cumulative_heat / self.sie)

                # Li-Petrasso stopping power
                if lp:
                    S_val = jit_calc_stopping_power(particle[3], self.Z, self.e,
                                                self.m_alpha, self.m_beta, self.kb,
                                                local_T_beta, self.hbar, self.n_e,
                                                self.n_a, self.w2)
                # constant stopping power value
                else: S_val = self.S_const

                self.S[I, J, K] = S_val

                # Calculate velocity first to find out how far it goes in one time bin
                v_cell = np.sqrt(2 * particle[3] / self.m_alpha)
                current_cycle = int(particle[4] / self.dt + self.eps)
                end_of_step_time = (current_cycle + 1) * self.dt
                time_remaining_in_step = end_of_step_time - particle[4]

                # Limit distance strictly to the remaining time in this bin
                max_time_dist = v_cell * time_remaining_in_step
                max_dist_before_death = (particle[3] - self.Eb) / S_val

                # particle constrained by the grid, death, or the time bin, allows for coarse mesh
                actual_dist = min(dmin, max_dist_before_death, max_time_dist)
                actual_loss = S_val * actual_dist

                self.tally_E_dep(I, J, K, t_idx, actual_loss)
                self.tally_phi(I,J,K,t_idx,dmin)

                particle[3] -= actual_loss

                # update Time
                t_cell = actual_dist / v_cell
                particle[4] += t_cell

                # thermalization
                if particle[3] <= self.Eb * 1.0001:
                    particle[3] = self.Eb
                    death_idx = min(int(particle[4] / self.dt), len(self.timesteps) - 1)
                    self.deaths_per_step[death_idx] += self.weight
                    Transport = False

                if particle[4] > self.timesteps[-1]:
                    Transport = False

                # push
                if Transport:
                    particle[0] += actual_dist * u
                    particle[1] += actual_dist * v
                    particle[2] += actual_dist * w

                    # periodic bc's
                    particle[0] %= self.length
                    particle[1] %= self.width
                    particle[2] %= self.height

    
    def run(self):
        if self.track_particles: self.particle_path = []
        stt = time.time()

        global_history_id = 0

        # Pre-calculate the exact snapped timesteps for all pulses
        pulse_target_times = [i * self.emis_time for i in range(self.npulses)]
        snapped_indices = [int(np.round(t / self.dt)) for t in pulse_target_times]

        # driver
        for t_idx, current_time in enumerate(self.timesteps):
            if self.verbose:
                print(f"CYCLE {t_idx}, t = {current_time:5g}")
            else: 
                if t_idx % 10 == 0:
                    print(f"CYCLE {t_idx}, t = {current_time:5g}")

            # count pulses for this step
            pulses_this_step = snapped_indices.count(t_idx)

            for _ in range(pulses_this_step):
                if self.verbose:
                    print(f"--> Sourcing Pulse at snapped t = {current_time:.4e} s")

                ptcl_batches = self.source_alphas()
                for cell_idx, (p0_batch, dir_batch) in enumerate(ptcl_batches):
                    self.TransportAlphas(p0_batch, dir_batch, global_history_id, current_time)
                    global_history_id += len(p0_batch)

        print(f"Total Transport Time: {np.round(time.time() - stt, 5)}s")
        print(f"Total Simulated Histories: {global_history_id}")

        # plotting
        #self.plot_paths()
        #self.animate()
        self.plot_energy_relaxation()

    # plotting functions
    def plot_paths(self):
        # Set up a 3D figure
        stt = time.time()
        print("Plotting 3D paths...")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        path_arr = np.array(self.particle_path)
        if len(path_arr) == 0:
            print("No particle paths to plot.")
            return

        # 1. VECTORIZED SPLIT: Find where the ID changes and split the array instantly
        change_idx = np.where(path_arr[:-1, 0] != path_arr[1:, 0])[0] + 1
        particle_blocks = np.split(path_arr, change_idx)

        all_segments = []
        all_energies = []
        start_x, start_y, start_z = [], [], []
        end_x, end_y, end_z = [], [], []

        # 2. COLLECT DATA: Loop through the pre-split blocks (very fast)
        for arr in particle_blocks:
            x, y, z = arr[:, 1], arr[:, 2], arr[:, 3]
            E = arr[:, 4] * self.erg / 16.02

            # Store start and end points for batch plotting
            start_x.append(x[0]); start_y.append(y[0]); start_z.append(z[0])
            end_x.append(x[-1]); end_y.append(y[-1]); end_z.append(z[-1])

            # Build segments for this particle
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            all_segments.extend(segments)
            all_energies.extend(E[:-1])

        # 3. BATCH PLOT: Pass everything to Matplotlib exactly ONCE
        lc = Line3DCollection(
            all_segments,
            cmap=self.cmap,
            array=np.array(all_energies),
            linewidth=1.5,
            alpha=0.1, # Adds transparency so dense tracks don't block each other
            zorder=.1
        )
        ax.add_collection3d(lc)

        # Draw all start markers at once, then all end markers at once
        ax.scatter(start_x, start_y, start_z, color='green', s=15, marker='o', zorder=.1)
        ax.scatter(end_x, end_y, end_z, color='red',  s=15, marker='X', zorder=.1)

        # Formatting
        cbar = plt.colorbar(lc, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label("Energy (MeV)")

        legend_elements = [
            Line2D([0], [0], marker='o', color='green', linestyle='None', label='Start'),
            Line2D([0], [0], marker='X', color='red',  linestyle='None', label='End'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(f'3D Stopping Power Paths, dx = {self.dx} cm')
        ax.set_xlabel('X / Length (cm)')
        ax.set_ylabel('Y / Width (cm)')
        ax.set_zlabel('Z / Height (cm)')

        ax.set_xlim([0, self.length])
        ax.set_ylim([0, self.width])
        ax.set_zlim([0, self.height])

        os.makedirs('fluxes/max_dist', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'fluxes/max_dist/paths_3D_{self.nparts}_dx_{self.dx}.png', dpi=300)
        plt.close(fig)

        print(f"3D Path plot saved in {np.round(time.time() - stt, 3)}s.")

    def plot_energy_relaxation(self):
        stt = time.time()
        ns = 1e9
        print(f"Generating snapped multi-pulse energy relaxation curve ({self.npulses} pulses)...")

        # 1. Match the exact snapping logic from the run loop
        pulse_target_times = [i * self.emis_time for i in range(self.npulses)]
        snapped_indices = [int(np.round(t / self.dt)) for t in pulse_target_times]

        physical_particles_per_pulse = self.physical_particles_per_pulse
        cumulative_particles = np.zeros(len(self.timesteps))
        current_particle_count = 0

        # Build the exact same staircase we used in the run loop
        for t_idx in range(len(self.timesteps)):
            pulses_here = snapped_indices.count(t_idx)
            current_particle_count += pulses_here * physical_particles_per_pulse
            cumulative_particles[t_idx] = current_particle_count

        # 2. Create time-dependent analytical target arrays (Staircases)
        total_initial_erg_array = cumulative_particles * self.E0
        target_kin_erg_array = cumulative_particles * self.Eb
        target_dep_erg_array = (total_initial_erg_array - target_kin_erg_array)
        #target_dep_erg_array = self.weight * (total_initial_erg_array - target_kin_erg_array)

        # 3. Calculate simulation deposition natively
        e_dep_per_step_erg = np.sum(self.E_dep, axis=(0, 1, 2))

        raw_cumsum = np.cumsum(e_dep_per_step_erg)
        e_dep_cumulative_erg = np.insert(raw_cumsum, 0, 0)[:-1]

        cumulative_dead = np.cumsum(self.deaths_per_step)
        dead_energy_bank = cumulative_dead * self.Eb

        # 4. Calculate ACTIVE kinetic energy in the system
        # (Total born) - (Total deposited) - (Energy trapped in dead particles)
        e_kin_active_erg = total_initial_erg_array - e_dep_cumulative_erg - dead_energy_bank

        # --- PLOTTING AESTHETICS ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # save results to an array
        arr = np.array((e_dep_cumulative_erg,e_kin_active_erg))
        np.savetxt(f"ke_and_edep_{self.npulses}_pulses.txt",arr)

        # Plot the simulation data
        ax.plot(self.timesteps * ns, e_kin_active_erg, label=r'$E_{kin}$ (Alphas)', color='tab:blue', linewidth=2)
        ax.plot(self.timesteps * ns, e_dep_cumulative_erg, label=r'$E_{dep}$ (Plasma)', color='tab:orange', linewidth=2)

        # Plot Theoretical Asymptotes (Staircases)
        ax.plot(self.timesteps * ns, target_dep_erg_array, color='black', linestyle='--', linewidth=1.2,
                    label=r'Analytic $E_{dep}$ Max')
        ax.axhline(self.Eb * self.nparts * self.weight, color='black', linestyle='--',linewidth=1.2,
                    label='Background Energy')

        # Formatting
        ax.set_title(f'Multi-Pulse System Energy Relaxation\n(Total Injected Particles: {int(cumulative_particles[-1] / self.weight)})', fontsize=14)
        ax.set_xlabel('t (ns)', fontsize=12)
        ax.set_ylabel('Energy (erg)', fontsize=12)

        ax.grid(True, which='major', linestyle='-', color='lightgray', linewidth=1)
        ax.legend(loc='center right', fontsize=11, frameon=True, edgecolor='lightgray')

        #os.makedirs('fluxes/max_dist/constant', exist_ok=True)
        os.makedirs('fluxes/max_dist', exist_ok=True)
        plt.tight_layout()
        #plt.savefig(f'fluxes/max_dist/constant/energy_relaxation_multipulse_dx_{self.dx}_S{self.S_const}.png', dpi=300)
        plt.savefig(f'fluxes/max_dist/energy_relaxation_multipulse_dx_{self.dx}_pulses_{self.npulses}.png', dpi=300)

        print(f"Energy relaxation curve saved in {np.round(time.time() - stt, 3)}s.")

    def animate(self):
        stt = time.time()
        # Compress the Z-axis (index 2) by summing the flux along the height
        # self.phi is shape (nx, ny, nz, nt) -> phi_2d becomes (nx, ny, nt)
        phi_2d = np.sum(self.phi, axis=2) / np.sum(self.phi)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Prevent division by zero errors in color normalization if the array is empty
        vmax = np.max(phi_2d)
        if vmax == 0: vmax = 1.0

        # Initialize the plot with the first time step.
        # .T transposes the array so X maps to the horizontal axis and Y to the vertical.
        im = ax.imshow(phi_2d[:, :, 0].T,
                       origin='lower',
                       extent=[0, self.length, 0, self.width],
                       vmin=0,
                       vmax=vmax,
                       cmap=self.cmap)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$\phi(x,y,t)$ (Z-Integrated)')

        ax.set_title(f'Time Evolution of Scalar Flux -- t = 0.0 s')
        ax.set_xlabel('Length / X (cm)')
        ax.set_ylabel('Width / Y (cm)')

        # The function that updates the frame for each timestep
        def update(t_idx):
            im.set_array(phi_2d[:, :, t_idx].T)
            current_t = self.timesteps[t_idx]
            ax.set_title(f'Scalar Flux -- t = {current_t:.3e} s')
            return [im]

        # Create the animation
        ani = FuncAnimation(fig, update, frames=self.timesteps.size, blit=True)

        # Ensure the output directory exists
        os.makedirs('fluxes/max_dist', exist_ok=True)
        filename = f"fluxes/max_dist/phi_2D_evol_{self.nparts}_dx_{self.dx}.mp4"

        # Save the file
        writer = FFMpegWriter(fps=12, bitrate=2000,
                              extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-loop', '0'])
        ani.save(filename, writer=writer)
        plt.close(fig)

        print(f"Animation Generation Time: {np.round(time.time() - stt, 5)}s")

    # helper functions
    @staticmethod
    def source_this_timestep(current_t, emis_time, tf):
        """Returns True if particles should be sourced at current_t (floating-point safe)."""
        if current_t == 0.0:
            return True # Always source at t=0

        #if current_t > tf/2:
        #    return False # don't source after halfway mark

        # Check if current_t is a multiple of emis_time
        ratio = current_t / emis_time

        # If the ratio is within a tiny epsilon of a perfect integer, it's a match!
        if abs(ratio - round(ratio)) < 1e-8:
            return True

        return False

    @staticmethod
    def get_psi(E_alpha,x):
        # returns one part of the psi array
        # Integrate from 0 to xi
        if x <= 0: return 0;
        return math.erf(np.sqrt(x)) - (2 / np.sqrt(np.pi)) * np.sqrt(x)* np.exp(-x);

    @staticmethod
    def get_psi_prime(x):
        # the derivative of an integral is the function evaluated at the bounds
        return 2 / np.sqrt(np.pi) * (np.sqrt(x) / np.exp(x))

# simulation
nparts = 5000
track_particles = True
verbose = True
ww = 1

# geometry in cm
length = 1
width  = 1
height = 1
dx = .25
eps = 1e-8

# timing
tf = 1e-10 #s
dt = 1e-12 #s
pulses = 100

# energy
E0 = 3.54 #MeV
Eb = .2 #MeV
Tbeta = 1e7 #Kelvin

lp3d = LPSTP_3D(nparts, ww, pulses, # particles
                    E0, Eb, Tbeta, # energy, MeV
                    length, width, height, dx, eps, # geometry
                    tf, dt, # timing
                    track_particles,verbose # debug
                    )
lp3d.run()
