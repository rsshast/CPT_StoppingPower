from input import *

# memory savings
if nparts > 100:
    track_particles = False
    verbose = False

class FourierStoppingPower():
    def __init__(self,nparts, swarm, # particles
                    num_ords,mu, theta_bins, d_theta, # direction
                    groups, E_grid, E0, Eb, # energy
                    length, width, dx, Area, eps, # geometry
                    K, # source order
                    track_particles,verbose # debug
                    ): 
        self.nparts = nparts
        self.swarm = swarm
        self.track_particles = track_particles
        self.verbose = verbose

        self.num_ords = num_ords
        self.mu = mu
        self.theta_bins = theta_bins
        self.d_theta = d_theta

        self.groups = groups
        self.E_grid = E_grid
        self.E0 = E0
        self.Eb = Eb

        self.length = length
        self.width = width
        self.dx = dx
        self.Area = Area
        self.eps = eps

        # build mesh
        self.K = K
        self.S = np.zeros((int(self.length/self.dx), int(self.width/self.dx)),dtype=float)
        # fluxes
        self.psi = np.zeros((self.S.shape[0], self.S.shape[1], self.num_ords, self.groups))

        # plotting
        self.cmap = 'plasma'

    def starting_points(self):
        # set same starting point for npoints
        self.p0 = np.zeros((self.nparts,2))
        self.ordinate = np.zeros((self.nparts))
        for n in range(nparts):
            self.p0[n,0] = np.random.uniform(.01,.99) * length
            self.p0[n,1] = np.random.uniform(.01,.99) * width
            self.ordinate[n] = np.random.choice(self.mu)
    
    def get_S(self):
        @njit(parallel=True,fastmath=True)
        def get_S_parallel(S,dx,eps,K,max_iters = 100):
            for i in prange(S.shape[0]):
                for j in range(S.shape[1]):
                    pos_source = False
                    iters = 0
                    while pos_source == False:
                        # build fourier expansion of order K. B0 = 0
                        coefs = np.random.uniform(-1, 1, size=(K+1, 2))
                        coefs[0,1] = 0
                        S_foo = 0
                        for k in range(K):
                            S_foo += (coefs[k,0] * np.cos(i*k*np.pi*dx / length)
                                      + coefs[k,1] * np.sin(j*k*np.pi*dx / width))

                        if S_foo > 0:
                            S[i,j] = S_foo
                            pos_source = True
        
                        iters += 1
                        if iters > max_iters:
                            S[i,j] = eps
                            break
        
            return S

        self.S = get_S_parallel(self.S,self.dx,self.eps,self.K)

    @staticmethod
    def _find_group(E_keV, edges_keV):
        """Return group index g such that edges[g] >= E > edges[g+1]."""
    
        if E_keV > edges_keV[0] or E_keV <= edges_keV[-1]:
            raise ValueError("Energy Out of Range")
    
        # edges are descending; search on reversed bins
        g = np.searchsorted(edges_keV[::-1], E_keV, side="left")
        # g counts from bottom; convert to top-index
        g = (len(edges_keV) - 1) - g
        return int(g)
    
    @staticmethod
    def _find_theta_bin(mu_x, mu_y, theta_edges):
        """Direction bin in [0, 2pi)"""
        theta = np.arctan2(mu_y, mu_x)
        if theta < 0.0:
            theta += 2.0*np.pi
        m = int(np.searchsorted(theta_edges, theta, side="right") - 1)
        m = max(0, min(m, len(theta_edges) - 2))
        return m
    
    def get_psi(self,
        psi, I, J, mu_x, mu_y, E_old_keV, ds, E_edges_keV, cell_area,
        theta_edges, dtheta, S, swarm, weight=1.0):
    
        # angle bin
        theta = np.arctan2(mu_y, mu_x)
        if theta < 0.0: theta += 2.0*np.pi
        m = self._find_theta_bin(mu_x, mu_y, theta_edges)
    
        # energy group at segment start
        g = self._find_group(E_old_keV, E_edges_keV)
        dE_g = E_edges_keV[g] - E_edges_keV[g+1]  # group width (keV)
        psi[I, J, m, g] += weight * ds / (cell_area * dtheta * dE_g)
    
        # energy update in this cell
        dE = swarm * S[I, J] * ds
        E_new = E_old_keV - dE
    
        return psi, E_new

    def get_phi(self,psi):
        self.phi = np.sum(psi,axis=2) * self.d_theta / self.nparts

    @staticmethod
    def min_d_boundary(x, y, mu_x, mu_y, dx):
        assert mu_x != 0
        assert mu_y != 0
        if mu_x > 0: dist_x = (np.ceil(x / dx) * dx - x) / mu_x
        else: dist_x = (x - np.floor(x / dx) * dx) / -mu_x
    
        if mu_y > 0: dist_y = (np.ceil(y / dx) * dx - y) / mu_y
        else: dist_y = (y - np.floor(y / dx) * dx) / -mu_y
    
        return min(dist_x, dist_y)

    def TransportAlphas(self):
        if self.track_particles: self.particle_path = []

        stt = time.time()
        for history in range(self.nparts):
            if history % 100 == 0: print(f"History Number: {history + 1}")
            Transport = True
            particle = np.array([self.p0[history,0],self.p0[history,1],self.E0])
            theta = np.arccos(self.ordinate[history])
            mu_x = np.cos(theta)
            mu_y = np.sin(theta)
            if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2]])
            if verbose: print('\n')
        
            # transport alphas
            while Transport:
                # find mesh element of particle and dist to nearest boundary along ordinate
                if self.verbose: print(particle)
                # particle outside x domain
                if particle[0] < 0 or particle[0] > self.length: break
                # particle outside y domain
                elif particle[1] < 0 or particle[1] > self.width: break
        
                I = int(particle[0] / self.dx)
                J = int(particle[1] / self.dx)
                dmin = self.min_d_boundary(particle[0], particle[1], mu_x, mu_y, self.dx)
                assert dmin > 0
                E_old = particle[2]
        
                psi, E_new = self.get_psi(
                    self.psi, I,J,mu_x,mu_y, E_old, dmin, self.E_grid,
                    self.Area, self.theta_bins, self.d_theta, self.S, self.swarm
                )

                # calculate energy loss and E deposition
                f =  (E_old - E_new) / E_old
                self.S[I,J] *= (1+f)
                particle[2] = E_new
        
                # kill the particle if it joins background
                if particle[2] <= self.Eb:
                    particle[2] = self.Eb
                    if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2]])
                    Transport = False
        
                # push particle
                else:
                    particle[0] -= dmin * mu_x
                    particle[1] -= dmin * mu_y

                    # particle outside x domain
                    if particle[0] < 0 or particle[0] > self.length:
                        particle[0] = np.clip(particle[0], 0, self.length)
                        if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2]])
                        break

                    # particle outside y domain
                    elif particle[1] < 0 or particle[1] > self.width:
                        particle[1] = np.clip(particle[1], 0, self.width)
                        if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2]])
                        break
                    if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2]])
            if self.verbose: print(particle)

        print(f"Transport Time for {nparts} Particles: {np.round(time.time() - stt,5)}s")

    def run(self):
        # get stopping power on mesh
        print("Build Stopping Power Mesh from Fourier Series")
        self.starting_points()
        self.get_S()

        # perform transport
        print("Begin Transport")
        self.TransportAlphas()
        # get scalar flux
        self.get_phi(self.psi)

        # plotting
        if self.track_particles:
            self.plot_paths()
        self.plot_group_phi()

    def plot_paths(self):
        print("Plotting Particle Paths")
        norm = LogNorm(vmin=self.Eb, vmax=self.E0)
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        color_mappable = None
        path_arr = np.array(self.particle_path)
        final_vals = np.zeros((self.nparts,3))
    
        for i in range(self.nparts):
            mask = path_arr[:,0] == float(i)
            arr = path_arr[mask]
            x = arr[:,1]
            y = arr[:,2]
            E = arr[:,3]
            final_vals[i] = arr[-1,1:]


            # Build line segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                cmap=self.cmap,
                norm=norm,
                array=E[:-1],
                linewidth=2,
                zorder=2
            )
            ax.add_collection(lc)
            if color_mappable is None: color_mappable = lc
    
            # markers
            ax.scatter(x[0], y[0], color='green', s=10, marker='o', zorder=5)
            ax.scatter(x[-1], y[-1], color='red',  s=10, marker='X', zorder=5)
    
    
        cbar = plt.colorbar(color_mappable, ax=ax)
        cbar.set_label("Energy (keV)")
        legend_elements = [
            Line2D([], [], marker='o', color='green', linestyle='None', label='Start'),
            Line2D([], [], marker='X', color='red',  linestyle='None', label='End'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
        ax.set_title(f'Stopping Power Paths, dx = {dx}')
        ax.set_xlabel('Length')
        ax.set_ylabel('Width')
        ax.set_xlim([0, self.length])
        ax.set_ylim([0, self.width])
        ax.grid(which='both')
    
        plt.savefig(f'fluxes/fourier/paths_{self.nparts}_dx_{self.dx}.png')

    def plot_group_phi(self):
        print("Plotting Group Fluxes")
        # plot flux
        for g in range(groups):
            plt.figure(figsize=(8,6))
            plt.imshow(self.phi[:,:,g].T, cmap=self.cmap,origin="lower", extent=[0,self.length,0,self.width])
            plt.colorbar()
            plt.title(fr"Stopping Power $\phi(x,y,g={g+1})$")
            plt.savefig(f"fluxes/fourier/phi_g{g+1}.png")

        

class BcStoppingPower():
    def __init__(self,nparts, swarm, # particles
                    num_ords,mu, theta_bins, d_theta, # direction
                    groups, E_grid, E0, Eb, # energy
                    length, width, dx, Area, eps, # geometry
                    track_particles,verbose # debug
                    ): 
        self.nparts = nparts
        self.swarm = swarm
        self.track_particles = track_particles
        self.verbose = verbose

        self.num_ords = num_ords
        self.mu = mu
        self.theta_bins = theta_bins
        self.d_theta = d_theta

        self.groups = groups
        self.E_grid = E_grid
        self.E0 = E0
        self.Eb = Eb

        self.length = length
        self.width = width
        self.dx = dx
        self.Area = Area
        self.eps = eps

        # build mesh
        self.S = self.swarm * np.ones((int(length/dx), int(width/dx)),dtype=float)
        # fluxes
        self.psi = np.zeros((self.S.shape[0], self.S.shape[1], self.num_ords, self.groups))

        # plotting
        self.cmap = 'plasma'

    def starting_points(self):
        # set same starting point for npoints
        self.p0 = np.zeros((self.nparts,2))
        self.ordinate = np.zeros((self.nparts))
        for n in range(nparts):
            self.p0[n,0]+= self.dx*self.dx
            self.p0[n,1] = np.random.uniform(.01,.99) * width
            self.ordinate[n] = np.random.choice(self.mu)

    @staticmethod
    def min_d_boundary(x, y, mu_x, mu_y, dx, tol):
        if mu_x > 0:
            i = int(np.floor(x / dx))
            xb = (i + 1) * dx
            if xb - x <= tol: xb += dx
            dist_x = (xb - x) / mu_x
        else:
            i = int(np.floor(x / dx))
            xb = i * dx
            if x - xb <= tol: xb -= dx
            dist_x = (x - xb) / (-mu_x)
    
        if mu_y > 0:
            j = int(np.floor(y / dx))
            yb = (j + 1) * dx
            if yb - y <= tol: yb += dx
            dist_y = (yb - y) / mu_y
        else:
            j = int(np.floor(y / dx))
            yb = j * dx
            if y - yb <= tol: yb -= dx
            dist_y = (y - yb) / (-mu_y)
    
        return min(dist_x, dist_y)

    @staticmethod
    def _find_group(E_keV, edges_keV):
        """Return group index g such that edges[g] >= E > edges[g+1]."""
    
        if E_keV > edges_keV[0] or E_keV <= edges_keV[-1]:
            raise ValueError("Energy Out of Range")
    
        # edges are descending; search on reversed bins
        g = np.searchsorted(edges_keV[::-1], E_keV, side="left")
        # g counts from bottom; convert to top-index
        g = (len(edges_keV) - 1) - g
        return int(g)

    @staticmethod
    def _find_theta_bin(mu_x, mu_y, theta_edges):
        """Direction bin in [0, 2pi)"""
        theta = np.arctan2(mu_y, mu_x)
        if theta < 0.0:
            theta += 2.0*np.pi
        m = int(np.searchsorted(theta_edges, theta, side="right") - 1)
        m = max(0, min(m, len(theta_edges) - 2))
        return m

    def get_psi(self,
        psi, I, J, mu_x, mu_y, E_old_keV, ds, E_edges_keV, cell_area,
        theta_edges, dtheta, S, weight=1.0):
    
        # angle bin
        theta = np.arctan2(mu_y, mu_x)
        if theta < 0.0: theta += 2.0*np.pi
        m = self._find_theta_bin(mu_x, mu_y, theta_edges)
    
        # energy group at segment start
        g = self._find_group(E_old_keV, E_edges_keV)
        dE_g = E_edges_keV[g] - E_edges_keV[g+1]  # group width (keV)
        psi[I, J, m, g] += weight * ds / (cell_area * dtheta * dE_g)
    
        # energy update in this cell
        dE = S[I, J] * ds
        E_new = E_old_keV - dE
    
        return psi, E_new
    
    def get_phi(self,psi):
        self.phi = np.sum(psi,axis=2) * self.d_theta / self.nparts

    def TransportAlphas(self):
        if self.track_particles: self.particle_path = []

        stt = time.time()
        for history in range(self.nparts):
            if history % 100 == 0: print(f"History Number: {history+1}")
            Transport = True
            # pick a starting point and direction from gauss-legendre quadrature
            particle = np.array([self.p0[history,0],self.p0[history,1],self.E0])
            theta = np.arccos(self.ordinate[history])
            mu_x = np.cos(theta)
            mu_y = np.sin(theta) * ((-1) ** history)
            if self.track_particles == True:
                self.particle_path.append([history,particle[0],particle[1], particle[2]])
            if self.verbose: print('\n')
        
            # transport alphas
            while Transport:
                # find mesh element of particle and dist to nearest boundary along ordinate
                if self.verbose: print(particle)
                # particle outside x domain
                if particle[0] <= 0 or particle[0] >= self.length: break
                # particle outside y domain
                elif particle[1] <= 0 or particle[1] >= self.width: break
        
                I = int(particle[0] / self.dx)
                J = int(particle[1] / self.dx)
                dmin = self.min_d_boundary(particle[0], particle[1], mu_x, mu_y, self.dx, self.eps)
                assert dmin > 0
        
                E_old = particle[2]
        
                psi, E_new = self.get_psi(
                    self.psi, I,J,mu_x,mu_y, E_old, dmin, self.E_grid,
                    self.Area, self.theta_bins, self.d_theta, self.S
                )
        
                # calculate energy loss and E deposition
                f =  (E_old - E_new) / E_old
                self.S[I,J] *= (1+f)
                particle[2] = E_new
        
                # kill the particle when it enters the background
                if particle[2] < self.Eb:
                    particle[2] = self.Eb
                    if self.track_particles == True:
                        self.particle_path.append([history,particle[0],particle[1], particle[2]])
                    Transport = False
        
                # push particle
                else:
                    particle[0] += dmin * mu_x
                    particle[1] += dmin * mu_y

                    # kill particle if it leaks out of the x domain
                    if particle[0] < 0 or particle[0] > self.length:
                        particle[0] = np.clip(particle[0], 0, self.length)
                        if self.track_particles == True:
                            self.particle_path.append([history,particle[0],particle[1], particle[2]])
                        break

                    # particle outside y domain
                    elif particle[1] < 0 or particle[1] > self.width:
                        particle[1] = np.clip(particle[1], 0, self.width)
                        if self.track_particles == True:
                            self.particle_path.append([history,particle[0],particle[1], particle[2]])
                        break
                    if self.track_particles == True:
                        self.particle_path.append([history,particle[0],particle[1], particle[2]])
            if self.verbose:
                print(particle)
                #d = np.sqrt((p0[0,0] - particle[0]) ** 2 + (p0[0,1] - particle[1]) ** 2)
                #print(d)
        
    def run(self):
        # get stopping power on mesh
        print("Build Stopping Power Mesh from Fourier Series")
        self.starting_points()

        # perform transport
        print("Begin Transport")
        self.TransportAlphas()
        # get scalar flux
        self.get_phi(self.psi)

        # plotting
        if self.track_particles:
            self.plot_paths()
        self.plot_group_phi()

    def plot_paths(self):
        print("Plotting Particle Paths")
        norm = LogNorm(vmin=self.Eb, vmax=self.E0)
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        color_mappable = None
        path_arr = np.array(self.particle_path)
        final_vals = np.zeros((self.nparts,3))
    
        for i in range(self.nparts):
            mask = path_arr[:,0] == float(i)
            arr = path_arr[mask]
            x = arr[:,1]
            y = arr[:,2]
            E = arr[:,3]
            final_vals[i] = arr[-1,1:]


            # Build line segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                cmap=self.cmap,
                norm=norm,
                array=E[:-1],
                linewidth=2,
                zorder=2
            )
            ax.add_collection(lc)
            if color_mappable is None: color_mappable = lc
    
            # markers
            ax.scatter(x[0], y[0], color='green', s=10, marker='o', zorder=5)
            ax.scatter(x[-1], y[-1], color='red',  s=10, marker='X', zorder=5)
    
    
        cbar = plt.colorbar(color_mappable, ax=ax)
        cbar.set_label("Energy (keV)")
        legend_elements = [
            Line2D([], [], marker='o', color='green', linestyle='None', label='Start'),
            Line2D([], [], marker='X', color='red',  linestyle='None', label='End'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
        ax.set_title(f'Stopping Power Paths, dx = {dx}')
        ax.set_xlabel('Length')
        ax.set_ylabel('Width')
        ax.set_xlim([0, self.length])
        ax.set_ylim([0, self.width])
        ax.grid(which='both')
    
        plt.savefig(f'fluxes/bc/paths_{self.nparts}_dx_{self.dx}.png')

    def plot_group_phi(self):
        print("Plotting Group Fluxes")
        # plot flux
        for g in range(groups):
            plt.figure(figsize=(8,6))
            plt.imshow(self.phi[:,:,g].T, cmap=self.cmap,origin="lower", extent=[0,self.length,0,self.width])
            plt.colorbar()
            plt.title(fr"Stopping Power $\phi(x,y,g={g+1})$")
            plt.savefig(f"fluxes/bc/phi_g{g+1}.png")

        
class LiPetrassoStoppingPower():
    def __init__(self,nparts, swarm, # particles
                    num_ords,mu, theta_bins, d_theta, # direction
                    groups, E_grid, E0, Eb, # energy
                    length, width, dx, Area, eps, # geometry
                    t_f, # timing
                    track_particles,verbose # debug
                    ): 
        self.nparts = nparts
        self.swarm = swarm
        self.track_particles = track_particles
        self.verbose = verbose

        self.num_ords = num_ords
        self.mu = mu
        self.theta_bins = theta_bins
        self.d_theta = d_theta
        self.groups = groups

        self.length = length
        self.width = width
        self.dx = dx
        self.Area = Area
        self.eps = eps

        # units and time
        self.t_f = t_f
        self.m_alpha = 6.644657230e-27 # kg
        self.e = 1.602e-19 # C
        self.m_beta = 9.109e-31 #kg
        self.E0 = E0 * self.e * 1000 #joules
        self.Eb = Eb * self.e * 1000 #joules
        self.Z = 2
        self.eps0 = 8.854e-12 # farads
        self.k = 1 / (4 * np.pi * self.eps0)
        self.kb = 1.3806503e-23
        self.E_beta = 5e3 * self.e # joules
        self.T_beta = 2 * self.E_beta / (3 * self.kb)
        self.hbar = 6.626e-34 / (2 * np.pi)
        self.dt = self.get_min_dt()
        self.timesteps = np.linspace(0,self.t_f,int(self.t_f / self.dt))
        self.n_e = self.swarm / self.Area
        self.E_grid = np.exp(np.linspace(np.log(self.E0),np.log(self.Eb),self.groups+1))

        # build stopping power mesh
        self.S = np.zeros((int(self.length/self.dx), int(self.width/self.dx)),dtype=float)
        self.psi = np.zeros((self.S.shape[0], self.S.shape[1], self.num_ords, self.groups, self.timesteps.size))
        self.cmap = 'viridis'

    def get_min_dt(self, dims=2):
        # take the max alpha energy and area of the cell, and say that no particle crosses more than
        # half the mesh element per timestep
        D = self.Area ** (1/dims)
        v = np.sqrt(2 * self.E0 / self.m_alpha)
    #    return np.sqrt(2) * D / v
        return D / (2 * v)

    def starting_points(self):
        # set same starting point for npoints
        self.p0 = np.zeros((self.nparts,2))
        self.ordinate = np.zeros((self.nparts))
        for n in range(nparts):
            self.p0[n,0] = np.random.uniform(.01,.99) * length
            self.p0[n,1] = np.random.uniform(.01,.99) * width
            self.ordinate[n] = np.random.choice(self.mu)
    
    def get_weight(self):
        #xi = self.swarm / self.Area * self.dt
        #return xi / self.nparts
        xi = np.sqrt(2.5e32)
        return xi

    def Debeye_length(self, E_alpha): # eqn 33, just alphas
        lambda_d = np.sqrt((self.eps0 * self.kb * self.T_beta) / (self.n_e * (self.e * self.e)))
        return lambda_d
    
    def get_T_alpha(self,E):
        return 2 * E / (3 * self.kb)
    
    def r_perp(self,E_alpha): # eqn_34
        m_ab = self.m_alpha * self.m_beta / (self.m_alpha + self.m_beta)
        u_bar = np.sqrt(2 * E_alpha / self.m_alpha)
        r1 = (self.Z * self.e * self.e) / (4 * np.pi * self.eps0 * m_ab * u_bar * u_bar)
        r2 = self.hbar / (2 * m_ab * u_bar)
        return max(r1,r2)
    
    def get_lambda_ab(self,E_alpha):
        lambda_d = self.Debeye_length(E_alpha)
        r = self.r_perp(E_alpha)
        return np.log(lambda_d / r)
    
    def get_x(self,v2_alpha):
        x = self.m_beta * v2_alpha / (2 * self.kb * self.T_beta)
        return x
    
    @staticmethod
    def get_psi(E_alpha,x):
        # returns one part of the psi array
        # Integrate from 0 to xi
        integrand = lambda t: np.sqrt(t) / np.exp(t)
        integral, _ = quad(integrand, 0, x)
        psi = (2 / np.sqrt(np.pi)) * integral
    
        return psi
    
    @staticmethod
    def get_psi_prime(x):
        # the derivative of an integral is the function evaluated at the bounds
        return 2 / np.sqrt(np.pi) * (np.sqrt(x) / np.exp(x))

    def get_G_parms(self,E_alpha,v2_alpha):
        m_ratio = self.m_beta / self.m_alpha
        x = self.get_x(v2_alpha)
        psi = self.get_psi(E_alpha,x)
        lambda_ab = self.get_lambda_ab(E_alpha)
        psi_prime = self.get_psi_prime(x)

        # eqn 36
        G = (psi -
                m_ratio * (psi_prime - (1 / lambda_ab) *(psi + psi_prime)))
        return lambda_ab, x, G
        #return psi, lambda_ab, x, G

    def get_spower_per_cell(self,
                        I, J, # cell index
                        E, m, g, t):

        v2_alpha = (2 * E) / self.m_alpha
        w = self.get_weight()
        w2 = w * w
    
        coef = self.k * w2 * (self.Z * self.Z) * (self.e * self.e) / v2_alpha
        l_ab, x, G = self.get_G_parms(E,v2_alpha)
        self.psi[I,J,m,g,t] += 1 / self.nparts
        #psi[I,J,m,g,t] += psi_foo
        Theta = 1 if x > 1 else 0
        self.S[I,J] += coef * (G * np.log(l_ab) + Theta * np.log(1.123 * np.sqrt(x)))
        #print(self.S[I,J])
        #print(l_ab,x,G)

    def _find_group(self,E):
        if E > self.E_grid[0] or E < self.E_grid[-1]:
            raise ValueError("Energy Out of Range")
        # bins are descending
        for g in range(len(self.E_grid)-1):
            if self.E_grid[g] >= E > self.E_grid[g+1]: return g
        return len(self.E_grid)-2  # Last bin

    def _find_theta_bin(self, mu_x, mu_y):
        """Direction bin in [0, 2pi)"""
        theta = np.arctan2(mu_y, mu_x)
        if theta < 0.0:
            theta += 2.0*np.pi
        m = int(np.searchsorted(self.theta_bins, theta, side="right") - 1)
        m = max(0, min(m, len(self.theta_bins) - 2))
        return m

    def min_d_boundary(self, x, y, mu_x, mu_y):
        assert mu_x != 0
        assert mu_y != 0
        if mu_x > 0: dist_x = (np.ceil(x / self.dx) * self.dx - x) / mu_x
        else: dist_x = (x - np.floor(x / self.dx) * self.dx) / -mu_x
    
        if mu_y > 0: dist_y = (np.ceil(y / self.dx) * self.dx - y) / mu_y
        else: dist_y = (y - np.floor(y / self.dx) * self.dx) / -mu_y
    
        return min(dist_x, dist_y)

    def get_phi(self):
        self.phi = (np.sum(self.psi,axis=2) * self.d_theta)

    def TransportAlphas(self):
        if self.track_particles: self.particle_path = []
        stt = time.time()
        for history in range(self.nparts):
            if self.verbose: print(f"History Number: {history + 1}")
            else:
                if history % 10 == 0: print(f"History Number: {history + 1}")
            Transport = True
            particle = np.array([self.p0[history,0],self.p0[history,1],self.E0, self.timesteps[0]])
            theta = np.arccos(self.ordinate[history])
            mu_x = np.cos(theta)
            mu_y = np.sin(theta)
            if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2], self.timesteps[0]])
            if self.verbose: print('\n')
            m = self._find_theta_bin(mu_x, mu_y)
        
            # transport alphas
            t = 0
            counter = 0
            while Transport:
                counter += 1
                # find mesh element of particle and dist to nearest boundary along ordinate
                if self.verbose: print(particle)
                # particle outside x domain
                if particle[0] < 0 or particle[0] > self.length: break
                # particle outside y domain
                elif particle[1] < 0 or particle[1] > self.width: break
        
                I = int(particle[0] / self.dx)
                J = int(particle[1] / self.dx)
                dmin = self.min_d_boundary(particle[0], particle[1], mu_x, mu_y)
                assert dmin > 0
                E_old = particle[2]
                g = self._find_group(E_old)
        
        
                self.get_spower_per_cell(I, J, # cell index
                                E_old, m, g, t)
        
                # calculate energy loss and E deposition
                if particle[2] - self.S[I,J] * dmin > self.Eb: particle[2] -= self.S[I,J] * dmin
                else: particle[2] = self.Eb
                #print(S[I,J])
        
                # find time particle takes to cross mesh element
                v_cell = np.sqrt(2 * particle[2] / self.m_alpha)
                t_cell = dmin / v_cell
                particle[3] += t_cell
                if particle[3] > self.timesteps[-1]:
                    t = self.timesteps.size - 1
                    Transport = False
                else: t = np.searchsorted(self.timesteps, particle[3])
        
                # kill the particle, particle joins background
                if particle[2] <= self.Eb:
                    particle[2] = self.Eb
                    if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2], self.timesteps[t]])
                    Transport = False
        
                # push particle
                else:
                    particle[0] -= dmin * mu_x
                    particle[1] -= dmin * mu_y
        
                    # particle outside x domain
                    if particle[0] < 0 or particle[0] > self.length:
                        particle[0] = np.clip(particle[0], 0, self.length)
                        if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2], self.timesteps[t]])
                        break
                    # particle outside y domain
                    elif particle[1] < 0 or particle[1] > self.width:
                        particle[1] = np.clip(particle[1], 0, self.width)
                        if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2], self.timesteps[t]])
                        break
        
                    if self.track_particles: self.particle_path.append([history,particle[0],particle[1], particle[2], self.timesteps[t]])
        
        #        if counter >= 5: assert 0 == 1
            if self.verbose: print(particle)
        
        print(f"Transport Time for {self.nparts} Particles: {np.round(time.time() - stt,5)}s")

    def run(self):
        # transport
        self.starting_points()
        self.TransportAlphas()

        # visualize
        self.get_phi()
        if track_particles: self.plot_paths()
        self.animate()
        self.plot_group_flux()

    def plot_paths(self):
        norm = LogNorm(vmin=self.Eb, vmax=self.E0)
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        color_mappable = None
        path_arr = np.array(self.particle_path)
        final_vals = np.zeros((self.nparts,3))
    
        for i in range(nparts):
            mask = path_arr[:,0] == float(i)
            arr = path_arr[mask]
            x = arr[:,1]
            y = arr[:,2]
            E = arr[:,3]
            final_vals[i] = arr[-1,1:-1]
    
    
            # Build line segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                cmap=self.cmap,
                norm=norm,
                array=E[:-1],
                linewidth=2,
                zorder=2
            )
            ax.add_collection(lc)
            if color_mappable is None: color_mappable = lc
    
            # markers
            ax.scatter(x[0], y[0], color='green', s=10, marker='o', zorder=5)
            ax.scatter(x[-1], y[-1], color='red',  s=10, marker='X', zorder=5)
    
    
        cbar = plt.colorbar(color_mappable, ax=ax)
        cbar.set_label("Energy (keV)")
        legend_elements = [
            Line2D([], [], marker='o', color='green', linestyle='None', label='Start'),
            Line2D([], [], marker='X', color='red',  linestyle='None', label='End'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
        ax.set_title(f'Stopping Power Paths, dx = {dx}')
        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Width (m)')
        ax.set_xlim([0, self.length])
        ax.set_ylim([0, self.width])
        ax.grid(which='both')
    
        plt.savefig(f'fluxes/lipetrasso/paths_{nparts}_dx_{dx}.png')
    #    df = pd.DataFrame(final_vals,index=None)
    #    df.to_csv(f"final_vals_dx_{dx}.csv")
    
    def animate(self):
        # animate fluxes
        stt = time.time()
        phi = self.phi
        dt = self.dt
        for g in range(self.groups):
            print(f"Generating movie, G{g+1}")

            fig, ax = plt.subplots(figsize=(8, 6))

            # Initialize the plot with the first time step
            im = ax.imshow(phi[:, :, g, 0],
                           origin='lower',
                           extent=[0, self.length, 0, self.width],
                           vmin=0,
                           vmax=np.max(phi[:,:,g,:]),
                           cmap=self.cmap)
        
            plt.colorbar(im, label=fr'$\phi(x,y,g={g+1},t)$')
            ax.set_title(f'Energy Group {g+1} Transport')
            ax.set_xlabel('Length (m)')
            ax.set_ylabel('Width (m)')
        
            def update(t):
                im.set_array(phi[:, :, g, t])
                ax.set_title(f'Group {g+1} -- t = {float(f"{dt * t:.3g}")}s')
                return [im]

            # Create animation for this specific group
            ani = FuncAnimation(fig, update, frames=self.timesteps.size, blit=True)
    
            # Save the file
            writer = FFMpegWriter(fps=36, bitrate=2000, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-loop', '0'])
            filename = f"fluxes/lipetrasso/phi_t_g{g+1}.mp4"
            ani.save(filename, writer=writer)
    
            # Close the figure to free up memory before the next group
            plt.close(fig)
    
        print(f"Animation Time: {np.round(time.time() - stt, 5)}s")

    def plot_group_flux(self):
        # plot flux
        for g in range(self.groups):
            t_arg = np.searchsorted(self.timesteps, self.t_f / 10)
            plt.figure(figsize=(8,6))
            plt.imshow(self.phi[:,:,g,t_arg], cmap=self.cmap,origin="lower", extent=[0,self.length,0,self.width],vmin=0)
            plt.colorbar()
            plt.title(fr"Stopping Power $\phi(x,y,g={g+1})$ at t={t_f/10:3g}s")
            plt.savefig(f"fluxes/lipetrasso/phi_g{g+1}.png")

