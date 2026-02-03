from models import *

if model == 'fourier':
    sp_mod = FourierStoppingPower(
                    nparts, swarm, # particles
                    num_ords,mu, theta_bins, d_theta, # direction
                    groups, E_grid, E0, Eb, # energy
                    length, width, dx, Area, eps, # geometry
                    track_particles,verbose # debug
                    )

elif model == 'bc':
    sp_mod = BcStoppingPower(
                    nparts, swarm, # particles
                    num_ords,mu, theta_bins, d_theta, # direction
                    groups, E_grid, E0, Eb, # energy
                    length, width, dx, Area, eps, # geometry
                    track_particles,verbose # debug
                    )

elif model == 'lipetrasso':
    sp_mod = LiPetrassoStoppingPower(
                    nparts, swarm, # particles
                    num_ords,mu, theta_bins, d_theta, # direction
                    groups, E_grid, E0, Eb, # energy
                    length, width, dx, Area, eps, # geometry
                    t_f,
                    track_particles,verbose # debug
                    )

else: raise ValueError("only fourier and bc model is implimented")

sp_mod.run()

