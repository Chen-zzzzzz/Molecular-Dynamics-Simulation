import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
# import pandas as pd
import tqdm

# filename = 'liquid256.txt'
filename = '/home/chenzhang/10703/10703_1/12623/liquid256.txt'
atom_type = 'O'
dt = 0.01
mass = 1
steps = 500
cutoff = 2.5
cell_length = 6.8
k_b = 1  # Boltzmann constant in reduced units


def import_file(filename):
    f = open(filename, 'r')
    filelist = f.read().splitlines()
    atom_count = len(filelist)
    pos = np.zeros((atom_count, 3))
    for line in range(0, atom_count):
        coords_list = filelist[line].split('\t')
        coords = [float(i) for i in coords_list if i]
        pos[line] = coords
    return pos, atom_count

# @nb.njit
def init_vel(atom_count,temperature):
    # Initialize velocities randomly
    # vel = np.random.randn(atom_count, 3)
    
    # # Adjust velocities for zero total momentum
    # total_momentum = np.sum(vel, axis=0)
    # correction = total_momentum / atom_count
    # vel -= correction
    # return vel
    half_vel = np.random.normal(0,np.sqrt(1.0*temperature),size = (atom_count//2,3))
    vel = np.concatenate([half_vel, - half_vel], axis = 0)
    return vel

@nb.njit
def apply_pbc(p, cell_length):
    # print(p)
    pos = np.mod(p, cell_length)
    return pos

def calculate_kinetic(vel):
    vel = np.array(vel)
    ke = np.sum(vel**2)/2
    return ke

# @nb.njit
def verlet(p, v, f):
    pos_out = np.zeros((atom_count, 3))
    vel_half = np.zeros((atom_count, 3))
    vel_out = np.zeros((atom_count, 3))
    for atom in range(0, atom_count):
        p_init = p[atom]
        v_init = v[atom]
        f_init = f[atom]
        v_half = update_vel(v_init, f_init)
        p_new = update_pos(p_init, v_half)
        p_new = apply_pbc(p_new, cell_length)
        vel_half[atom] = v_half
        pos_out[atom] = p_new
    
    f_out, pot_out, pre = calculate_forces(pos_out,  cell_length)

    for atom in range(0, atom_count):
        vel_out[atom] = update_vel(vel_half[atom], f_out[atom])

    ke = calculate_kinetic(vel_out)
    temp = 2*ke/(3*(atom_count-1))
    pre_with_inst = pre + atom_count*temp/(cell_length**3)
    return pos_out, vel_out, f_out, pot_out, ke, temp, pre_with_inst

# @nb.njit
def calculate_forces(p, cell_length):
    forces = np.zeros((atom_count, 3))
    cutoff = 2.5
    epsilon = 1e-5  # Small value to prevent division by zero

    # Compute the Lennard-Jones potential at r_c
    u_LJ_rc = 4 * ((1/cutoff**12) - (1/cutoff**6))

    # Shift value for the Lennard-Jones force
    F_shift = 24 * ((2/cutoff**13) - (1/cutoff**7))
    u = 0.
    pre = 0.

    # inst
    p_inst = 0.
    for atom in range(atom_count):
        for other in range(atom + 1, atom_count):
            rx, ry, rz, r = calculate_dist(p, atom, other)

            # Apply periodic boundary conditions
            if rx > cell_length / 2: rx -= cell_length
            if rx < -cell_length / 2: rx += cell_length
            if ry > cell_length / 2: ry -= cell_length
            if ry < -cell_length / 2: ry += cell_length
            if rz > cell_length / 2: rz -= cell_length
            if rz < -cell_length / 2: rz += cell_length
            r = np.sqrt(rx**2 + ry**2 + rz**2)

            if r < epsilon: 
                continue  # Skip computation for too close particles

            if r < cutoff:
                # Derivative of the Lennard-Jones potential with the shifted function
                F_LJ = 24 * ((2/r**13) - (1/r**7))

                # shift
                F_SF = -(F_LJ - F_shift)
                Fx, Fy, Fz = F_SF * rx / r, F_SF * ry / r, F_SF * rz / r
                forces[atom] += [Fx, Fy, Fz]
                forces[other] -= [Fx, Fy, Fz]

                u_LJ_r =  4 * ((1/r**12) - (1/r**6))
                # Shifted Force potential at r
                u_SF_r = u_LJ_r - (r - cutoff) * -F_shift - u_LJ_rc
                u += u_SF_r

                pre_wo = (Fx*rx + Fy*ry + Fz*rz)/(3*(cell_length**3))
                pre += pre_wo

    return forces, u, pre

@nb.njit
def pbc_delta(x, L):
    """Apply periodic boundary conditions."""
    if x > 0.5 * L:
        return x - L
    elif x < -0.5 * L:
        return x + L
    return x

def nearest_image_dist(pos1, pos2, L):
    delta = pos1 - pos2
    delta = np.array([pbc_delta(d, L) for d in delta])
    return np.linalg.norm(delta)


# def calculate_dist(p, atom, other):
#     # p = np.array(p)
#     # if rx > cell_length / 2: rx -= cell_length
#     # if rx < -cell_length / 2: rx += cell_length
#     # if ry > cell_length / 2: ry -= cell_length
#     # if ry < -cell_length / 2: ry += cell_length
#     # if rz > cell_length / 2: rz -= cell_length
#     # if rz < -cell_length / 2: rz += cell_length
#     rx = pbc_delta(p[other, 0] - p[atom, 0], cell_length)
#     ry = pbc_delta(p[other, 1] - p[atom, 1], cell_length)
#     rz = pbc_delta(p[other, 2] - p[atom, 2], cell_length)
#     r = (rx**2 + ry**2 + rz**2)**0.5
#     return rx, ry, rz, r
# @nb.njit
def calculate_dist(p, atom, other):
    # print(p[other])
    # exit()
    # rx = p[other, 0] - p[atom, 0]
    # ry = p[other, 1] - p[atom, 1]
    # rz = p[other, 2] - p[atom, 2]
    rx = p[other, 0] - p[atom, 0]
    ry = p[other, 1] - p[atom, 1]
    rz = p[other, 2] - p[atom, 2]
    rx -=  cell_length * np.round(rx/cell_length)
    ry -=  cell_length * np.round(ry/cell_length)
    rz -=  cell_length * np.round(rz/cell_length)
    r = (rx**2 + ry**2 + rz**2)**0.5
    return rx, ry, rz, r



def instantaneous_temperature(v):
    # Kinetic energy per particle
    KE = 0.5 * mass * np.sum(np.square(v)) / (atom_count-1)
    return 2.0 * KE / (3 * k_b)

def instantaneous_pressure(p, v, L):
    # Calculate using ideal gas law. More detailed calculations would require more details about the system.
    T = instantaneous_temperature(v)
    rho = atom_count / L**3
    return rho * k_b * T

@nb.njit
def update_vel(v, f):
    v_new = np.zeros(3)
    for dir in range(0, 3):
        v_new[dir] = v[dir] + (dt / 2) * (f[dir] / mass)
    return v_new

@nb.njit
def update_pos(p, v):
    p_new = np.zeros(3)
    for dir in range(0, 3):
        p_new[dir] = p[dir] + dt * v[dir]
    return p_new

def run_md(p, v, f, steps):
    all_positions = [p]
    all_velocities = [v]
    all_forces = [f]
    all_potentials = []
    # all_temp = []
    # all_pressures = []
    ke_list = []
    all_temp = []
    all_pre = []
    pbar = tqdm.tqdm(range(steps))
    for step in pbar:
        p, v, f, pot, ke, temp_inst, pre = verlet(p, v, f)
        ke_list.append(ke)
        temp = 2*ke/(3*(atom_count-1))
        # temp = instantaneous_temperature(v)
        # pre = instantaneous_pressure(p, v, cell_length)
        all_positions.append(p)
        all_velocities.append(v)
        all_forces.append(f)
        all_potentials.append(pot)
        pbar.set_postfix_str({f'Potential': np.round(pot,2),'Ke':np.around(ke,2),'Temp':np.around(temp,2),'Pressure':np.around(pre,2)})
        all_temp.append(temp)
        all_pre.append(pre)
        # all_temp.append(temp)
        # all_pressures.append(pre)
    return all_positions, all_velocities, all_forces, all_potentials, ke_list, all_temp, all_pre

# def calculate_energies(v):
#     e_out = []
#     px_out = []
#     py_out = []
#     pz_out = []
#     for timestep in v_out:
#         e_sum = 0
#         px_sum = 0
#         py_sum = 0
#         pz_sum = 0
#         for atom in timestep:
#             v_x = atom[0]
#             v_y = atom[1]
#             v_z = atom[2]
#             v = (v_x ** 2 + v_y ** 2 + v_z **2) ** 0.5
#             e_sum += (1 / 2) * mass * (v **2)
#             px_sum += mass * v_x
#             py_sum += mass * v_y
#             pz_sum += mass * v_z
#         e_out.append(e_sum)
#         px_out.append(px_sum)
#         py_out.append(py_sum)
#         pz_out.append(pz_sum)
#     return e_out, px_out, py_out, pz_out

# def calculate_potentials(p, cell_length):
    # total_energy = 0
    # cutoff = 2.5

    # # Compute the Lennard-Jones potential and its derivative at r_c
    # u_LJ_rc = 4 * ((1/cutoff**12) - (1/cutoff**6))
    # f_LJ_rc = 24 * ((2/cutoff**13) - (1/cutoff**7))

    # for atom in range(atom_count):
    #     for other in range(atom + 1, atom_count):
    #         rx, ry, rz, r = calculate_dist(p, atom, other)

    #         # Apply periodic boundary conditions
    #         if rx > cell_length / 2: rx -= cell_length
    #         if rx < -cell_length / 2: rx += cell_length
    #         if ry > cell_length / 2: ry -= cell_length
    #         if ry < -cell_length / 2: ry += cell_length
    #         if rz > cell_length / 2: rz -= cell_length
    #         if rz < -cell_length / 2: rz += cell_length
    #         r = np.sqrt(rx**2 + ry**2 + rz**2)

    #         if r < cutoff:
    #             # Lennard-Jones potential at r
    #             u_LJ_r = 4 * ((1/r**12) - (1/r**6))

    #             # Shifted Force potential at r
    #             u_SF_r = u_LJ_r - (r - cutoff) * f_LJ_rc - u_LJ_rc
    #             total_energy += u_SF_r

    # return total_energy

# def calculate_potentials(p, cell_length):
#     p = np.array(p)
#     cutoff = 2.5
#     u = []
#     # Compute the Lennard-Jones potential and its derivative at r_c
#     u_LJ_rc = 4 * ((1/cutoff**12) - (1/cutoff**6))
#     f_LJ_rc =  24 * ((2/cutoff**13) - (1/cutoff**7))

#     for t in range(steps):
#         # exit()
#         total_energy = 0
#         for atom in range(atom_count):
#             for other in range(atom + 1, atom_count):
                
#                 rx, ry, rz, r = calculate_dist(p[t], atom, other)
#                 # print(p.shape)

#                 # Apply periodic boundary conditions
                

#                 # if rx > cell_length / 2: rx -= cell_length
#                 # if rx < -cell_length / 2: rx += cell_length
#                 # if ry > cell_length / 2: ry -= cell_length
#                 # if ry < -cell_length / 2: ry += cell_length
#                 # if rz > cell_length / 2: rz -= cell_length
#                 # if rz < -cell_length / 2: rz += cell_length
#                 # r = np.sqrt(rx**2 + ry**2 + rz**2)

#                 if r < cutoff:
#                     # Lennard-Jones potential at r
#                     u_LJ_r = 4 * ((1/r**12) - (1/r**6))

#                     # Shifted Force potential at r
#                     u_SF_r = u_LJ_r - (r - cutoff) * -f_LJ_rc - u_LJ_rc

#                     total_energy += u_SF_r
#             u.append(total_energy)
#     return u
    #     for atom in range(atom_count):
    #         for other in range(atom + 1, atom_count):
    #             rx, ry, rz, r = calculate_dist(p, atom, other)

    #             if r < cutoff:
    #                 # Lennard-Jones potential at r
    #                 u_LJ_r = 4 * ((1/r**12) - (1/r**6))

    #                 # Shifted Force potential at r
    #                 u_SF_r = u_LJ_r - (r - cutoff) * f_LJ_rc - u_LJ_rc
    #                 total_energy += u_SF_r

    # return total_energy


def calculate_hamiltonian(e, pot):
    h = []
    for timestep in range(0, steps):
        h.append(e[timestep] + pot[timestep])
    return h

def export_file(p):
    with open('output.xyz', 'w') as out:
        for time in p:
            out.write(str(atom_count) + '\n')
            out.write('dt: ' + str(dt) + '\n')
            for atom in time:
                atom_string = atom_type
                for dir in range(0, 3):
                    atom_string += ' ' + str(atom[dir])
                out.write(atom_string)
                out.write('\n')

# def plot_all(v, p):
#     #calculate kinetic energy, potential, and hamiltonian
#     e_out, px_out, py_out, pz_out = calculate_energies(v)
#     pot_out = calculate_potentials(p, cell_length)
#     h_out = calculate_hamiltonian(e_out, pot_out)
#     #write pot_out, e_out, h_out to .txt
#     np.savetxt('pot_out.txt', pot_out)
#     np.savetxt('e_out.txt', e_out)
#     np.savetxt('h_out.txt', h_out)


#     #Plot kinetic energy, potential, and hamiltonian
#     plt.plot(pot_out, label='Potential')
#     plt.plot(e_out, label='Kinetic Energy')
#     plt.plot(h_out, label='Hamiltonian')
#     plt.legend()
#     plt.xlabel("Timestep (dimensionless)")
#     plt.ylabel("Energy (dimensionless)")
#     plt.savefig('energy')
#     plt.show()
#     plt.close()

#     #Plot directional momenta 
#     plt.plot(px_out, label='x-momentum')
#     plt.plot(py_out, label='y-momentum')
#     plt.plot(pz_out, label='z-momentum')
#     plt.legend()
#     plt.xlabel("Timestep (dimensionless)")
#     plt.ylabel("Momentum (dimensionless)")
#     plt.savefig('momentum')
#     plt.show()
#     plt.close()



tic = time.time()

#import and initialize positions
init_positions, atom_count = import_file(filename)

#initialize velocities
init_velocities = init_vel(atom_count,temperature=0.9)

#calculate forces (CHANGE THIS)
init_forces, init_potential, pre= calculate_forces(init_positions,  cell_length)

#run MD
p_out, v_out, f_out, pot, ke, temps, pre = run_md(init_positions, init_velocities, init_forces, steps)
plt.figure()
plt.plot(np.array(pot),label = 'Potential')
plt.plot(np.array(ke),label = 'Kinetic')
plt.plot(np.array(pot)+np.array(ke),label = 'Hamiltonian')
plt.plot()
plt.legend()
plt.savefig('Energy.png')

#calculate temperature
# temps = [instantaneous_temperature(v) for v in v_out]
np.savetxt('temps.txt', temps)
plt.figure()
plt.plot(temps, label='CF Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.savefig('temperature.png')
# plt.show()
plt.close()

# #calculate pressures
# pressures = [instantaneous_pressure(p, v, cell_length) for p, v in zip(p_out, v_out)]
np.savetxt('pressures.txt', pre)
plt.figure()
plt.plot(pre, label='CF Pressure')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.savefig('pressure.png')
# plt.show()
plt.close()

# #export xyz
# export_file(p_out)

# toc = time.time()
# elapsed = toc - tic
# print(f'Execution Time: {elapsed} seconds')
# # p_out = np.array(p_out)
# # print(f'p out: {p_out.shape}')
# #plot
# # plot_all(p_out, v_out)

