# Aref Hashemi & Aliakbar Izadkhah (2024)
# This code simulates a two-dimensional mixture of point particles in a box

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['text.usetex'] = True

s = 2e-3
rho = 2300
rho_f = 1000
g = 9.8066
L = 10e-2
k = 1.25

p = 6
U0_coeff = 0.025 #
r_m_coeff = 1.75 # multiplied by min[a_i,a_j] gives r_m
R_coeff = np.power(2,1/p) # multiplied by sigma gives R

gamma = 0.8

# randomly place N particles
import numpy as np

def generate_disk_positions(L, y0, N, a, max_attempts=1e6):
    def is_overlapping(new_disk, new_radius, disks, radii):
        for disk, radius in zip(disks, radii):
            if np.linalg.norm(new_disk - disk) < new_radius + radius:
                return True
        return False

    disks = []
    attempts = 0
    
    while len(disks) < N:
        if attempts >= max_attempts:
            raise ValueError(f"Exceeded maximum number of attempts ({max_attempts}) to place particles without overlap.")
        
        x = np.random.uniform(-L + a[len(disks)], L - a[len(disks)])
        y = np.random.uniform(y0 + a[len(disks)], L - a[len(disks)])
        new_disk = np.array([x, y])
        new_radius = a[len(disks)]
        
        if not is_overlapping(new_disk, new_radius, disks, a[:len(disks)]):
            disks.append(new_disk)
            attempts = 0  # Reset attempts counter after a successful placement
        else:
            attempts += 1
            
    return np.array(disks)

### Total pair interaction force on particle i
def dUdr(r,sigma,U0):
    return -4*p*U0/r*(2*np.power(sigma/r,2*p)-np.power(sigma/r,p))

def Fi(x,a,i):
    # given position of the particles, x with shape (N,2) and the radii of the particles, a with shape (N,), this function calculates the force on the ith particle due to LJ steric potential. The output has shape (2,)
    N = len(x);
    diff = x[i,:]-x # shape (N,2)
    sigma = (a[i]+a).reshape((N,1)); # shape (N,1)
    U0 = U0_coeff*np.power(sigma,2); # shape (N,1)
    r = np.sqrt(np.power(diff[:,0],2)+np.power(diff[:,1],2)).reshape((N,1)); # shape (N,1)
    # remove the ith rows from diff and r vectors (to avoid division by zero)
    rMod = np.delete(r,i,0); #shape (N-1,1)
    diffMod = np.delete(diff,i,0); #shape (N-1,2)
    sigmaMod = np.delete(sigma,i,0); #shape (N-1,1)
    U0Mod = np.delete(U0,i,0); #shape (N-1,1)

    # logical conditions to find the vector of forces
    r_m = r_m_coeff*np.minimum(a[i],a).reshape((N,1)) # shape (N,1)
    R = 2*R_coeff*np.maximum(a[i],a).reshape((N,1)) # shape (N,1)
    r_mMod = np.delete(r_m,i,0); #shape (N-1,1)
    RMod = np.delete(R,i,0); #shape (N-1,1)
    ForceVec = -dUdr(r_mMod,sigmaMod,U0Mod)*diffMod/rMod*(rMod<=r_mMod)-dUdr(rMod,sigmaMod,U0Mod)*diffMod/rMod*(rMod>r_mMod)#*(rMod<=RMod) # shape (N-1,2)
    return np.sum(ForceVec,axis=0);

def FVec(x,a):
    N = len(x);
    DIM = (N,2);
    FVec = np.zeros(DIM);
    for i in range(N):
        FVec[i,:] = Fi(x,a,i);

    return FVec # shape (N,2)

ex = np.array([1,0])
ey = np.array([0,1])
def collision(wallpos, direction, nhat, dt, x, v, a, index, F_ext):
    dt_c = (wallpos-x[index,direction])/v[index,direction]
    dt_r = dt-dt_c
    xc  = x[index,:]+dt_c*v[index,:]
    vbc = v[index,:]+dt_c/(np.pi*a[index]**2*s*rho)*(-k*a[index]*v[index,:]+(F_ext[index,:]+Fi(x,a,index)))
    vac = np.sqrt(gamma)*(vbc-2*(np.dot(vbc,nhat))*nhat)
    x[index,:] = xc
    vNew = vac+dt_r/(np.pi*a[index]**2*s*rho)*(-k*a[index]*v[index,:]+(F_ext[index,:]+Fi(x,a,index)))
    xNew = xc+dt_r*vac

    return (vNew,xNew)


### Simulation
multidisperse = True
# a_values = [2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3, 4.5e-3, 5e-3]
# a_values = [2e-3, 2.25e-3, 2.75e-3, 3.25e-3, 3.75e-3, 4.25e-3, 4.75e-3, 5e-3]
a_values = [2.75e-3, 4.25e-3]
data_category = 'test'
if data_category == 'test':
    np.random.seed(123)
    if multidisperse:
        num_trajectories = 6
    else:
        num_trajectories = len(a_values)
else:
    np.random.seed(1)
    num_trajectories = 16
    
save_step = 15
for tr in range(num_trajectories):
    packing = 0.7
    y0 = np.random.uniform(-0.25*L,0,1).item()
    if multidisperse:
        Nmax = 2*L*(L-y0)/(4*max(a_values)**2)
        Nscale = int(packing*Nmax)
        N = np.random.randint(Nscale, Nscale+4, 1).item()
        a = np.random.choice(a_values, size=(N,))
    else:
        if data_category=="test":
            chosen_value = a_values[tr]
        else:
            chosen_value = np.random.choice(a_values)

        Nmax = 2*L*(L-y0)/(4*chosen_value**2)
        Nscale = packing*Nmax
        N = np.random.randint(Nscale, Nscale+4, 1).item()
        a = np.full(N, chosen_value)
    
    DIM = (N,2);
    F_ext = np.zeros(DIM);
    F_ext[:,1] = -(rho-rho_f)*s*np.pi*a**2*g;
    
    x = generate_disk_positions(L,y0,N,a)
    vscale = (rho-rho_f)*np.pi*np.min(a)*s*g/k
    vp0x = np.random.uniform(-vscale,vscale)
    vp0y = 0.25*np.random.uniform(-vscale,vscale)
    v = np.tile([vp0x,vp0y],(N,1))

    # print(f'v[:10,:]=\n{v[:10,:]}')
    Lx = 5;# in
    Ly = 5;# in
    plt.figure(figsize=[Lx, Ly])
    ax = plt.axes([0.15, 0.15, 0.8, 0.8], xlim=(-1, 1), ylim=(-1, 1))
    points_whole_ax = Lx*0.8*72# 1 point = dpi / 72 pixels
    points_radius = a/L*points_whole_ax
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xticks([-1,-0.5,0,0.5,1])
    plt.yticks([-1,-0.5,0,0.5,1])
    curve = ax.scatter(10, 10, s=10, color='r')

    dt = 0.0001; # time step
    tf = 0.75  ; # total simulation time
    tM = int(tf/dt); # number of time steps
    t = np.linspace(0,tf,tM);
    position = np.zeros((tM,DIM[0],DIM[1]))
    position[0,:,:] = x
    particle_properties = np.full((N,), a/L)
    for n in range(tM-1):
        vNew = v+dt/(np.pi*(a.reshape(N,1))**2*s*rho)*(-k*(a.reshape(N,1))*v+(F_ext+FVec(x,a)))
        xNew = x+dt*v

        # Boundary conditions
        indices_hix, indices_lox = np.where(xNew[:,0]>L-a), np.where(xNew[:,0]<-L+a)
        indices_hiy, indices_loy = np.where(xNew[:,1]>L-a), np.where(xNew[:,1]<-L+a)
        # Right wall
        if indices_hix[0].size>0:
            for index in indices_hix[0]:
                (vNew[index,:],xNew[index,:]) = collision(L-a[index], 0, -ex, dt, x, v, a, index, F_ext)
        # Left wall
        if indices_lox[0].size>0:
            for index in indices_lox[0]:
                (vNew[index,:],xNew[index,:]) = collision(-L+a[index], 0, ex, dt, x, v, a, index, F_ext)
        # Top wall
        if indices_hiy[0].size>0:
            for index in indices_hiy[0]:
                (vNew[index,:],xNew[index,:]) = collision(L-a[index], 1, -ey, dt, x, v, a, index, F_ext)
        # Bottom wall
        if indices_loy[0].size>0:
            for index in indices_loy[0]:
                (vNew[index,:],xNew[index,:]) = collision(-L+a[index], 1, ey, dt, x, v, a, index, F_ext)

        v, x = vNew, xNew
        position[n+1,:,:] = x
        # print(format(t[n], '.2f'))
        curve.remove()
        t_pause = 0.001;
        print(f'time={(n+1)*dt:.4f}')
        curve = ax.scatter(x[:,0]/L,x[:,1]/L, s=points_radius**2, color='r', edgecolor='k')
        
        plt.pause(t_pause)


    plt.show(block=False)
    plt.close()
    
    data_tuple = np.empty(2, dtype=object)
    data_tuple[0] = position[0:-1:save_step,:,:]/L # normalized
    data_tuple[1] = particle_properties
    np.save(f"{data_category}{tr}.npy", data_tuple)


data_dict = {}
for tr in range(num_trajectories):
    # Load the data from each .npy file
    data = np.load(f"{data_category}{tr}.npy", allow_pickle=True)
    
    # Assign the loaded data to the dictionary with the desired naming convention
    data_dict[f"simulation_trajectory_{tr}"] = data

# Save all data into a single .npz file
import os
np.savez(f"{data_category}.npz", **data_dict)
os.system("rm *.npy")

# def compute_mean_std(file_path):
#     # Load data from the .npz file
#     data = np.load(file_path, allow_pickle=True)
    
#     all_velocities = []
#     all_accelerations = []
    
#     for key in data.files:
#         ds = data[key]
#         positions = ds[0]
        
#         # Compute velocities and accelerations for each trajectory
#         velocities = positions[1:] - positions[:-1]
#         accelerations = positions[2:] - 2 * positions[1:-1] + positions[:-2]
        
#         # Flatten the time step and number of particles dimensions    
#         velocities_flat = velocities.reshape(-1, 2)
#         accelerations_flat = accelerations.reshape(-1, 2)
        
#         # Append to the global lists
#         all_velocities.append(velocities_flat)
#         all_accelerations.append(accelerations_flat)
    
#     # Concatenate all velocities and accelerations
#     all_velocities = np.concatenate(all_velocities, axis=0)
#     all_accelerations = np.concatenate(all_accelerations, axis=0)
    
#     # Calculate mean and std
#     velocity_mean = np.mean(all_velocities, axis=0)
#     velocity_std = np.std(all_velocities, axis=0)
    
#     acceleration_mean = np.mean(all_accelerations, axis=0)
#     acceleration_std = np.std(all_accelerations, axis=0)
    
#     return {
#         'velocity_mean': velocity_mean,
#         'velocity_std': velocity_std,
#         'acceleration_mean': acceleration_mean,
#         'acceleration_std': acceleration_std
#     }


if data_category == 'train':
    # stats = compute_mean_std('train.npz')
    import json
    data = np.load('train.npz', allow_pickle=True)
    radii = []
    for example in range(len(data.files)):
        radii.append(data[f'simulation_trajectory_{example}'][1])

    max_radius = np.max(np.concatenate(radii))
    default_connectivity_radius = 4*max_radius
    dt_save = save_step*dt
    # Define the metadata dictionary
    metadata = {
        "bounds": [[-1, 1], [-1, 1]],
        "dt": dt_save,
        "default_connectivity_radius": default_connectivity_radius,
        "boxSize": L
        # "vel_mean": stats['velocity_mean'].tolist(),
        # "vel_std": stats['velocity_std'].tolist(),
        # "acc_mean": stats['acceleration_mean'].tolist(),
        # "acc_std": stats['acceleration_std'].tolist()
    }

    # Write the metadata dictionary to a JSON file
    with open('metadata.json', 'w') as json_file:
        json.dump(metadata, json_file)
