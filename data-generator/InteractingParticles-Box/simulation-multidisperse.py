# Aref Hashemi & Aliakbar Izadkhah (2024)
# This code simulates a two-dimensional mixture of point particles in a box

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['text.usetex'] = True

s = 2e-3
rho = 1200
rho_f = 1000
g = 9.8066
L = 10e-2
k = 1

p = 4
U0 = 1e-6 #
r_m_coeff = 1.3 # multiplied by min[a_i,a_j] gives r_m
R_coeff = 2 # multiplied by sigma gives r_m

gamma = 0.8

# places N particles in a rectangular box of (lo) -- (hi)
def generate_equispaced_positions(N, lo, hi):
    L = hi[0]-lo[0]
    H = hi[1]-lo[1]
    # Determine the number of points along each axis
    n_cols = int(np.ceil(np.sqrt(N*(L/H))))
    n_rows = int(np.ceil(N/n_cols))

    # Generate equispaced points along each axis
    x = np.linspace(lo[0], hi[0], n_cols, endpoint=False)
    y = np.linspace(lo[1], hi[1], n_rows, endpoint=False)
    
    # Create a grid of points
    xv, yv = np.meshgrid(x, y)
    grid_points = np.vstack([xv.ravel(), yv.ravel()]).T

    # Select the first N points
    positions = grid_points[:N]

    return positions

### Total pair interaction force on particle i
def dUdr(r,sigma):
    return -4*p*U0/r*(2*np.power(sigma/r,2*p)-np.power(sigma/r,p))

def Fi(x,a,i):
    # given position of the particles, x with shape (N,2) and the radii of the particles, a with shape (N,), this function calculates the force on the ith particle due to LJ steric potential. The output has shape (2,)
    N = len(x);
    diff = x[i,:]-x # shape (N,2)
    sigma = (a[i]+a).reshape((N,1)); # shape (N,1)
    r = np.sqrt(np.power(diff[:,0],2)+np.power(diff[:,1],2)).reshape((N,1)); # shape (N,1)
    # remove the ith rows from diff and r vectors (to avoid division by zero)
    rMod = np.delete(r,i,0); #shape (N-1,1)
    diffMod = np.delete(diff,i,0); #shape (N-1,2)
    sigmaMod = np.delete(sigma,i,0); #shape (N-1,1)

    # logical conditions to find the vector of forces
    r_m = r_m_coeff*np.minimum(a[i],a).reshape((N,1)) # shape (N,1)
    R = 2*R_coeff*np.maximum(a[i],a).reshape((N,1)) # shape (N,1)
    r_mMod = np.delete(r_m,i,0); #shape (N-1,1)
    RMod = np.delete(R,i,0); #shape (N-1,1)
    ForceVec = -dUdr(r_mMod,sigmaMod)*diffMod/rMod*(rMod<=r_mMod)-dUdr(rMod,sigmaMod)*diffMod/rMod*(rMod>r_mMod)*(rMod<=RMod) # shape (N-1,2)
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

num_trajectories = 16
data_category = 'train'
np.random.seed(1) # train: 1; valid: 123; test: 12357
save_step = 2
for tr in range(num_trajectories):
    N = np.random.randint(30, 35, 1).item()
    a_values = [0.5e-2, 1e-2]
    a = np.random.choice(a_values, size=(N,))
    a_ref = np.max(a)
    DIM = (N,2);
    F_ext = np.zeros(DIM);
    F_ext[:,1] = -(rho-rho_f)*s*np.pi*a**2*g;
    lo = [np.random.uniform(-L+a_ref,-a_ref,1)[0],np.random.uniform(-0.75*L,-0.5*L,1)[0]]
    box_sizex = np.random.uniform(0.75*L,L,1)[0]
    box_sizey = 4*a_ref**2*N/(box_sizex)
    if lo[1]+box_sizey > L-a_ref:
        lo[1] -= lo[1]+box_sizey-(L-a_ref)
    hi = np.add(lo,np.array([box_sizex,box_sizey]))
    x = generate_equispaced_positions(N,lo,hi)
    vp0x = 0.5*np.random.uniform(-np.power(2*L,0.5),np.power(2*L,0.5))
    vp0y = 0
    v = 2*np.tile([vp0x,vp0y],(N,1))

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

    dt = 0.001; # time step
    tf = 1.5  ; # total simulation time
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

if data_category == 'train':
    import json
    data = np.load('train.npz', allow_pickle=True)
    radii = []
    for example in range(len(data.files)):
        radii.append(data[f'simulation_trajectory_{example}'][1])

    max_radius = np.max(np.concatenate(radii))
    default_connectivity_radius = 2*R_coeff*max_radius
    # Define the metadata dictionary
    metadata = {
        "bounds": [[-1, 1], [-1, 1]],
        "dt": step*dt,
        "default_connectivity_radius": default_connectivity_radius,
        "boxSize": L
    }

    # Write the metadata dictionary to a JSON file
    with open('metadata.json', 'w') as json_file:
        json.dump(metadata, json_file)
