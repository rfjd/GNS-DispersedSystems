import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['text.usetex'] = True

p = 6;
U0 = 0.5;
rm = 0.5;
R  = 2*np.power(2,1/p)  ;
dUdrm = -4*p*U0/rm*(2*np.power(2/rm,2*p)-np.power(2/rm,p)); # dU(rm)/dr
r = np.linspace(0.1,10,100)
F = -dUdrm*(r<=rm)+4*p*U0*(2*np.power(2/r,2*p)-np.power(2/r,p))/r*np.logical_and(rm<r, r<=R)
print(f'(r,F):\n {np.column_stack((r,F))}')
print(f'r_m={rm}')
print(f'(2^1/p)(2a)={np.power(2,1/p)*2}')

Lx = 5;# in
Ly = 5;# in
plt.figure(figsize=[Lx, Ly])
ax = plt.axes([0.15, 0.15, 0.8, 0.8])
plt.xlabel("$r$")
plt.ylabel("$F_{\mathrm{LJ}}(r)$")
plt.plot(r,F)
plt.show()
