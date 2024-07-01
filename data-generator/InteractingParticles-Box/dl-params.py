import numpy as np
a = 0.1e-3
rho_p = 2300 # silicon
eta = 1e-3
g = 9.8066
alpha = eta/(np.power(g,0.5)*np.power(a,1.5)*rho_p)
print(f"alpha={alpha}")
