import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

def Vp_over_EHA(d, sign):
    # sign = +1 for bonding, -1 for antibonding.
    # Expression in atomic units (a0 = 1)
    num = (1/d) - (1 + 1/d)*np.exp(-2*d) + sign*(1+d)*np.exp(-d)
    den = 1 + sign*(1 + d + d**2/3)*np.exp(-d)
    return -1 + (1/d) - 2*num/den

x = np.linspace(0.01, 5, 100)
# Compute for bonding (plus sign) and antibonding (minus sign) states.
Vp_bond = Vp_over_EHA(x, sign=+1)
Vp_anti = Vp_over_EHA(x, sign=-1)


plt.plot(x, Vp_anti, label=r'$\epsilon_a$', lw=2)
plt.plot(x, Vp_bond, label=r'$\epsilon_b$', lw=2, ls='--')
plt.xlim(0, 5)
plt.ylim(-5, 5)
plt.xlabel(r'$\frac{d}{a_0}$')
plt.ylabel(r'$V_p/E_{HA}$')
plt.title(r'$V_p/E_{HA}$ vs $\frac{d}{a_0}$')
plt.legend()
plt.grid()
plt.savefig('hw5_q1_d.png')
plt.show()

