import numpy as np
import matplotlib.pyplot as plt

a = 0.4
b = 2.0
V0 = 1.0
S = 10.0
gamma = a / b

v_even = 3.46525
A_even = np.cos(v_even)
lambda_decay = np.sqrt(S**2 - v_even**2) / (b - a)

v_odd = 3.49913
A_odd = np.sin(v_odd)
mu_decay = np.sqrt(S**2 - v_odd**2) / (b - a)

def potential(x):
    V = np.zeros_like(x)
    cond1 = np.abs(x) < a
    cond2 = (np.abs(x) >= a) & (np.abs(x) <= b)
    V[cond1] = 0.0
    V[cond2] = -V0
    return V

def psi_even(x):
    x_abs = np.abs(x)
    return np.where(x_abs < a,
                    np.cos(v_even * x_abs / a),
                    np.where(x_abs <= b,
                             A_even * np.exp(-lambda_decay*(x_abs - a)),
                             0.0))
    
def psi_odd(x):
    x_abs = np.abs(x)
    return np.where(x_abs < a,
                    np.sign(x) * np.sin(v_odd * x_abs / a),
                    np.where(x_abs <= b,
                             np.sign(x) * A_odd * np.exp(-mu_decay*(x_abs - a)),
                             0.0))

x_vals = np.linspace(-b, b, 800)
psi_even_vals = psi_even(x_vals)
prob_even = psi_even_vals**2
psi_odd_vals = psi_odd(x_vals)
prob_odd = psi_odd_vals**2
V_vals = potential(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, psi_even_vals, 'b-', label=r'$\psi_{even}(x)$')
plt.plot(x_vals, prob_even, 'r--', label=r'$|\psi_{even}(x)|^2$')
plt.plot(x_vals, V_vals, 'k-', label=r'$V(x)$', linewidth=2)
plt.title("Lowest Even-Parity Wavefunction, Density, and Potential")
plt.xlabel("x")
plt.ylabel("Amplitude / Energy")
plt.legend()
plt.grid(True)
plt.xlim(-b, b)
plt.ylim(-1.2*V0, 1.2*np.max(psi_even_vals))
plt.savefig("ps_2/even.png")

plt.figure(figsize=(10, 6))
plt.plot(x_vals, psi_odd_vals, 'b-', label=r'$\psi_{odd}(x)$')
plt.plot(x_vals, prob_odd, 'r--', label=r'$|\psi_{odd}(x)|^2$')
plt.plot(x_vals, V_vals, 'k-', label=r'$V(x)$', linewidth=2)
plt.title("Lowest Odd-Parity Wavefunction, Density, and Potential")
plt.xlabel("x")
plt.ylabel("Amplitude / Energy")
plt.legend()
plt.grid(True)
plt.xlim(-b, b)
plt.ylim(-1.2*V0, 1.2*np.max(np.abs(psi_odd_vals)))
plt.savefig("ps_2/odd.png")
