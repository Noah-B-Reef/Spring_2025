import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

f_exact = lambda x : np.log(factorial(x + 1)) - np.log(factorial(x))
f_approx = lambda x : np.log(x) 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# plot for n between 1 and 100
x = np.arange(1, 101)
y_exact = f_exact(x)
y_approx = f_approx(x)
ax1.plot(x, y_exact, label='Exact', color='blue')
ax1.plot(x, y_approx, label='Approx', color='red')
ax1.set_title('Exact vs Approximation for n between 1 and 100')
ax1.set_xlabel('n')
ax1.set_ylabel('f(n)')
ax1.legend()
ax1.grid()

# plot for n between 1 and 10
x = np.arange(1, 11)
y_exact = f_exact(x)
y_approx = f_approx(x)
ax2.plot(x, y_exact, label='Exact', color='blue')
ax2.plot(x, y_approx, label='Approx', color='red')
ax2.set_title('Exact vs Approximation for n between 1 and 10')
ax2.set_xlabel('n')
ax2.set_ylabel('f(n)')
ax2.legend()
ax2.grid()
plt.tight_layout()

plt.savefig('ps_5/hw5_q2_c.png')