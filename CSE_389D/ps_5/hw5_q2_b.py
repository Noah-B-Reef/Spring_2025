import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

f_exact = lambda x : np.log(factorial(x + 1)) - np.log(factorial(x))
f_b_approx = lambda x : np.log(x) - (1-12*(x))/(24*(x)**2 + 2*(x)) 
f_c_approx = lambda x: np.log(x)

# plot for n between 1 and 10
x = np.arange(1, 11)
y_exact = f_exact(x)
y_b_approx = f_b_approx(x)
y_c_approx = f_c_approx(x)

plt.plot(x, y_exact, label='Exact', color='blue')
plt.plot(x, y_b_approx, label='Approx B', color='red')
plt.plot(x, y_c_approx, label='Approx C', color='green')
plt.title('Exact vs Approximation for n between 1 and 10')
plt.xlabel('n')
plt.ylabel('f(n)')
plt.legend()
plt.grid()
plt.savefig('ps_5/hw5_q2_n_10.png')
plt.clf()  # Clear the current figure


# plot for n between 1 and 100
x = np.arange(1, 101)
y_exact = f_exact(x)
y_b_approx = f_b_approx(x)
y_c_approx = f_c_approx(x)

plt.plot(x, y_exact, label='Exact', color='blue')
plt.plot(x, y_b_approx, label='Approx B', color='red')
plt.plot(x, y_c_approx, label='Approx C', color='green')
plt.title('Exact vs Approximation for n between 1 and 100')
plt.xlabel('n')
plt.ylabel('f(n)')
plt.legend()
plt.grid()
plt.savefig('ps_5/hw5_q2_n_100.png')
plt.clf()  # Clear the current figure

