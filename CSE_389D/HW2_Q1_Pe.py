import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

# define the secular equation
S = 10
gamma = 0.2

f = lambda x: np.sqrt(S**2 - x**2)/np.tanh(np.sqrt(S**2 - x**2)*gamma) - x * (np.tan(gamma*x)*np.tan(x) + 1)/(np.tan(gamma*x) - np.tan(x))

# plot the function
x = np.linspace(-5, 5, 1000)
y = f(x)

# find the roots
roots = []
test_points = np.arange(-6,6,1)
for i in test_points:
    try :
        root = fsolve(f, i)
        if root not in roots:
            roots.append(root)
    except:
        pass
print(roots)

# plot the roots
plt.plot(roots, [0]*len(roots), 'ro')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Secular Equation')

plt.plot(x, y)
plt.grid()
plt.show()
