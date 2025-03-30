import matplotlib.pyplot as plt
import numpy as np

a = 1
b = 3
V_0 = 10

def V(x,a,b,V_0):
    if abs(x) < a:
        return 0
    elif abs(x) >= a and abs(x) <= b:
        return -V_0
    else:
        return np.inf

x = np.linspace(-10,10,1000)
V_x = [V(i,a,b,V_0) for i in x]

plt.plot(x,V_x)
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Potential Energy')
plt.show()

