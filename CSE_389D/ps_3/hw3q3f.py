import matplotlib.pyplot as plt
import numpy as np

def V(x, k):
    return 0.25*(x**2) + 0.25*k*(x**4)

@np.vectorize
def E(x,n,k):
    return (n+0.5)

x = np.linspace(-4, 4, 100)
plt.xlabel(r'$\frac{x}{\sqrt{\frac{\hbar}{2m\omega}}}$')
plt.ylabel(r'$\frac{V}{\hbar\omega}$')
plt.plot(x, V(x, 0.2), label='k=0.2')
plt.plot(x, V(x, 0), label='k=0')
plt.plot(x,E(x,0,0), label='$n = 0$')
plt.plot(x,E(x,1,0),label='$n=1$')
plt.plot(x,E(x,2,0), label='n=2')
plt.legend()
plt.grid()
#plt.show()
plt.savefig('hw3q3f.png')
