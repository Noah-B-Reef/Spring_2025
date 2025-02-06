import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.3,10,100)
y = -2/x + 1/(x**2)

plt.plot(x,y)
plt.xlabel('$r/r_0$')
plt.ylabel('$V(r)/E_0$')
plt.title('Plot of $V(r)/E_0$ vs $r/r_0$')
plt.grid()
plt.savefig('Q3_Pk.png')
