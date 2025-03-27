from scipy.optimize import fsolve
import numpy as np


S = 10
gamma = 0.2

f = lambda v: np.sqrt(S**2 - v**2)*np.tanh(np.sqrt(S**2 - v**2) * gamma) - v * (1 + np.tan(gamma * v) * np.tan(v))/(np.tan(gamma * v) - np.tan(v))

for i in range(2,100):
    try:
        v = fsolve(f, i)
        print(v)
    except:
        pass
