import numpy as np


def h(x):
    beta = np.random.uniform(0,1)
    omega = np.random.randn(x.shape[0],1)
    epsilon = 1
    w = 4*epsilon
    return np.floor((np.dot(x.T,omega)[0,0]/w) + beta)

def main():

    num_trials = int(1e6)
    num_collisions = 0
    dim = 3
    p1_count = 0
    p2_count = 0

    beta = np.random.uniform(0,1)
    omega = np.random.randn(x.shape[0],1)
    epsilon = 1
    w = 4*epsilon

    for i in range(num_trials):
        x = np.random.randn(dim,1)
        y = np.random.randn(dim,1)

        while np.linalg.norm(x-y) > 1:
            x = np.random.randn(3,1)
            y = np.random.randn(3,1)
        
        if h(x) == h(x):
            num_collisions += 1

        p1_count += 1
    
    print("Probability of collision for ||x-y||_2 <= 1: ", num_collisions/p1_count)
    
    num_collisions = 0

    for i in range(num_trials):
        x = np.random.randn(dim,1)
        y = np.random.randn(dim,1)

        while np.linalg.norm(x-y) < 2:
            x = np.random.randn(3,1)
            y = np.random.randn(3,1)
        
        if h(x) == h(x):
            num_collisions += 1
        
        p2_count += 1

    print("Probability of collision ||x-y|| >= 2: ", num_collisions/p2_count)

        

main()
