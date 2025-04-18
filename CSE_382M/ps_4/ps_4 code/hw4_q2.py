import numpy as np
import pandas as pd

def nystrom_approx(X, sigma, r):
    K = kernel_matrix(X,sigma)
    A = K[0:r, 0:r]
    B = K[r:, 0:r]
    Z = np.block([[A],[B]])
    
    return Z @ np.linalg.pinv(A) @ Z.T
    
def fourier_approx(X, sigma, r):
    n, d = X.shape
    # Sample random frequencies from N(0, 1/sigma^2 I)
    W = np.random.normal(loc=0.0, scale=1.0/sigma, size=(d, r))
    # Sample random offsets from Uniform[0, 2pi]
    b = np.random.uniform(0, 2*np.pi, size=r)
    # Compute the feature mapping
    Phi = np.sqrt(2.0 / r) * np.cos(X @ W + b)
    return Phi @ Phi.T



def kernel_matrix(X, sigma=1):
    # Compute the squared norms of each row in X (shape: (n, 1))
    sum_X = np.sum(X ** 2, axis=1, keepdims=True)
    
    # Compute the pairwise squared Euclidean distances in a vectorized manner
    dists = sum_X + sum_X.T - 2 * X @ X.T
    
    # Compute the Gaussian kernel matrix
    K = np.exp(-dists / (2 * sigma ** 2))
    return K

# Create pandas DataFrame
df = pd.DataFrame(columns=["d", "n", "r", "Nystrom Error", "Fourier Error"])

# Test the Nystrom approximation and Fourier approximation
ds = [2, 4 , 8 , 16]
ns = [1024,4096,16384]
rs = [128,512,1024]

for d in ds:
    for n in ns:
        # Generate random data
        mu = 0
        sigma = 1
        shape = (n, d)
        X = np.random.normal(mu,sigma,shape)
        
        # Construct total kernel matrix
        K = kernel_matrix(X)
            
        for r in rs:
            
            # Construct Nystrom approximation
            K_nys = nystrom_approx(X, 1, r) 
           
            # Construct Fourier approximation
            K_fourier = fourier_approx(X, 1, r)

            # Compute the error
            nystrom_error = np.linalg.norm(K - K_nys) / np.linalg.norm(K)
            fourier_error = np.linalg.norm(K - K_fourier) / np.linalg.norm(K)
            
            
            # Append results to DataFrame
            df = pd.concat([df, pd.DataFrame({"d": [d], "n": [n], "r": [r], "Nystrom Error": [nystrom_error], "Fourier Error": [fourier_error]})], ignore_index=True)
            
# Save DataFrame to CSV
df.to_csv("kernel_approximation_errors.csv", index=False)

print(df.to_latex(
    index=False, 
    float_format="%.4f", 
    column_format="|c|c|c|c|c|",
    escape=False,
    caption="Kernel Approximation Errors",
    label="tab:kernel_approximation_errors"
))