from numpy import *
from numpy.linalg import *
from numpy.random import randn
import builtins

def getAb(m,n):
    a = -1.0
    S = (arange(m)+1)**a
    G = randn(m,n); Q,_ = qr(G)
    Q = identity(m)[:m,:n]
    A = diag(S*S)@Q
    x = Q.T@diag(S)@randn(m); x /= norm(x)
    return A,A@x

def truncated_SVD(A,b,k):
    U,S,V = svd(A,full_matrices=False)
    U = U[:,:k]
    S = diag(S[:k])
    V = V[:k,:]
    A_m = U @ S @ V
    x_m = V.T @ inv(S) @ U.T @ b
    return A_m,x_m


def truncated_svd_solution(A, b, k):
    # Exact SVD
    U, s, Vt = svd(A, full_matrices=False)
    # Truncate to rank k
    U_k, s_k, Vt_k = U[:, :k], s[:k], Vt[:k, :]
    Sigma_k_inv = np.diag(1/s_k)
    x_k = Vt_k.T @ Sigma_k_inv @ U_k.T @ b
    A_k = U_k @ np.diag(s_k) @ Vt_k
    return A_k,x_k

def rand_truncated_SVD(A, b, k, p=10):
    m, n = A.shape
    Omega = randn(n, k+p)
    Y = A @ Omega
    Q,_ = qr(Y)
    B = Q.T @ A
    U_B, s_B, Vt = svd(B, full_matrices=False)
    U = Q @ U_B
    A_k = U @ diag(s_B) @ Vt
    x_k = Vt.T @ inv(diag(s_B)) @ U.T @ b
    return A_k, x_k

def rand_sketch_JLT(A, b, k,):
    m,n = A.shape
    # Create a Gaussian sketching matrix (with appropriate scaling)
    S = random.randn(n,m) / sqrt(k)
    A_k = S @ A
    b_sketch = S @ b
    x_k = lstsq(A_k, b_sketch)[0]  
    A_k = pinv(S) @ A_k
    return A_k,x_k

def main():
    m=1024
    n=512
    k=[32*j for j in range(1,5)]
    for rank in k:# Try k=32, 64, 128
        A,b = getAb(m,n)
        x = lstsq(A,b)[0]

        # Approximate A by A_m and get LS solution x_m by truncated SVD
        A_m,x_m = truncated_SVD(A,b,rank)
        
        # Compute residual error of truncated SVD
        trunc_svd_res_err = norm(b - A_m@x_m)/norm(b)

        # Compute relative solution error of truncated SVD
        trunc_svd_sol_err = norm(x_m - x)/norm(x)

        # Compute relative matrix error of truncated SVD
        trunc_svd_mat_err = norm(A - A_m)/norm(A)
        
        # true truncated SVD solution
        _,S,_ = svd(A,full_matrices=False)
        print(S[rank]/S[0])
        # Approximate A by A_m and get LS solution x_m by random truncated SVD
        A_m,x_m = rand_truncated_SVD(A,b,rank)

        # Compute residual error of random truncated SVD
        rand_trunc_svd_res_err = norm(b - A_m@x_m)/norm(b)

        # Compute relative solution error of random truncated SVD
        rand_trunc_svd_sol_err = norm(x_m - x)/norm(x)

        # Compute relative matrix error of random truncated SVD
        rand_trunc_svd_mat_err = norm(A - A_m)/norm(A)

        # Approximate A by A_m and get LS solution x_m by random sketching JLT
        A_m,x_m = rand_sketch_JLT(A,b,rank)
        # Compute residual error of random sketching JLT
        rand_sketching_JLT_res_err = norm(b - A.dot(x_m))/norm(b)

        # Compute relative solution error of random sketching JLT
        rand_sketching_JLT_sol_err = norm(x_m - x)/norm(x)

        # Compute relative matrix error of random sketching JLT
        rand_sketching_JLT_mat_err = norm(A - A_m)/norm(A)

        # Output results
        print(f'k : {rank}')
        print(f'Truncated SVD residual error : {trunc_svd_res_err}')
        print(f'Truncated SVD relative solution error : {trunc_svd_sol_err}')
        print(f'Truncated SVD relative matrix error : {trunc_svd_mat_err}')
        print(f'Random Truncated SVD residual error : {rand_trunc_svd_res_err}')
        print(f'Random Truncated SVD relative solution error : {rand_trunc_svd_sol_err}')
        print(f'Random Truncated SVD relative matrix error : {rand_trunc_svd_mat_err}')
        print(f'Random Sketching JLT residual error : {rand_sketching_JLT_res_err}')
        print(f'Random Sketching JLT relative solution error : {rand_sketching_JLT_sol_err}')
        print(f'Random Sketching JLT relative matrix error : {rand_sketching_JLT_mat_err}')
    

main()

