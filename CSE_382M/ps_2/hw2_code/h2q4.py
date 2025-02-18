from numpy import *
from numpy.linalg import *
from numpy.random import randn


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

def rand_sketch_JLT(A, b, k, m_sketch=200):
    m, n = A.shape
    # Construct a Gaussian JL sketching matrix S of size m_sketch x m.
    S = randn(m_sketch, m) / sqrt(m_sketch)
    
    # Compute the sketched matrix.
    SA = S @ A  # size: (m_sketch x n)
    
    # Compute the SVD of the sketched matrix.
    U_y, s_y, Vt_y = svd(SA, full_matrices=False)
    
    # Extract the top k right singular vectors.
    V_y_k = Vt_y[:k, :].T  # shape: (n x k)
    
    # Compute the rank-k approximation of A via projection:
    A_k = A @ (V_y_k @ V_y_k.T)
    
    # Solve the sketched least-squares problem:
    # Instead of solving the full LS problem, we solve S A x ≈ S b.
    # (The sketching preserves distances approximately.)
    x_k, residuals, rank, s = lstsq(S @ A, S @ b, rcond=None)
    
    return x_k, A_k 
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
        rand_sketching_JLT_res_err = norm(b - A_m@x_m)/norm(b)

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

