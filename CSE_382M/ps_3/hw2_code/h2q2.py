from numpy import *
from numpy.linalg import *
from numpy.random import randn, rand
from sklearn.cluster import KMeans
import pandas as pd


        # number of clusters
nk = 2000     # number of points per cluster 
ds = [128,64,16,8]      # dimensions. Try these vaulues (128, 64,16 2)
seps = [0.8,2.8,4.8,8.8]   # separation factor. Try these values: 0.8,2.8,4.8,8.8
df = pd.DataFrame(columns=['d','sep','err','err_k'])

for d in ds:
    for sep in seps:
        k = d//2
        # ---- GENERATE POINTS
        def getpoints(d, k, nk, sep):
            '''
            d: dimensions (or number of features)
            k: number of Gaussians
            nk: average number of points per Gaussian
            sep: separation between means:  sep*  d**(1/4) 
                
            Returns:
                X : 2d array  d by n,  n : total number of points
                mu: Gaussian means -- ground truth
                R:  distance of Gaussian means from origin  
            '''
            a = 2*pi/k   
            R = d**(1/4)/sqrt(2*(1-cos(a)))*sep #prediction of separatoin
            angles = arange(k)*a
            mu = zeros((d,k))
            mu[0,:] = cos(angles)*R
            mu[1,:] = sin(angles)*R
            nk = int64( nk*(rand(k)+0.5))
            X = zeros((d,sum(nk)))
            cnt = 0
            for j in range(k):
                m = mu[:,j]
                X[:,cnt:cnt+nk[j]] = randn(d,nk[j]) + m[:,newaxis]
                cnt += nk[j]
            return X,mu,R



        X,mu,R = getpoints(d,k,nk,sep)
        
        # remove NaN values from X and mu
        X = X[:,~isnan(X).any(axis=0)]
        mu = mu[:,~isnan(mu).any(axis=0)]

        # ---- CLUSTERING
        kmeans = KMeans(n_clusters=k).fit(X.T)
        print(kmeans.cluster_centers_.shape)
        print(mu.shape)

        def l2_err(X,Y):
            return sum(norm(X[:,i]-Y[:,i]) for i in range(X.shape[1]))

        # Error without best projection
        err = l2_err(mu,kmeans.cluster_centers_.T)
        
        if d != 2:
            # ---- FIND BEST PROJECTION

            # Use PCA to find best projection
            
            # SVD of X
            U,s,Vt = svd(X,full_matrices=False)

            # Truncate
            U = U[:,:k]
            s = s[:k]
            Vt = Vt[:k,:]

            # Project
            X_k = U @ diag(s) @ Vt

            # Error with best projection    
            kmeans_k = KMeans(n_clusters=k).fit(X_k.T)
            err_k = l2_err(mu,kmeans_k.cluster_centers_.T)
        
        
        # store result in dataframe
        df = pd.concat([df,pd.DataFrame([[d,sep,err,err_k]],columns=['d','sep','err','err_k'])],ignore_index=True)

print(df.to_latex(index=False))
