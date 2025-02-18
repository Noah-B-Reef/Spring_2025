from numpy import *
from numpy.linalg import *
from numpy.random import randn, rand
from sklearn.cluster import KMeans



k = 8         # number of clusters
nk = 2000     # number of points per cluster 
d = 128      # dimensions. Try these vaulues (128, 64,16 2)
sep = 8.8     # separation factor. Try these values: 0.8,2.8,4.8,8.8


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

# ---- CLUSTERING
kmeans = KMeans(n_clusters=k).fit(X)
print(kmeans.cluster_centers_)
