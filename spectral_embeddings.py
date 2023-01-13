
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

def spectral_embeddings(x, k, n=50, laplacian_norm=False):
    # TODO: maybe add number of neighbors as argument
    # create the similarity matrix
    S = NearestNeighbors(n_neighbors=n, n_jobs=-1).fit(x).kneighbors_graph(x)

    # ensure S is symetric
    S = 0.5*(S + S.T)

    # construct he laplacian
    L, d = laplacian(S, normed=laplacian_norm, return_diag=True)

    # find the 
    w, v = eigsh(L, k=k, sigma=0, which='LM')

    if laplacian_norm:
        v = (v.T / d).T
    
    return w, v