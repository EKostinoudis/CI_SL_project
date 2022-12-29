import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh

DTYPE = np.float32

def construct_kernel(kernel, x, y, degree=2, gamma=1, c=0):
    '''
    Calculate the gram-kernel matrix for x, y based on the given kernel name
    '''
    if kernel == 'linear':
        return x @ y.T
    elif kernel == 'poly':
        K = x @ y.T
        K *= gamma
        K += c
        return K ** degree
    elif kernel == 'rbf':
        K = euclidean_distances(x, y, squared=True)
        return np.exp(-gamma*K)
    elif kernel == 'sigmoid':
        K = x @ y.T
        K *= gamma
        K += c
        np.tanh(K, out=K)
        return K
    else:
        raise Exception(f'Kernel: {kernel} doesn\'t exist')

class KPCA_LDA():
    def __init__(self, theta=1, kernel='linear', gamma=1, degree=2, c=0):
        self.theta = theta
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.c = c

    @staticmethod
    def construct_between_within(x, y):
        classes, counts = np.unique(y, return_counts=True)
        class_means = np.asarray([np.mean(x[y==c,:], axis=0) for c in classes])
        all_mean = np.mean(x, axis=0)
        between = np.zeros((all_mean.shape[0], all_mean.shape[0]), dtype=DTYPE)
        for c_m, count in zip(class_means, counts):
            a = (c_m - all_mean)
            between += count*(a.reshape(-1,1) @ a.reshape(1,-1))
        between /= y.shape[0]

        within = np.zeros((all_mean.shape[0], all_mean.shape[0]), dtype=DTYPE)
        for c, c_m in zip(classes, class_means):
            # a = (x[y==c,:] - c_m).sum(axis=0)
            # within += (a.reshape(-1,1) @ a.reshape(1,-1))
            for xx in x[y==c,:]:
                a = (xx - c_m)
                within += (a.reshape(-1,1) @ a.reshape(1,-1))
        within /= y.shape[0]
        return between, within

    def fit(self, x, y, return_dist=True, return_z=False):
        # need to hold x for new data transformations
        self.x = x

        # calculate the gram matrix
        R = construct_kernel(self.kernel, self.x, self.x, self.degree, self.gamma, self.c)

        # centralize R
        # R = R - np.repeat(R.mean(axis=0).reshape(1,-1), R.shape[0], axis=0) - \
        #         np.repeat(R.mean(axis=1).reshape(-1,1), R.shape[1], axis=1) + R.mean()
        self.mean_rows = R.mean(axis=0) # needed for gram centralization
        self.mean_all = self.mean_rows.mean() # needed for gram centralization
        mean_cols = R.mean(axis=1).reshape(-1,1)
        R -= self.mean_rows
        R -= mean_cols
        R += self.mean_all

        # calculate the eigenvalues and vectors
        eigen_values, eigen_vectors = eigh(R)

        # hold only positive eigenvalues
        eigen_vectors = eigen_vectors[:, eigen_values>0]
        eigen_values = eigen_values[eigen_values>0]

        # sort based on eigenvalues
        idxs = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idxs]
        eigen_vectors = eigen_vectors[:, idxs]

        self.eigen_values = eigen_values
        self.eigen_vectors = eigen_vectors

        # same result as:
        # x = np.dot(R, eigen_vectors / np.sqrt(eigen_values))
        x = eigen_vectors * np.sqrt(eigen_values)

        # construct the between-class and within-class scatter matrices
        sb, sw = self.construct_between_within(x, y)

        # calculate the eigenvalues and vectors of within-class
        eigen_values_w, eigen_vectors_w = eigh(sw)
        
        # sort based on eigenvalues
        idxs = eigen_values_w.argsort()[::-1]
        eigen_values_w = eigen_values_w[idxs]
        eigen_vectors_w = eigen_vectors_w[:, idxs]

        # positive eigen-vectors (bigger than lambda_max/2000)
        self.p1 = eigen_vectors_w[:, eigen_values_w > (eigen_values_w[0]/2000)]

        sb_tilde = self.p1.T @ sb @ self.p1
        sw_tilde = self.p1.T @ sw @ self.p1

        # extract the regular discriminant features
        eigen_values_u, eigen_vectors_u = eigh(sb_tilde, sw_tilde)

        # sort based on eigenvalues
        idxs = eigen_values_u.argsort()[::-1]
        eigen_values_u = eigen_values_u[idxs]
        eigen_vectors_u = eigen_vectors_u[:, idxs]

        # number of classes
        c = np.unique(y).shape[0]
        
        self.u = eigen_vectors_u[:, :c-1]

        # non positive (smaller than lambda_max/2000)
        self.p2 = eigen_vectors_w[:, eigen_values_w <= (eigen_values_w[0]/2000)]
        sb_hat = self.p2.T @ sb @ self.p2

        eigen_values_v, eigen_vectors_v = eigh(sb_hat)

        # sort based on eigenvalues
        idxs = eigen_values_v.argsort()[::-1]
        eigen_values_v = eigen_values_v[idxs]
        eigen_vectors_v = eigen_vectors_v[:, idxs]

        self.v = eigen_vectors_v[:, :c-1]

        z1 = x @ self.p1 @ self.u
        z2 = x @ self.p2 @ self.v

        if return_dist is True:
            # calculate the distance matrix
            dist = euclidean_distances(z1, z1)
            dist /= dist.sum(axis=0)
            dist *= self.theta
            dist2 = euclidean_distances(z2, z2)
            dist2 /= dist.sum(axis=0)

        self.z1 = z1
        self.z2 = z2

        if return_dist is True and return_z is False:
            return dist + dist2
        elif return_dist is True and return_z is True:
            return dist + dist2, np.concatenate((z1, z2), axis=1)
        else:
            return np.concatenate((z1, z2), axis=1)

    def train_dist(self, theta=None):
        if theta is None:
            self.theta = theta
        dist = euclidean_distances(self.z1, self.z1)
        dist /= dist.sum(axis=0)
        dist *= self.theta
        dist2 = euclidean_distances(self.z2, self.z2)
        dist2 /= dist.sum(axis=0)
        return dist + dist2
    
    def transform(self, x, return_dist=True, return_z=False):
        # calculate the gram matrix
        R = construct_kernel(self.kernel, x, self.x, self.degree, self.gamma, self.c)

        # centralize R
        mean_cols = (R.sum(axis=1) / self.mean_rows.shape[0]).reshape(-1,1)
        R -= self.mean_rows
        R -= mean_cols
        R -= self.mean_all

        x = np.dot(R, self.eigen_vectors / np.sqrt(self.eigen_values))

        z1 = x @ self.p1 @ self.u
        z2 = x @ self.p2 @ self.v

        if return_dist is True:
            # calculate the distance matrix
            dist = euclidean_distances(z1, self.z1)
            dist /= dist.sum(axis=1).reshape(-1,1)
            dist *= self.theta
            dist2 = euclidean_distances(z2, self.z2)
            dist2 /= dist.sum(axis=1).reshape(-1,1)

        if return_dist is True and return_z is False:
            return dist + dist2
        elif return_dist is True and return_z is True:
            return dist + dist2, np.concatenate((z1, z2), axis=1)
        else:
            return np.concatenate((z1, z2), axis=1)