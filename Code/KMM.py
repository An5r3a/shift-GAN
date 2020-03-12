import pandas as pd
import numpy as np
from sklearn.svm import SVC
import tensorflow as tf
from sklearn import svm
import scipy.stats as st
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from os.path import basename
from cvxopt import matrix, solvers




def iwe_kernel_mean_matching(X, Z):
        """
        Estimate importance weights based on kernel mean matching.
        Parameters
        ----------
        X : array
            source data (N samples by D features)
        Z : array
            target data (M samples by D features)
        Returns
        -------
        iw : array
            importance weights (N samples by 1)
        """
        kernel_type = 'rbf'
        bandwidth=1
        
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Compute sample pairwise distances
        KXX = cdist(X, X, metric='euclidean')
        KXZ = cdist(X, Z, metric='euclidean')

        # Check non-negative distances
        if not np.all(KXX >= 0):
            raise ValueError('Non-positive distance in source kernel.')
        if not np.all(KXZ >= 0):
            raise ValueError('Non-positive distance in source-target kernel.')

        # Compute kernels
        if kernel_type == 'rbf':
            # Radial basis functions
            KXX = np.exp(-KXX / (2*bandwidth**2))
            KXZ = np.exp(-KXZ / (2*bandwidth**2))

        # Collapse second kernel and normalize
        KXZ = N/M * np.sum(KXZ, axis=1)

        # Prepare for CVXOPT
        Q = matrix(KXX, tc='d')
        p = matrix(KXZ, tc='d')
        G = matrix(np.concatenate((np.ones((1, N)), -1*np.ones((1, N)),
                                   -1.*np.eye(N)), axis=0), tc='d')
        h = matrix(np.concatenate((np.array([N/np.sqrt(N) + N], ndmin=2),
                                   np.array([N/np.sqrt(N) - N], ndmin=2),
                                   np.zeros((N, 1))), axis=0), tc='d')

        # Call quadratic program solver
        sol = solvers.qp(Q, p, G, h)

        # Return optimal coefficients as importance weights
        return np.array(sol['x'])[:, 0]
        
        
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

def transform(Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new
        '''
        kernel_type ='primal'
        dim=30 
        lamb=1 
        gamma=1
        
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel('primal', X, None, gamma=gamma)
        n_eye = m if kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new
