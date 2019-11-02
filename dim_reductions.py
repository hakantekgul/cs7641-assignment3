import numpy as np
import scipy 
import sklearn.random_projection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA



# COV = (1/K)(X - X_mean).(X - X_mean)^T
def PCA(X, n):
    # getting 0 mean
    X_ = X - np.mean(X, axis = 1, keepdims = True)
    cov_X = (X_ @ X_.T) / (X.shape[1] - 1)
    evals, evecs = scipy.sparse.linalg.eigsh(cov_X, n)
    norm = LA.norm(evecs)
    evals = np.sqrt(evals)
    evals = 1 / evals
    evals = np.diag(evals)
    W = evals.dot(evecs.T)
    Z = W.dot(X_)    
    return W, Z, norm

def ICA(X):
    I_2 = np.identity(X.shape[0])
    W = np.identity(X.shape[0]) 
    const = 0.0001
    N = X.shape[1]
    for i in range(500):
        Y = W.dot(X)
        delta_w = (N*I_2-2*np.tanh(Y).dot(Y.T)).dot(W)
        W += const*delta_w
    return Y,W

def randomized_projections(X,n):
    projector = random_projection.GaussianRandomProjection(n_components=n)
    X_new = projector.fit_transform(X)
    return X_new

def NMF(X,n): 
    Dim1 = X.shape[0]
    Dim2 = X.shape[1]
    H = np.random.rand(n,Dim2)
    W = np.random.rand(Dim1,n) # so that X = W.H
    for i in range(100): 
        H = H * ((W.T.dot(X)) / (0.000001 + W.T.dot(W).dot(H))) # prevent division by 0
        W = W * ((X.dot(H.T)) / (0.000001 + W.dot(H).dot(H.T)))
    return W, W @ H

def Plot_2d(Z,Y): 
    for i in range(len(Y)):
        if Y[i] == 0:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'r')
        elif Y[i] == 1:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'b')
        elif Y[i] == 2:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'g')
        elif Y[i] == 3:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'k')
        elif Y[i] == 4:
            plt.scatter(Z[i, 1], Z[i, 0], color = 'c')
    plt.show()

def Plot_3d(Z,Y):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(Y)):
        if Y[i] == 0:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = 'b', marker='o')
        elif Y[i] == 1:
            ax.scatter(Z[i, :][0], Z[i, :][1], Z[i, :][2], c = 'r', marker='o')
    plt.show()

def PCA_Analysis(pca):
    plt.figure()
    plt.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), pca.explained_variance_ratio_)
    plt.xticks(np.arange(1, pca.explained_variance_ratio_.size + 1))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance')
    plt.title('Variance vs. Number of Components')
    plt.grid()
    plt.show()

def TSVD_Analysis(tsvd):
    plt.figure()
    plt.plot(np.arange(1, tsvd.explained_variance_ratio_.size + 1), tsvd.explained_variance_ratio_)
    plt.xticks(np.arange(1, tsvd.explained_variance_ratio_.size + 1))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance')
    plt.title('Variance vs. Number of Components')
    plt.grid()
    plt.show()


