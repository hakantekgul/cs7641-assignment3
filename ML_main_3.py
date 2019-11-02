import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, datasets
import matplotlib.pyplot as plt
import warnings
import os
from clustering import K_Means_silhouette_analysis, K_Means_Results, gmm_analysis, gmm_results, K_Means_inertia_analysis, kurtosis_analysis, RP_analysis
from classifiers import neural_net, tuned_neural_net
from dim_reductions import PCA, ICA, NMF, randomized_projections, Plot_2d, Plot_3d, PCA_Analysis, TSVD_Analysis
from sklearn.decomposition import PCA as PCA_sk
from sklearn.decomposition import FastICA, TruncatedSVD
from sklearn.decomposition import NMF as NMF_sk
from numpy import linalg as LA
import scipy
from sklearn.random_projection import GaussianRandomProjection
import sys
import time

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
os.system('clear')

print('##### STARING ASSIGNMENT 3 EXPERIMENTS #####')
print()

print('Loading data...')
# BREAST CANCER DATASET # 
'''
sklearn_data = datasets.load_breast_cancer()
x, y = sklearn_data.data, sklearn_data.target
'''
#######

# HEART DATASET #

train_heart = np.array(pd.read_csv('SPECTF.train'))
test_heart = np.array(pd.read_csv('SPECTF.test'))
train_labels = train_heart[:,0]
test_labels = test_heart[:,0]
x = np.vstack((train_heart[:,1:],test_heart[:,1:]))
y = np.hstack((train_labels,test_labels))

######

# Normalize the data
x = preprocessing.scale(x)
# Apply different clustering algorithms
print('#### APPLYING CLUSTERING ALGORITHMS TO THE DATASET ####')

print('KMeans Clustering: ')
print()
#K_Means_silhouette_analysis(x,y)
#K_Means_inertia_analysis(x)

K_Clustered_X = K_Means_Results(7,x,y)

print('GMM: ')
print()
print(x.shape)
#best_gmm_n = gmm_analysis(x,y)

GMM_Clustered_X = gmm_results(2,x,y)

# Apply different dim reduction algorithms
print('#### APPLYING DIM REDUCTION ALGORITHMS TO THE DATASET ###')
### PCA ###
# Run a variance analysis to find the best number of components
print('PCA: ')
print()
pca = PCA_sk()
pca.fit(x)
#PCA_Analysis(pca)
best_n = 6

# Visualize the data in 2d and 3d
Z_2d = PCA_sk(n_components = 2).fit_transform(x)
##Plot_2d(Z_2d,y)

Z_3d = PCA_sk(n_components = 3).fit_transform(x)
##Plot_3d(Z_3d,y)

best_X_pca = PCA_sk(n_components = best_n).fit_transform(x)

### ICA ###
print('ICA: ')
print()
#kurtosis_analysis(x)
Z_2d = FastICA(n_components = 2).fit_transform(x)
##Plot_2d(Z_2d,y)

Z_3d = FastICA(n_components = 3).fit_transform(x)
##Plot_3d(Z_3d,y)

best_X_ica = FastICA(n_components = best_n).fit_transform(x)


### Randomized Projections ###
print('Randomized Projections: ')
print()
#RP_analysis(x)
Z_2d = GaussianRandomProjection(n_components = 2).fit_transform(x)
#Plot_2d(Z_2d,y)

Z_3d = GaussianRandomProjection(n_components = 3).fit_transform(x)
#Plot_3d(Z_3d,y)

best_X_rp = GaussianRandomProjection(n_components = best_n).fit_transform(x)

print('Truncated SVD:')
print()
tsvd = TruncatedSVD(n_components=25)
tsvd.fit(x)
#TSVD_Analysis(tsvd)
best_n = 7

Z_2d = TruncatedSVD(n_components = 2).fit_transform(x)
#Plot_2d(Z_2d,y)

Z_3d = TruncatedSVD(n_components = 3).fit_transform(x)
#Plot_3d(Z_3d,y)

best_X_tsvd = TruncatedSVD(n_components = best_n).fit_transform(x)


# Re-do clustering experiments on dim reductioned data
print('#### RE-APPLYING CLUSTERING ALGORITHMS ON REDUCED DATA ####')
### PCA ###
print('Kmeans and GMM on PCA Data: ')
print()
#K_Means_silhouette_analysis(best_X_pca,y)
#K_Means_inertia_analysis(best_X_pca)

#K_Means_Results(7,best_X_pca,y)

#best_gmm_n = gmm_analysis(best_X_pca,y)

#gmm_results(4,best_X_pca,y)


### ICA ###
print('Kmeans and GMM on ICA Data: ')
print()
best_gmm_n = 3
#K_Means_silhouette_analysis(best_X_ica,y)
#K_Means_inertia_analysis(best_X_ica)


#K_Means_Results(5,best_X_ica,y)

#best_gmm_n = gmm_analysis(best_X_ica,y)

gmm_results(best_gmm_n,best_X_ica,y)

### Randomized Projections ###
print('Kmeans and GMM on Randomly Projected Data: ')
print()
#K_Means_silhouette_analysis(best_X_rp,y)
#K_Means_inertia_analysis(best_X_rp)


K_Means_Results(5,best_X_rp,y)

#best_gmm_n = gmm_analysis(best_X_rp,y)

gmm_results(2,best_X_rp,y)


# Run neural network with reduced data
print('### RUNNING NEURAL NETWORK WITH REDUCED DATA ###')
print('Neural Network with PCA Data: ')
print()
X_train, X_test, y_train, y_test = train_test_split(best_X_pca, y, test_size=0.4, random_state=3169)

st = time.time()
neural_net(X_train,y_train,X_test,y_test,learning_rate=0.1,plotting=False)
end = time.time()
print('PCA Time:' + str(end-st))

#best_nn = tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate=0.0001,plotting=False)

print('Neural Network with ICA Data: ')
print()
X_train, X_test, y_train, y_test = train_test_split(best_X_ica, y, test_size=0.4, random_state=3169)

neural_net(X_train,y_train,X_test,y_test,learning_rate=0.1,plotting=False)

#best_nn = tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate=0.0001,plotting=False)

print('Neural Network with Randomly Projected Data: ')
print()
X_train, X_test, y_train, y_test = train_test_split(best_X_rp, y, test_size=0.4, random_state=3169)

neural_net(X_train,y_train,X_test,y_test,learning_rate=0.1,plotting=False)

#best_nn = tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate=0.0001,plotting=False)

# Run neural network with clustered data 
print('### RUNNING NEURAL NETWORK WITH CLUSTERED DATA AS FEATURES')
#K_Means_silhouette_analysis(best_X_pca,y)

K_Clustered_X = K_Means_Results(5,x,y)

#best_gmm_n = gmm_analysis(best_X_ica,y)

GMM_Clustered_X = gmm_results(2,x,y)

print('K-Means Clustering & Neural Network: ')
X_train, X_test, y_train, y_test = train_test_split(K_Clustered_X, y, test_size=0.4, random_state=3169)

neural_net(X_train,y_train,X_test,y_test,learning_rate=0.1,plotting=False)

#best_nn = tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate=0.0001,plotting=False)
print()

print('GMM & Neural Network: ')
X_train, X_test, y_train, y_test = train_test_split(GMM_Clustered_X, y, test_size=0.4, random_state=3169)

neural_net(X_train,y_train,X_test,y_test,learning_rate=0.1,plotting=False)

#best_nn = tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate=0.0001,plotting=False)
print()


