from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture
import itertools
import scipy.stats
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from sklearn.random_projection import GaussianRandomProjection
from scipy.linalg import pinv



def RP_analysis(X):
	arr = []
	for i in range(1,50):
		rp = GaussianRandomProjection(n_components=i)
		X_rp = rp.fit(X)
		p = pinv(X_rp.components_)
		w = X_rp.components_
		reconstructed = ((p@w)@(X.T)).T 
		arr.append(mean_squared_error(X,reconstructed))

	arr = np.array(arr)
	plt.plot(np.arange(1,50),arr)
	plt.xlabel('Number of Components')
	plt.ylabel('Reconstruction Error')
	plt.grid()
	plt.show()

def K_Means_inertia_analysis(X):
	cluster_range = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
	inertia_arr = []
	for num_cluster in cluster_range:
		clusterer = KMeans(n_clusters=num_cluster, random_state=10)
		cluster_labels = clusterer.fit(X)
		inertia_arr.append(clusterer.inertia_)
	inertia_arr = np.array(inertia_arr)
	plt.plot([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],inertia_arr)
	plt.xlabel('Number of Clusters')
	plt.ylabel('Inertia')
	plt.title('Choosing Best k with Inertia')
	plt.show()

def kurtosis_analysis(X):
	arr = []
	for i in range(1,60):
		dim_red = FastICA(n_components = i).fit_transform(X)
		kurt = scipy.stats.kurtosis(dim_red)
		arr.append(np.mean(kurt))
	arr = np.array(arr)
	plt.plot(np.arange(1,60),arr)
	plt.xlabel('Number of Components')
	plt.ylabel('Kurtosis Value')
	plt.grid()
	plt.show()


def K_Means_silhouette_analysis(X,y):
	cluster_range = [3, 5,7,9,11,13,15]
	for num_cluster in cluster_range:
	    figure_to_show, (ax1, ax2) = plt.subplots(1, 2)
	    figure_to_show.set_size_inches(20, 8)
	   
	    ax1.set_xlim([-0.1, 1])
	    ax1.set_ylim([0, len(X) + (num_cluster + 1) * 10])

	    clusterer = KMeans(n_clusters=num_cluster, random_state=10)
	    cluster_labels = clusterer.fit_predict(X)

	    silhouette_avg = silhouette_score(X, cluster_labels)
	    print("For n_clusters = ", num_cluster,
	          "The average silhouette_score is :", silhouette_avg)
	    sample_silhouette_values = silhouette_samples(X, cluster_labels)

	    y_lower = 10
	    for i in range(num_cluster):
	        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

	        ith_cluster_silhouette_values.sort()

	        size_cluster_i = ith_cluster_silhouette_values.shape[0]
	        y_upper = y_lower + size_cluster_i

	        color = cm.nipy_spectral(float(i) / num_cluster)
	        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

	        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

	        y_lower = y_upper + 10 

	    ax1.set_title("The silhouette plot for the various clusters.")
	    ax1.set_xlabel("The silhouette coefficient values")
	    ax1.set_ylabel("Cluster label")

	    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

	    ax1.set_yticks([])  
	    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

	    colors = cm.nipy_spectral(cluster_labels.astype(float) / num_cluster)
	    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
	                c=colors, edgecolor='k')

	    centers = clusterer.cluster_centers_

	    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')

	    for i, c in enumerate(centers):
	        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

	    ax2.set_title("The visualization of the clustered data.")
	    ax2.set_xlabel("Feature space for the 1st feature")
	    ax2.set_ylabel("Feature space for the 2nd feature")

	    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data with num_cluster = %d" % num_cluster),fontsize=14, fontweight='bold')
	plt.show()

def K_Means_Results(best_n,X,y):
	kmeans = KMeans(n_clusters=best_n, random_state=3169)
	kmeans.fit(X)
	print('K-Means Inertia: ', kmeans.inertia_)
	silh_result = silhouette_score(X, kmeans.labels_)
	print('K-Means Silhouette score: ', silh_result)
	AMI = adjusted_mutual_info_score(y, kmeans.labels_)
	print('K-Means Adjusted Mutual Information (AMI) score: ', AMI)
	print()
	return kmeans.fit_transform(X)

def gmm_analysis(X,y):
	lowest_bic = np.infty
	bic = []
	n_components_range = range(1, 25)
	cv_types = ['spherical', 'tied', 'diag', 'full']
	for cv_type in cv_types:
	    for n_components in n_components_range:
	        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
	        gmm.fit(X)
	        bic.append(gmm.bic(X))
	        if bic[-1] < lowest_bic:
	            lowest_bic = bic[-1]
	            best_gmm = gmm

	bic = np.array(bic)
	color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
	clf = best_gmm
	bars = []
	plt.figure(figsize=(8, 6))
	spl = plt.subplot(2, 1, 1)

	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
	    xpos = np.array(n_components_range) + .2 * (i - 2)
	    bars.append(plt.bar(xpos, bic[i * len(n_components_range):(i + 1) * len(n_components_range)], width=.2, color=color))
	
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
	best_num = np.mod(bic.argmin(), len(n_components_range)) + 1
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)

	
	splot = plt.subplot(2, 1, 2)
	Y_ = clf.predict(X)
	
	for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,color_iter)):
	    v, w = linalg.eigh(cov)
	    if not np.any(Y_ == i):
	        continue
	    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

	    angle = np.arctan2(w[0][1], w[0][0])
	    angle = 180. * angle / np.pi 
	    v = 2. * np.sqrt(2.) * np.sqrt(v)
	    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
	    ell.set_clip_box(splot.bbox)
	    ell.set_alpha(.5)
	    splot.add_artist(ell)
	
	plt.xticks(())
	plt.yticks(())
	plt.title('Selected GMM: full model,' + str(best_num) + ' components')
	plt.subplots_adjust(hspace=.35, bottom=.02)
	plt.show()
	return best_num

def gmm_results(best_n,X,y):
	gmm_best = GaussianMixture(n_components=best_n, random_state=3169)
	gmm_best.fit(X)
	gmm_labels = gmm_best.predict(X)

	print('GMM BIC: ', gmm_best.bic(X))
	score_gmm = silhouette_score(X, gmm_labels)
	print('GMM Silhouette score: ', score_gmm)
	AMI_gmm = adjusted_mutual_info_score(y, gmm_labels)
	print('GMM Adjusted Mutual Information (AMI) score: ', AMI_gmm)
	print()
	return gmm_best.predict_proba(X)

