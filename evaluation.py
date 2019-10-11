import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score

def assign_cluster_label(X, Y):
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0] # assign label.
    return Y_pred

def frobenius_norm(X):
    return np.linalg.norm(X, ord="fro")

def rre_score(Y_hat, Y_rec):
    return frobenius_norm(Y_hat - Y_rec) / frobenius_norm(Y_hat)

def acc_score(Y_hat, H):
    Y_pred = assign_cluster_label(H.T, Y_hat)
    return accuracy_score(Y_hat, Y_pred)

def nmi_score(Y_hat, H):
    Y_pred = assign_cluster_label(H.T, Y_hat)
    return normalized_mutual_info_score(Y_hat, Y_pred)