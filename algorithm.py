import numpy as np
import matplotlib.pyplot as plt 
from evaluation import rre_score

class L2NMF(object):
    def __init__(self, n_components, random_seed=2):
        self.n_components = n_components
        np.random.seed(random_seed)

    def initialize_WH(self, X):

        n_components = self.n_components
        n_samples, n_features = X.shape

        avg = np.sqrt(X.mean() / n_components)
        W = avg * np.abs(np.random.randn(n_samples, n_components))
        H = avg * np.abs(np.random.randn(n_components, n_features))

        return W, H

    def fit(self, X, max_iter=1000, print_iter=None):
        # -------------- Objective --------------------
        # || X - WH ||_F^2

        # -------------- Initialization --------------------
        W, H = self.initialize_WH(X)
        # -------------- Optimization --------------------
        error_list = []
        for iter in range(max_iter):

            # Update W
            W = W * (X.dot(H.T) / W.dot(H.dot(H.T)))

            # Update H
            H = H * (W.T.dot(X) / W.T.dot(W).dot(H))

            if not (print_iter is None) and (iter+1) % print_iter == 0:
                error = rre_score(X, W.dot(H))
                print("    iter = {}, error = {}".format(iter+1, error))
                error_list.append(error)

        # Plot the learning curve 
        if not (print_iter is None):
            plt.figure(figsize=(4,2))
            plt.title('#iterations vs. RRE')
            plt.plot(error_list)
            plt.show()

        return W, H
        

class L1NMF(object):
    def __init__(self, n_components, random_seed=2):
        self.n_components = n_components
        np.random.seed(random_seed)

    def initialize_DR(self, X):

        n_components = self.n_components
        n_samples, n_features = X.shape

        avg = np.sqrt(X.mean() / n_components)
        W = avg * np.abs(np.random.randn(n_samples, n_components))
        H = avg * np.abs(np.random.randn(n_components, n_features))

        return W, H

    def fit(self, X, max_iter=1000, print_iter=None):

        # -------------- Initialization --------------------
        W, H = self.initialize_DR(X)
    
        # -------------- Optimization --------------------
        error_list = []
        for iter in range(max_iter):
            e = 1e-5
            Q = ((X-np.dot(W, H))**2 + e**2)**(-1/2)
            W = W * ( (X*Q).dot(H.T) / (W.dot(H)*Q).dot(H.T) )
            H = H * ( (W.T.dot(X*Q)) / (W.T.dot(W.dot(H)*Q)) )

            if not (print_iter is None) and (iter + 1) % print_iter == 0:
                error = rre_score(X, W.dot(H))
                print("    iter = {}, error = {}".format(iter + 1, error))
                error_list.append(error)

        # Plot the learning curve 
        if not (print_iter is None):
            plt.figure(figsize=(4,2))
            plt.title('#iterations vs. RRE')
            plt.plot(error_list)
            plt.show()

        return W, H


class L1RegularizationNMF(object):

    def __init__(self, n_components, regularization_factor=0.05, random_seed=2):
        self.n_components = n_components
        self.regularization_factor = regularization_factor
        np.random.seed(random_seed)

    def initialize_UVE(self, X):

        n_components = self.n_components
        n_samples, n_features = X.shape

        avg = np.sqrt(X.mean() / n_components)
        U = avg * np.abs(np.random.randn(n_samples, n_components))
        V = avg * np.abs(np.random.randn(n_components, n_features))
        E = avg * np.abs(np.random.randn(n_samples, n_features))
        E = np.minimum(E, X)

        return U, V, E

    def fit(self, X, max_iter=1000, print_iter=None):

        # -------------- Initialization --------------------
        n_components = self.n_components
        n_samples, n_features = X.shape

        U, V, E = self.initialize_UVE(X)

        # -------------- Optimization --------------------
        error_list = []
        for iter in range(max_iter):

            # Update U
            X_hat = X - E

            U = U * (X_hat.dot(V.T)) / U.dot(V.dot(V.T))

            # Update V, E
            E_p = np.abs(E) + E / 2
            E_n = np.abs(E) - E / 2

            V_hat = np.concatenate([V, E_p, E_n], axis=0)
            X_hat = np.concatenate([X, np.zeros((1, n_features))], axis=0)
            U_hat = np.concatenate([U, np.eye(n_samples), -np.eye(n_samples)], axis=1)
            padding = np.concatenate([np.zeros((1, n_components)), np.sqrt(self.regularization_factor) * np.exp(1) * np.ones((1, 2 * n_samples))], axis=1)
            U_hat = np.concatenate([U_hat, padding], axis=0)
            SV_hat = np.abs(U_hat.T.dot(U_hat.dot(V_hat)))
            temp = ((U_hat.T.dot(U_hat.dot(V_hat))) - (U_hat.T.dot(X_hat))) / SV_hat
            V_hat = np.maximum(0, V_hat - V_hat * temp)
            V = V_hat[0:n_components,:]
            E_p = V_hat[n_components:n_components+n_samples,:]
            E_n = V_hat[n_components+n_samples:,:]
            E = E_p - E_n

            if not (print_iter is None) and (iter+1) % print_iter == 0:
                error = rre_score(X, U.dot(V) + E)
                print("    iter = {}, error = {}".format(iter+1, error))
                error_list.append(error)

        # Plot the learning curve 
        if not (print_iter is None):
            plt.figure(figsize=(4,2))
            plt.title('#iterations vs. RRE')
            plt.plot(error_list)
            plt.show()

        return U, V, E
