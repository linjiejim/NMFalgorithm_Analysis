import matplotlib.pyplot as plt
import numpy as np

def add_guassian_noise(X, loc=0, scale=0.1):
    # generate Guanssian noise
    guas_noise = np.random.normal(loc=0, scale=0.1, size=X.shape)

    # add noise to X
    X_noisy = X + guas_noise
    
    X_noisy[X_noisy>1] = 1
    X_noisy[X_noisy<0] = 0

    return X_noisy

def add_laplace_noise(X, loc=0, scale=0.1):
    # generate Laplace noise 
    lapl_noise = np.random.laplace(loc=0, scale=0.1, size=X.shape)

    # add noise to X
    X_noisy = X + lapl_noise

    X_noisy[X_noisy>1] = 1
    X_noisy[X_noisy<0] = 0

    return X_noisy

def add_block_noise(X, img_size):
    n_sample = X.shape[1]
    w,h = img_size
    b = h//8
    
    X_noisy = X.copy()
    
    # add noise to X
    for j in range(n_sample):
        rand_h = int(np.random.rand() * h)
        rand_w = int(np.random.rand() * w)

        pos_h = rand_h-b//2
        pos_w = rand_w-b//2
        
        for i in range(b):
            X_noisy[w*pos_h+pos_w:w*pos_h+pos_w+b, j] = 1e-6
            pos_h += 1
    
    return X_noisy