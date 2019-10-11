# import common packages
import numpy as np
import pandas as pd
import time
import csv
import os
import matplotlib.pyplot as plt

# import implemented algorithms and evaluation tools
from algorithm import L2NMF, L1NMF, L1RegularizationNMF
from dataset import load_data
from noise import add_guassian_noise, add_laplace_noise, add_block_noise
from evaluation import rre_score, acc_score, nmi_score

def Load_dataset(dataset, reduce_factor):
    '''
    DESCRIPTION
        load dataset(ORL or YaleB) from local 'data' folder in specified image size.
    INPUT
        dataset
        reduce_factor
    OUTPUT
        X
        y
        img_size
    '''
    if dataset.lower()=='orl':
        X, y, img_size = load_data('data/ORL/', reduce=reduce_factor)
    
    elif dataset.lower()=='yaleb':
        X, y, img_size = load_data('data/CroppedYaleB/', reduce=reduce_factor)
        
    else:
        print('!!! No such dataset. Try again !!!')
        return
        
    print('>>Dataset loaded, X_ds: {}, y_ds: {}, img_size: {}'.format(X.shape, y.shape, img_size))
    return X, y, img_size



def Random_samples(X, y, rand_size=0.9):
    '''
    DESCRIPTION
        Randomly return given percentage of the entire dataset.
    INPUT
        X
        y
        rand_size
    OUTPUT
        X_trim
        y_trim
    '''
    arr = np.arange(X.shape[1])
    np.random.shuffle(arr)
    arr = arr[0:int(len(arr)*rand_size)]

    X_trim, y_trim = X[:, arr], y[arr]
    print('>>X: {}, y: {}'.format(X_trim.shape, y_trim.shape))
    return X_trim, y_trim


def Add_noise(X, noise, img_size):
    '''
    DESCRIPTION
        add given specified noise to dataset X.
    INPUT
        X
        noise
        img_size
    OUTPUT
        X_noisy
    '''
    if noise.lower()=='guassian':
        X_noisy = add_guassian_noise(X)
        
    elif noise.lower()=='laplace': 
        X_noisy = add_laplace_noise(X)
        
    elif noise.lower()=='block':
        X_noisy = add_block_noise(X, img_size)
    
    else:
        print('!!! No such noise type. Try again !!!')
        return
    print('>>{} noise added'.format(noise))
    return X_noisy


def Fit_nmf_model(X, model, n_comp, iters=1000, print_iter=None):
    '''
    DESCRIPTION
        train the given type of NMF algorithm and return the factorized result and training time.
    INPUT
        X
        model
        n_comp
        iters=
        print_iter=
    OUTPUT
        W
        H
        time
        E
    '''
    tic = time.time()
    E = None
    
    if model.lower()=='l2nmf':
        l2nmf = L2NMF(n_components=n_comp)
        W, H = l2nmf.fit(X, max_iter=iters, print_iter=print_iter)
        
    elif model.lower()=='l1nmf':
        l1nmf = L1NMF(n_components=n_comp)
        W, H = l1nmf.fit(X, max_iter=iters, print_iter=print_iter)

    elif model.lower()=='l1renmf':
        l1renmf = L1RegularizationNMF(n_components=n_comp)
        W, H, E = l1renmf.fit(X, max_iter=iters, print_iter=print_iter)
        
    else:
        print('!!! No such model type. Try again !!!')
        return
    
    toc = time.time()
    return W, H, toc-tic, E


def Reconstruct_images(W, H, E=None):
    '''
    DESCRIPTION
        Reconstruct the given images given dictionary and representation.
    INPUT
        W
        H
        E=
    OUTPUT
        X_re
    '''
    if E is None:
        E = 0
    X_re = np.dot(W, H) - E
        
    return X_re


def Visualize_results(X_org, X_noi, X_rec, img_size, idx=0):
    '''
    DESCRIPTION
        plot the original, noise, noisy and reconstructed images to compare. 
    INPUT
        X_org
        X_noi
        X_rec
        img_size
        idx
    '''
    plt.figure(figsize=(12,3))
    
    plt.subplot(141)
    plt.title('Original')
    plt.imshow(X_org[:,idx].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)
    
    plt.subplot(142)
    plt.title('Noise')
    noise = X_noi - X_org
    plt.imshow(noise[:,idx].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)

    plt.subplot(143)
    plt.title('Contaminated')
    plt.imshow(X_noi[:,idx].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)

    plt.subplot(144)
    plt.title('Reconstructed')
    plt.imshow(X_rec[:,idx].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)

    plt.show()
    return


def Save_images(X_org, X_noi, X_rec, img_size, file_name, path_name, n_image=10):
    '''
    DESCRIPTION
        save images to given path in given name.
    INPUT
        X_org
        X_noi
        X_rec
        img_size
        path_name
        file_name
        n_image=
    '''
    path_name = os.path.join('Images/', path_name)

    # Make a directory
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
        print('>>Directory created as {}'.format(path_name)) 

    for idx in range(n_image):
        plt.figure(figsize=(12,3))

        plt.subplot(141)
        plt.title('Original')
        plt.imshow(X_org[:,idx].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)
        
        plt.subplot(142)
        plt.title('Noise')
        noise = X_noi - X_org
        plt.imshow(noise[:,idx].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)

        plt.subplot(143)
        plt.title('Contaminated')
        plt.imshow(X_noi[:,idx].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)

        plt.subplot(144)
        plt.title('Reconstructed')
        plt.imshow(X_rec[:,idx].reshape(img_size[1],img_size[0]), cmap=plt.cm.gray)

        plt.savefig(path_name+'/'+file_name+'_'+str(idx) + '.png')
        plt.close()

    print('>>Images saved to {}/'.format(path_name))
    
    return

def Evaluate_performance(X_ori, X_rec, y, H):
    '''
    DESCRIPTION
        analyse the return the evaluation metrics given original and reconstructed image.
    INPUT
        X_org
        X_rec
        y
        H
    OUTPUT
        rre
        acc
        nmi
    '''
    rre = rre_score(X_ori, X_rec)
    acc = acc_score(y, H)
    nmi = nmi_score(y, H)
    print('>>RRE: {}, ACC:{}, NMI:{}'.format(rre, acc, nmi))
    
    return rre, acc, nmi

    
def Save_metrics(ls, file_name, path_name):
    '''
    DESCRIPTION
        save a list with metrics to given path.
        the format of ls should be [Algorithm, Noise, RRE, ACC, NMI, Time, max_iter].
    INPUT
        ls
        path_name
    '''
    # formant 
    path_name = os.path.join('metrics/' , path_name)
    file_name= file_name + '.csv'

    # Make directory
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
        print('>>Directory created as {}'.format(path_name))    
    
    # save to csv
    df = pd.DataFrame(ls)
    df.to_csv(os.path.join(path_name, file_name))
    
    return 