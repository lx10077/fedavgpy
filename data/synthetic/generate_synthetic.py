"""
This repository is modefied and referred from https://github.com/litian96/FedProx/tree/master/data/synthetic_iid
"""

import pickle
import numpy as np
import random
import os
cpath = os.path.dirname(__file__)


NUM_USER = 100
NUM_CLASS = 10
DIMENSION = 60
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex


def generate_synthetic(alpha, beta, iid=True):
    samples_per_user = np.random.lognormal(4, 2, NUM_USER).astype(int) + 50
    print('>>> Sample per user: {}'.format(samples_per_user.tolist()))

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    # prior for parameters
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, DIMENSION))

    diagonal = np.zeros(DIMENSION)
    for j in range(DIMENSION):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(DIMENSION) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, DIMENSION)

    W_global = b_global = None
    if iid == 1:
        W_global = np.random.normal(0, 1, (DIMENSION, NUM_CLASS))
        b_global = np.random.normal(0, 1,  NUM_CLASS)

    for i in range(NUM_USER):

        if iid == 1:
            assert W_global is not None and b_global is not None
            W = W_global
            b = b_global
        else:
            W = np.random.normal(mean_W[i], 1, (DIMENSION, NUM_CLASS))
            b = np.random.normal(mean_b[i], 1, NUM_CLASS)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i].extend(xx.tolist())
        y_split[i].extend(yy.tolist())

        # print("{}-th users has {} exampls".format(i, len(y_split[i])))
    return X_split, y_split


def generate_for_task(alpha, beta, iid):
    dataset_name = 'synthetic_alpha{}_beta{}_{}'.format(alpha, beta, 'iid' if iid else 'niid')
    print('\n')
    print('>>> Generate data for {}'.format(dataset_name))
    train_path = "{}/data/train/{}.pkl".format(cpath, dataset_name)
    test_path = "{}/data/test/{}.pkl".format(cpath, dataset_name)

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    X, y = generate_synthetic(alpha=alpha, beta=beta, iid=iid)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    
    for i in range(NUM_USER):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile, pickle.HIGHEST_PROTOCOL)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    task = {'niid(0,0)': (0, 0, 0),
            'niid(0.5,0.5)': (0.5, 0.5, 0),
            'niid(1, 1)': (1, 1, 0),
            'iid': (0, 0, 1)}

    for key, value in task.items():
        generate_for_task(*value)
