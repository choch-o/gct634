# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
#
# Juhan Nam
#

import sys
import os
import numpy as np
import librosa
from feature_summary import *

from sklearn.svm import SVC as svc
from sklearn.model_selection import RepeatedKFold

MFCC_DIM = 13
NUM_STATS = 3
FEATURE_DIM = 2 * 6 * MFCC_DIM + NUM_STATS * 5 + NUM_STATS * 7 + NUM_STATS * 12 * 3

def train_model(train_X, train_Y, valid_X, valid_Y, hyper_param1):

    # Choose a classifier (here, linear SVM)
    clf = svc(C=1.0, kernel='rbf', max_iter=1000)

    # train
    clf.fit(train_X, train_Y)

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/200.0*100.0
    print 'validation accuracy = ' + str(accuracy) + ' %'

    return clf, accuracy

def run_test(feature_path='./'):

    valid_test_acc_str = ''
    # load data
    train_X = combine_features('train', feature_path)
    valid_X = combine_features('valid', feature_path)
    test_X = combine_features('test', feature_path)

    # label generation
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 100)
    valid_Y = np.repeat(cls, 20)
    test_Y = np.repeat(cls, 20)

    # Repeated 6-fold
    kf = RepeatedKFold(n_splits=6, n_repeats=10, random_state=None)
    kfX = np.concatenate((train_X.T, valid_X.T))
    kfY = np.concatenate([train_Y, valid_Y])
    for train_index, valid_index in kf.split(kfX):
        train_X, valid_X = kfX[train_index], kfX[valid_index]
        train_Y, valid_Y = kfY[train_index], kfY[valid_index]

    train_X = train_X.T
    valid_X = valid_X.T

    # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)

    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)

    # training model
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    model = []
    valid_acc = []
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, a)
        model.append(clf)
        valid_acc.append(acc)
        valid_test_acc_str += str(acc) + ','

    # choose the model that achieve the best validation accuracy
    final_model = model[np.argmax(valid_acc)]

    # now, evaluate the model with the test set
    test_X = test_X.T
    test_X = test_X - train_X_mean
    test_X = test_X/(train_X_std + 1e-5)
    test_Y_hat = final_model.predict(test_X)

    accuracy = np.sum((test_Y_hat == test_Y))/200.0*100.0
    print 'test accuracy = ' + str(accuracy) + ' %'
    valid_test_acc_str += str(accuracy)
    return valid_test_acc_str

if __name__ == '__main__':

    # Run tests with different parameters
    n_fft_candidates = [512, 1024, 2048]
    hop_length_candidates = [128, 256, 512]
    n_mels_candidates = [40, 48, 64]
    dct_type_candidates = [2, 3]

    results_file = open('./test_results.csv', 'w')
    results_file.write("n_fft, hop_length, n_mels, dct_type, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5, val_acc6, test_accuracy\n")
    for n_fft in n_fft_candidates:
        for hop_length in hop_length_candidates:
            for n_mels in n_mels_candidates:
                for dct_type in dct_type_candidates:
                    feature_path = './' + str(n_fft) + '/' + str(hop_length) + '/' + str(n_mels) + '/' + str(dct_type) + '/'
                    results_file.write(str(n_fft) + ',' + str(hop_length) + ',' + str(n_mels) + ',' + str(dct_type) + ',' + run_test(feature_path) + '\n')
    results_file.close()
