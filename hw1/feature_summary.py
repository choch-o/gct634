# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
#
# Juhan Nam
#

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

data_path = './dataset/'
mfcc_path = 'mfcc/'
mfcc_cqt_path = 'mfcc_cqt/'
zcr_path = 'zcr/'
s_centroid_path = 's_centroid/'
s_rolloff_path = 's_rolloff/'
s_flatness_path = 's_flatness/'
s_bandwidth_path = 's_bandwidth/'
s_contrast_path = 's_contrast/'
crm_stft_path = 'crm_stft/'
crm_cqt_path = 'crm_cqt/'
crm_cens_path = 'crm_cens/'

MFCC_DIM = 13
NUM_STATS = 3
FEATURE_DIM = 2 * 6 * MFCC_DIM + NUM_STATS * 5 + NUM_STATS * 7 + NUM_STATS * 12 * 3

def combine_features(dataset='train', feature_path='./'):
    f = open(data_path + dataset + '_list.txt', 'r')
    if dataset == 'train':
        feature_mat = np.zeros(shape=(FEATURE_DIM, 1000))
    else:
        feature_mat = np.zeros(shape=(FEATURE_DIM, 200))

    i = 0
    for file_name in f:
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav', '.npy')

        mat_i = 0

        # load mfcc file
        mfcc_file = feature_path + mfcc_path + file_name
        mfcc = np.load(mfcc_file)

        # mfcc
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.mean(mfcc, axis=1)
        mat_i += MFCC_DIM
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.std(mfcc, axis=1)
        mat_i += MFCC_DIM
        # feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.var(mfcc, axis=1)
        # mat_i += MFCC_DIM

        # delta_mfcc
        delta = librosa.feature.delta(mfcc)
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.mean(delta, axis=1)
        mat_i += MFCC_DIM
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.std(delta, axis=1)
        mat_i += MFCC_DIM
        # feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.var(delta, axis=1)
        # mat_i += MFCC_DIM

        # double_delta_mfcc
        double_delta = librosa.feature.delta(mfcc, order=2)

        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.mean(double_delta, axis=1)
        mat_i += MFCC_DIM
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.std(double_delta, axis=1)
        mat_i += MFCC_DIM
        # feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.var(double_delta, axis=1)
        # mat_i += MFCC_DIM

        # load mfcc_cqt file
        mfcc_cqt_file = feature_path + mfcc_cqt_path + file_name
        mfcc_cqt = np.load(mfcc_cqt_file)

        # mfcc_cqt
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.mean(mfcc_cqt, axis=1)
        mat_i += MFCC_DIM
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.std(mfcc_cqt, axis=1)
        mat_i += MFCC_DIM
        # feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.var(mfcc_cqt, axis=1)
        # mat_i += MFCC_DIM

        # delta_mfcc_cqt
        delta = librosa.feature.delta(mfcc_cqt)
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.mean(delta, axis=1)
        mat_i += MFCC_DIM
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.std(delta, axis=1)
        mat_i += MFCC_DIM
        # feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.var(delta, axis=1)
        # mat_i += MFCC_DIM

        # double_delta_mfcc
        double_delta = librosa.feature.delta(mfcc_cqt, order=2)

        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.mean(double_delta, axis=1)
        mat_i += MFCC_DIM
        feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.std(double_delta, axis=1)
        mat_i += MFCC_DIM
        # feature_mat[mat_i:mat_i+MFCC_DIM, i] = np.var(double_delta, axis=1)
        # mat_i += MFCC_DIM

        index = mat_i
        # load zcr file
        zcr_file = feature_path + zcr_path + file_name
        zcr = np.load(zcr_file)
        feature_mat[index:++index, i] = np.mean(zcr, axis=1)
        feature_mat[index:++index, i] = np.std(zcr, axis=1)
        feature_mat[index:++index, i] = np.var(zcr, axis=1)

        # load spectral features
        sc_file = feature_path + s_centroid_path + file_name
        sc = np.load(sc_file)
        feature_mat[index:++index, i] = np.mean(sc, axis=1)
        feature_mat[index:++index, i] = np.std(sc, axis=1)
        feature_mat[index:++index, i] = np.var(sc, axis=1)

        sr_file = feature_path + s_rolloff_path + file_name
        sr = np.load(sr_file)
        feature_mat[index:++index, i] = np.mean(sr, axis=1)
        feature_mat[index:++index, i] = np.std(sr, axis=1)
        feature_mat[index:++index, i] = np.var(sc, axis=1)

        sf_file = feature_path + s_flatness_path + file_name
        sf = np.load(sf_file)
        feature_mat[index:++index, i] = np.mean(sf, axis=1)
        feature_mat[index:++index, i] = np.std(sf, axis=1)
        feature_mat[index:++index, i] = np.var(sc, axis=1)

        sb_file = feature_path + s_bandwidth_path + file_name
        sb = np.load(sb_file)
        feature_mat[index:++index, i] = np.mean(sb, axis=1)
        feature_mat[index:++index, i] = np.std(sb, axis=1)
        feature_mat[index:++index, i] = np.var(sc, axis=1)

        scr_file = feature_path + s_contrast_path + file_name
        scr = np.load(scr_file)
        feature_mat[index:index+7, i] = np.mean(scr, axis=1)
        index += 7
        feature_mat[index:index+7, i] = np.std(scr, axis=1)
        index += 7
        feature_mat[index:index+7, i] = np.var(scr, axis=1)
        index += 7

        # load chroma features
        crm_stft_file = feature_path + crm_stft_path + file_name
        crm_stft = np.load(crm_stft_file)
        feature_mat[index:index+12, i] = np.mean(crm_stft, axis=1)
        index += 12
        feature_mat[index:index+12, i] = np.std(crm_stft, axis=1)
        index += 12
        feature_mat[index:index+12, i] = np.var(crm_stft, axis=1)
        index += 12

        crm_cqt_file = feature_path + crm_cqt_path + file_name
        crm_cqt = np.load(crm_cqt_file)
        feature_mat[index:index+12, i] = np.mean(crm_cqt, axis=1)
        index += 12
        feature_mat[index:index+12, i] = np.std(crm_cqt, axis=1)
        index += 12
        feature_mat[index:index+12, i] = np.var(crm_cqt, axis=1)
        index += 12

        crm_cens_file = feature_path + crm_cens_path + file_name
        crm_cens = np.load(crm_cens_file)
        feature_mat[index:index+12, i] = np.mean(crm_cens, axis=1)
        index += 12
        feature_mat[index:index+12, i] = np.std(crm_cens, axis=1)
        index += 12
        feature_mat[index:index+12, i] = np.var(crm_cens, axis=1)
        index += 12

        i = i + 1
    f.close()
    return feature_mat


def mean_mfcc(dataset='train'):

    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        mfcc_mat = np.zeros(shape=(MFCC_DIM, 1000))
    else:
        mfcc_mat = np.zeros(shape=(MFCC_DIM, 200))

    i = 0
    for file_name in f:

        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)

        # mean pooling
        temp = np.mean(mfcc, axis=1)
        mfcc_mat[:,i]= np.mean(mfcc, axis=1)
        i = i + 1

    f.close();

    return mfcc_mat

if __name__ == '__main__':
    train_data = mean_mfcc('train')
    valid_data = mean_mfcc('valid')
    test_data = mean_mfcc('test')

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3,1,3)
    plt.imshow(test_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.show()








