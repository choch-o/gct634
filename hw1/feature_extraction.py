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
rms_path = 'rms/'
tonnetz_path = 'tonnetz/'

feature_paths = [mfcc_path, mfcc_cqt_path, zcr_path,
        s_centroid_path, s_rolloff_path, s_flatness_path,
        s_bandwidth_path, s_contrast_path,
        crm_stft_path, crm_cqt_path, crm_cens_path,
        rms_path, tonnetz_path]

MFCC_DIM = 13
SAMPLING_RATE = 22050

def extract_features(dataset='train', n_fft=512, hop_length=128, n_mels=40, dct_type=3):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print i

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        #print file_path
        y, sr = librosa.load(file_path, sr=SAMPLING_RATE)

        # mel-scaled spectrogram
        mel_S = librosa.feature.melspectrogram(y, sr=SAMPLING_RATE, n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, fmin=0.0, fmax=8000)

        #log compression
        log_mel_S = librosa.power_to_db(mel_S)

        # mfcc (DCT)
        mfcc = librosa.feature.mfcc(S=log_mel_S, dct_type=dct_type, n_mfcc=MFCC_DIM)
        mfcc = mfcc.astype(np.float32)    # to save the memory (64 to 32 bits)

        # constant-q transform
        C = np.abs(librosa.core.cqt(y, sr=sr))
        log_cqt = librosa.amplitude_to_db(C, ref=np.max)

        mfcc_cqt = librosa.feature.mfcc(S=log_cqt, dct_type=dct_type, n_mfcc=MFCC_DIM)
        mfcc_cqt = mfcc_cqt.astype(np.float32)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)

        # STFT
        S, phase = librosa.magphase(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))

        # Spectral centroid
        s_centroid = librosa.feature.spectral_centroid(S=S)

        # Spectral rolloff
        s_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)

        # Spectral flatness
        s_flatness = librosa.feature.spectral_flatness(S=S)

        # Spectral bandwidth
        s_bandwidth = librosa.feature.spectral_bandwidth(S=S)

        # Spectral contrast
        s_contrast = librosa.feature.spectral_contrast(S=S)

        # Chromagram
        crm_stft = librosa.feature.chroma_stft(S=S, sr=sr)
        crm_cqt = librosa.feature.chroma_cqt(C=C, sr=sr)
        crm_cens = librosa.feature.chroma_cens(C=C, sr=sr)

        rms = librosa.feature.rms(S=S)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        features = [mfcc, mfcc_cqt, zcr, s_centroid, s_rolloff, s_flatness,
                s_bandwidth, s_contrast, crm_stft, crm_cqt, crm_cens, rms, tonnetz]

        # save mfcc as a file
        file_name = file_name.replace('.wav','.npy')

        for feature in range(len(features)):
            save_file = './rms_tonnetz' + str(n_fft) + '/' + str(hop_length) + '/' + str(n_mels) + '/' + str(dct_type) + '/' + feature_paths[feature] + file_name
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, features[feature])
    f.close();

if __name__ == '__main__':
    n_fft_candidates = [512, 1024, 2048]
    hop_length_candidates = [128, 256, 512]
    n_mels_candidates = [40, 48, 64]
    dct_type_candidates = [2, 3]

    for f_index in range(len(n_fft_candidates)):
        for h_index in range(len(hop_length_candidates)):
            for m_index in range(len(n_mels_candidates)):
                for d_index in range(len(dct_type_candidates)):
                    extract_features(dataset='train',
                            n_fft=n_fft_candidates[f_index],
                            hop_length=hop_length_candidates[h_index],
                            n_mels=n_mels_candidates[m_index],
                            dct_type=dct_type_candidates[d_index])
                    extract_features(dataset='valid',
                            n_fft=n_fft_candidates[f_index],
                            hop_length=hop_length_candidates[h_index],
                            n_mels=n_mels_candidates[m_index],
                            dct_type=dct_type_candidates[d_index])
                    extract_features(dataset='test',
                            n_fft=n_fft_candidates[f_index],
                            hop_length=hop_length_candidates[h_index],
                            n_mels=n_mels_candidates[m_index],
                            dct_type=dct_type_candidates[d_index])


