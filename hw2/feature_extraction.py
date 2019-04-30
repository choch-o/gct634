'''
feature_extraction.py

A file related with extracting feature.
For the baseline code it loads audio files and extract mel-spectrogram using Librosa.
Then it stores in the './feature' folder.
'''
import os
import numpy as np
import librosa

from hparams import hparams

def load_list(list_name, hparams):
    with open(os.path.join(hparams.dataset_path, list_name)) as f:
        file_names = f.read().splitlines()

    return file_names

def melspectrogram(file_path, file_name, hparams):
    y, sr = librosa.load(os.path.join(file_path, file_name), hparams.sample_rate)
    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)

    mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.T

    return mel_S

def resize_array(array, length):
    resized_array = np.zeros((length, array.shape[1]))
    if array.shape[0] >= length:
        resized_array = array[:length]
    else:
        resized_array[:array.shape[0]] = array

    return resized_array

def main():
    print('Extracting Feature')
    list_names = ['train_list.txt', 'valid_list.txt', 'test_list.txt']
    augment_paths = [hparams.dataset_path, hparams.noise_dataset_path, hparams.shift_dataset_path, hparams.stretch_dataset_path]

    for list_name in list_names:
        set_name = list_name.replace('_list.txt', '')
        file_names = load_list(list_name, hparams)

        for file_name in file_names:
            feature = melspectrogram(hparams.dataset_path, file_name, hparams)
            if set_name == 'test':
                feature = feature[:feature.shape[0]-feature.shape[0]%hparams.feature_length]
                save_path = os.path.join(hparams.feature_path, set_name,
                        file_name.split('/')[0], file_name.split('/')[1].split('.')[1])
                save_name = 'test.npy'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.save(os.path.join(save_path, save_name), feature.astype(np.float32))
                print(os.path.join(save_path, save_name))
            elif set_name == 'train':
                for augment_path in augment_paths:
                    feature = feature[:feature.shape[0]-feature.shape[0]%hparams.feature_length]
                    for i in range(0, feature.shape[0], hparams.feature_length):
                        save_path = os.path.join(hparams.feature_path, set_name,
                                file_name.split('/')[0], file_name.split('/')[1].split('.')[1])
                        if len(augment_path.split('_')) > 1:
                            save_name = augment_path.split('_')[1] + str(int(i/hparams.feature_length)) + '.npy'
                        else:
                            save_name = str(int(i/hparams.feature_length)) + '.npy'
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        np.save(os.path.join(save_path, save_name), feature[i:i+hparams.feature_length].astype(np.float32))
                        print(os.path.join(save_path, save_name))
            else:
                feature = feature[:feature.shape[0]-feature.shape[0]%hparams.feature_length]
                for i in range(0, feature.shape[0], hparams.feature_length):
                    save_path = os.path.join(hparams.feature_path, set_name,
                            file_name.split('/')[0], file_name.split('/')[1].split('.')[1])
                    save_name = str(int(i/hparams.feature_length)) + '.npy'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    np.save(os.path.join(save_path, save_name), feature[i:i+hparams.feature_length].astype(np.float32))
                    print(os.path.join(save_path, save_name))
    print('finished')

if __name__ == '__main__':
    main()
