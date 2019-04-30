import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

from hparams import hparams

class AudioAugmentation:
    def read_audio_file(self, file_path, sr):
        input_length = 661794
        data = librosa.load(file_path, sr)[0]
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def add_noise(self, data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

    def shift(self, data):
        return np.roll(data, 1600)

    def stretch(self, data, rate=1):
        input_length = 661794
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def write_audio_file(self, file, data, sample_rate=22050):
        librosa.output.write_wav(file, data, sample_rate)

    def plot_time_series(self, data):
        fig = plt.figure(figsize=(14,8))
        plt.title('Raw wave')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()

def load_list(list_name, hparams):
    with open(os.path.join(hparams.dataset_path, list_name)) as f:
        file_names = f.read().splitlines()

    return file_names

def main():
    aa = AudioAugmentation()

    list_names = ['train_list.txt', 'valid_list.txt']

    plot_index = 0
    for list_name in list_names:
        set_name = list_name.replace('_list.txt', '')
        file_names = load_list(list_name, hparams)

        for file_name in file_names:
            data = aa.read_audio_file(os.path.join(hparams.dataset_path, file_name), hparams.sample_rate)

            # Augment with noise
            data_noise = aa.add_noise(data)
            save_path = os.path.join(hparams.noise_dataset_path, file_name.split('/')[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            aa.write_audio_file(os.path.join(save_path, file_name.split('/')[1]), data_noise, hparams.sample_rate)

            # Augment with shift
            data_shift = aa.shift(data)
            save_path = os.path.join(hparams.shift_dataset_path, file_name.split('/')[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            aa.write_audio_file(os.path.join(save_path, file_name.split('/')[1]), data_shift, hparams.sample_rate)

            # Augment with stretch

            data_stretch = aa.stretch(data)
            save_path = os.path.join(hparams.stretch_dataset_path, file_name.split('/')[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            aa.write_audio_file(os.path.join(save_path, file_name.split('/')[1]), data_stretch, hparams.sample_rate)

if __name__ == '__main__':
    main()

