import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd

# Step 2
# Load audio file and play it
ipd.Audio('/Users/chocho/Desktop/lately.mp3', autoplay=True)



y, sr = librosa.load('/Users/chocho/Desktop/Mixdown.mp3')

# Step 3
'''
plt.plot(y)

plt.title('Waveform')
plt.show()
'''

# Short-time Fourier transform
D = librosa.stft(y)

# Obtain the magnitude spectrum
y_mag = np.abs(D)

'''
# Step 4
# Compress the magnitude spectrum in a log scale
y_mag_db = librosa.amplitude_to_db(y_mag, ref=np.max)

# Show the spectogram as an image
plt.imshow(y_mag_db, interpolation='bilinear', origin='lower', aspect='auto',
        cmap="viridis")
plt.title('Spectogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.grid('off')
plt.show()
'''

'''
# Step 5
# mel spectrogram
y_mag_mel = librosa.feature.melspectrogram(S=y_mag, sr=sr, n_mels=128)
y_mag_mel_db = librosa.amplitude_to_db(y_mag_mel, ref=np.max)

# constant-Q spectrogram
y_mag_cqt = librosa.cqt(y, n_bins=96, bins_per_octave=12)
y_mag_cqt_db = librosa.amplitude_to_db(y_mag_cqt, ref=np.max)

plt.figure(1)

# Plot Mel-Spectogram
plt.subplot(2,1,1)
plt.imshow(y_mag_mel_db, interpolation='bilinear', origin='lower', aspect='auto',
        cmap='viridis')
plt.title('Mel Spectogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.grid('off')

# Plot Constant-Q transform
plt.subplot(2,1,2)
plt.imshow(y_mag_cqt_db, interpolation='bilinear', origin='lower', aspect='auto',
        cmap='viridis')
plt.title('Constant-Q Transform')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.grid('off')

plt.show()
'''
