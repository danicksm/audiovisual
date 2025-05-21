import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
from scipy.ndimage import gaussian_filter
import os

# === Parameters ===
filename = "guitar.wav"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# === 1. Loading audio file ===
print("Загрузка аудиофайла...")
rate, data = wav.read(filename)

# Convert to mono if stereo
if len(data.shape) > 1:
    print("Преобразование стерео в моно...")
    data = data[:, 0]

# Normalize
data = data.astype(np.float32)
data = data / np.max(np.abs(data))

# === 2. Spectrogram with Hann window ===
print("Построение спектрограммы с окном Ханна...")
window_duration_sec = 0.05  # 50ms window
window_size = int(rate * window_duration_sec)
overlap = window_size // 2  # 50% overlap

# Hann window
window = signal.windows.hann(window_size)

# Compute the STFT
f, t, Zxx = signal.stft(data, fs=rate, window=window, 
                       nperseg=window_size, noverlap=overlap)

# Convert to power spectrogram
spectrogram = np.abs(Zxx)**2

# Plot spectrogram with logarithmic frequency scale
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, 10 * np.log10(spectrogram + 1e-10), shading='gouraud')
plt.yscale('log')
plt.ylim([20, rate/2])  # From 20Hz to Nyquist frequency
plt.xlabel('Время [с]')
plt.ylabel('Частота [Гц]')
plt.title('Спектрограмма с окном Ханна')
plt.colorbar(label='Мощность [дБ]')
plt.tight_layout()
plt.savefig(f"{output_dir}/spectrogram_original.png")
plt.close()

# === 3. Noise reduction ===
print("Оценка и удаление шума...")

# Estimate noise profile from a quiet segment (first 0.5 seconds)
noise_segment = data[:int(rate * 0.5)]
noise_spectrum = np.abs(np.fft.rfft(noise_segment))**2
noise_profile = np.mean(noise_spectrum)

# Apply Wiener filter for noise reduction
# Compute STFT for filtering
f, t, Zxx = signal.stft(data, fs=rate, window=window, 
                       nperseg=window_size, noverlap=overlap)

# Estimate the SNR for each time-frequency bin
noise_power = noise_profile
signal_power = np.abs(Zxx)**2
snr = signal_power / (noise_power + 1e-10)

# Apply Wiener filter
gain = snr / (1 + snr)
Zxx_filtered = Zxx * gain

# Inverse STFT to recover the filtered signal
_, filtered_data = signal.istft(Zxx_filtered, fs=rate, window=window, 
                              nperseg=window_size, noverlap=overlap)

# Make sure the filtered signal has the same length as the original
if len(filtered_data) > len(data):
    filtered_data = filtered_data[:len(data)]
else:
    filtered_data = np.pad(filtered_data, (0, len(data) - len(filtered_data)))

# Compute spectrogram of the filtered signal
f, t, Zxx_filtered_spec = signal.stft(filtered_data, fs=rate, window=window, 
                                    nperseg=window_size, noverlap=overlap)
filtered_spectrogram = np.abs(Zxx_filtered_spec)**2

# Plot filtered spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, 10 * np.log10(filtered_spectrogram + 1e-10), shading='gouraud')
plt.yscale('log')
plt.ylim([20, rate/2])
plt.xlabel('Время [с]')
plt.ylabel('Частота [Гц]')
plt.title('Спектрограмма после фильтрации шума')
plt.colorbar(label='Мощность [дБ]')
plt.tight_layout()
plt.savefig(f"{output_dir}/spectrogram_filtered.png")
plt.close()

# Save audio files
wav.write(f"{output_dir}/filtered_audio.wav", rate, filtered_data.astype(np.float32))

# Plot both spectrograms side by side for comparison
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.pcolormesh(t, f, 10 * np.log10(spectrogram + 1e-10), shading='gouraud')
plt.yscale('log')
plt.ylim([20, rate/2])
plt.ylabel('Частота [Гц]')
plt.title('Исходная спектрограмма')
plt.colorbar(label='Мощность [дБ]')

plt.subplot(2, 1, 2)
plt.pcolormesh(t, f, 10 * np.log10(filtered_spectrogram + 1e-10), shading='gouraud')
plt.yscale('log')
plt.ylim([20, rate/2])
plt.xlabel('Время [с]')
plt.ylabel('Частота [Гц]')
plt.title('Спектрограмма после фильтрации шума')
plt.colorbar(label='Мощность [дБ]')

plt.tight_layout()
plt.savefig(f"{output_dir}/spectrogram_comparison.png")
plt.close()

# === 4. Find moments with highest energy in 40-50 Hz range ===
print("Анализ энергии в диапазоне 40-50 Гц...")
window_duration_sec = 0.1
window_size = int(rate * window_duration_sec)
step_size = window_size  # No overlap
frequencies = np.fft.rfftfreq(window_size, d=1/rate)

energy_by_window = []
time_stamps = []

for start in range(0, len(data) - window_size, step_size):
    end = start + window_size
    segment = data[start:end]
    
    # FFT and power spectrum
    spectrum = np.fft.rfft(segment)
    power_spectrum = np.abs(spectrum)**2
    
    # Energy in the 40-50 Hz range
    mask = (frequencies >= 40) & (frequencies <= 50)
    energy = np.sum(power_spectrum[mask])
    energy_by_window.append(energy)
    time_stamps.append(start/rate)

# Find peaks (moments with highest energy)
from scipy.signal import find_peaks
peaks, _ = find_peaks(energy_by_window, height=0.5*max(energy_by_window), distance=5)

# Visualization
plt.figure(figsize=(12, 6))

# Mark peaks
plt.plot([time_stamps[i] for i in peaks], 
         [energy_by_window[i] for i in peaks], 
         'ro', markersize=8, label='Пики энергии')

plt.xlabel('Время [с]')
plt.ylabel('Энергия')
plt.title('Анализ энергии в диапазоне 40-50 Гц')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/energy_40_50Hz.png")
plt.close()

# Print peak times
print("\nМоменты времени с наибольшей энергией в диапазоне 40-50 Гц:")
for i in peaks:
    print(f"  {time_stamps[i]:.2f} с (энергия: {energy_by_window[i]:.2e})")

print(f"\n✅ Результаты сохранены в директории: {output_dir}/")
