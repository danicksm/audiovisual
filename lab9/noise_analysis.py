#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal as signal
import soundfile as sf
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# 1. Load audio file
def load_audio(file_path):
    """Load audio file and return the signal and sample rate"""
    print("Loading audio file:", file_path)
    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Number of samples: {len(y)}")
    return y, sr

# 2. Generate and save spectrogram with Hann window
def generate_spectrogram(y, sr, window='hann', save_path='results/spectrogram.png'):
    """Generate spectrogram using Short-Time Fourier Transform with Hann window"""
    print("Generating spectrogram with Hann window...")
    
    plt.figure(figsize=(12, 8))
    D = librosa.stft(y, n_fft=2048, hop_length=512, window=window)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plot with logarithmic frequency scale
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (with logarithmic frequency scale)')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Spectrogram saved to {save_path}")
    return D, S_db

# 3. Estimate noise level and remove noise
def remove_noise(y, sr):
    """Estimate noise level and remove it using different methods"""
    print("Estimating noise level and applying noise reduction...")
    
    # Method 1: Simple noise reduction using low-pass filter
    fc = 2000  # cutoff frequency (Hz)
    b, a = signal.butter(5, fc/(sr/2), 'low')
    y_lowpass = signal.filtfilt(b, a, y)
    
    # Method 2: Noise reduction using Wiener filter
    y_wiener = signal.wiener(y, mysize=2049)
    
    # Method 3: Savitzky-Golay filter for smoothing
    y_savgol = signal.savgol_filter(y, 2049, 3)
    
    # Compute noise levels (standard deviation of difference)
    noise_level_original = np.std(y)
    noise_level_lowpass = np.std(y - y_lowpass)
    noise_level_wiener = np.std(y - y_wiener)
    noise_level_savgol = np.std(y - y_savgol)
    
    print(f"Estimated noise levels (standard deviation):")
    print(f"Original signal: {noise_level_original:.6f}")
    print(f"Noise removed by low-pass filter: {noise_level_lowpass:.6f}")
    print(f"Noise removed by Wiener filter: {noise_level_wiener:.6f}")
    print(f"Noise removed by Savitzky-Golay filter: {noise_level_savgol:.6f}")
    
    # Save processed audio files
    sf.write('results/lowpass_filtered.wav', y_lowpass, sr)
    sf.write('results/wiener_filtered.wav', y_wiener, sr)
    sf.write('results/savgol_filtered.wav', y_savgol, sr)
    
    return y_lowpass, y_wiener, y_savgol

# Compare spectrograms before and after noise reduction
def compare_spectrograms(y, y_lowpass, y_wiener, y_savgol, sr):
    """Compare spectrograms before and after noise reduction"""
    print("Comparing spectrograms before and after noise reduction...")
    
    plt.figure(figsize=(16, 16))
    
    # Original spectrogram
    plt.subplot(2, 2, 1)
    D_original = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db_original = librosa.amplitude_to_db(np.abs(D_original), ref=np.max)
    librosa.display.specshow(S_db_original, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original')
    
    # Low-pass filtered spectrogram
    plt.subplot(2, 2, 2)
    D_lowpass = librosa.stft(y_lowpass, n_fft=2048, hop_length=512, window='hann')
    S_db_lowpass = librosa.amplitude_to_db(np.abs(D_lowpass), ref=np.max)
    librosa.display.specshow(S_db_lowpass, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Low-pass filtered')
    
    # Wiener filtered spectrogram
    plt.subplot(2, 2, 3)
    D_wiener = librosa.stft(y_wiener, n_fft=2048, hop_length=512, window='hann')
    S_db_wiener = librosa.amplitude_to_db(np.abs(D_wiener), ref=np.max)
    librosa.display.specshow(S_db_wiener, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Wiener filtered')
    
    # Savitzky-Golay filtered spectrogram
    plt.subplot(2, 2, 4)
    D_savgol = librosa.stft(y_savgol, n_fft=2048, hop_length=512, window='hann')
    S_db_savgol = librosa.amplitude_to_db(np.abs(D_savgol), ref=np.max)
    librosa.display.specshow(S_db_savgol, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Savitzky-Golay filtered')
    
    plt.tight_layout()
    plt.savefig('results/spectrogram_comparison.png')
    print("Spectrograms comparison saved to results/spectrogram_comparison.png")

# 4. Find time moments with the highest energy
def find_high_energy_moments(y, sr, dt=0.1, df=50):
    """Find time moments with the highest energy in a neighborhood"""
    print("Finding moments with the highest energy...")
    
    # Calculate spectrogram
    n_fft = 2048
    hop_length = int(sr * dt)  # hop length for the desired time step
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    
    # Calculate frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Convert df from Hz to bin indices
    df_bins = int(df * n_fft / sr)
    
    # Calculate the energy (magnitude squared)
    # Energy = square of the magnitude of the STFT coefficients
    energy = np.abs(D)**2
    
    # Find time segments with the highest energy
    # Sum energy across frequency bands
    total_energy_per_time = np.sum(energy, axis=0)
    
    # Find peaks in energy
    # A peak is defined as a point where energy exceeds the mean energy by 1.5 times
    # The 1.5 multiplier is an empirical choice for distinguishing significant peaks
    peaks, properties = signal.find_peaks(total_energy_per_time, height=np.mean(total_energy_per_time)*1.5, distance=1)
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    peak_values = total_energy_per_time[peaks]
    
    # Sort peaks by energy
    sorted_indices = np.argsort(peak_values)[::-1]
    peak_times_sorted = peak_times[sorted_indices]
    peak_values_sorted = peak_values[sorted_indices]
    
    # Plot the energy and mark the peaks
    plt.figure(figsize=(12, 6))
    times = librosa.frames_to_time(np.arange(len(total_energy_per_time)), sr=sr, hop_length=hop_length)
    plt.plot(times, total_energy_per_time)
    plt.scatter(peak_times, peak_values, color='r', marker='x', label='Energy peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.title('Energy over time with peaks')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/energy_peaks.png')
    
    # Save all peaks data to file
    with open('results/all_energy_peaks.txt', 'w') as f:
        f.write("Time (s), Energy\n")
        for i in range(len(peak_times_sorted)):
            f.write(f"{peak_times_sorted[i]:.2f}, {peak_values_sorted[i]:.2f}\n")
    
    # Print the top high-energy moments
    print("\nTop 10 moments with highest energy:")
    for i in range(min(10, len(peak_times_sorted))):
        print(f"{i+1}. Time: {peak_times_sorted[i]:.2f}s, Energy: {peak_values_sorted[i]:.2f}")
    
    print(f"\nTotal number of energy peaks found: {len(peak_times_sorted)}")
    print("All peaks saved to results/all_energy_peaks.txt")
    
    return peak_times_sorted, peak_values_sorted

def main():
    # Specify the audio file path
    audio_file = 'fortepiannaya-petlya.wav'
    
    # 1. Load the audio file
    y, sr = load_audio(audio_file)
    
    # 2. Generate and save spectrogram
    D, S_db = generate_spectrogram(y, sr)
    
    # 3. Estimate noise level and apply noise reduction
    y_lowpass, y_wiener, y_savgol = remove_noise(y, sr)
    
    # Compare spectrograms
    compare_spectrograms(y, y_lowpass, y_wiener, y_savgol, sr)
    
    # 4. Find time moments with the highest energy
    peak_times, peak_values = find_high_energy_moments(y, sr)
    
    print("\nAnalysis complete! Results are saved in the 'results' directory.")

if __name__ == "__main__":
    main() 