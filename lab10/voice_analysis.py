#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal as signal
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# List of audio files to analyze
audio_files = ['aaa.wav', 'iii.wav', 'gaf.wav']
labels = ["'А'", "'И'", "Имитация животного"]

# Function to analyze voice files
def analyze_voice(file_path, label, file_number):
    print(f"Processing {file_path}...")
    
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Get duration and sample rate
    duration = librosa.get_duration(y=y, sr=sr)
    n_samples = len(y)
    
    print(f"File: {file_path}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Number of samples: {n_samples}")
    
    # 2. Generate spectrogram using Hann window
    n_fft = 2048
    hop_length = 512
    
    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    
    # Convert to magnitude spectrogram
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plot spectrogram with logarithmic frequency scale
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', 
                            y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Спектрограмма звука {label}')
    plt.tight_layout()
    plt.savefig(f'results/spectrogram_{file_number}.png')
    plt.close()
    
    # 3. Find minimum and maximum voice frequency
    # Use only portions where there is significant energy
    # Filter out silence and background noise
    
    # Calculate the magnitude spectrogram
    S = np.abs(D)
    
    # Threshold to filter out noise (adjust as needed)
    threshold = np.mean(S) * 2  # Use twice the mean as threshold
    
    # Create a mask for values above the threshold
    mask = S > threshold
    
    # Find frequency bins that have significant energy somewhere in time
    freq_with_energy = np.any(mask, axis=1)
    
    # Get frequencies
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Find min and max frequencies that have significant energy
    significant_freqs = frequencies[freq_with_energy]
    if len(significant_freqs) > 0:
        min_freq = significant_freqs[0]
        max_freq = significant_freqs[-1]
    else:
        min_freq = 0
        max_freq = 0
    
    print(f"Minimum voice frequency: {min_freq:.2f} Hz")
    print(f"Maximum voice frequency: {max_freq:.2f} Hz")
    
    # 4. Find the most timbrally colored fundamental tone (with the most overtones)
    
    # Use harmonic-percussive source separation to enhance harmonic content
    y_harmonic, _ = librosa.effects.hpss(y)
    
    # Calculate pitch (f0) and harmonics using Spectral Harmonic-to-Noise Ratio
    pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # For each time frame, count overtones above significant threshold
    overtone_counts = []
    fundamental_tones = []
    
    # For each time frame
    for t in range(pitches.shape[1]):
        # Get the pitch with maximum magnitude in this frame
        idx = np.argmax(magnitudes[:, t])
        if magnitudes[idx, t] > 0:  # If we have a pitch
            f0 = pitches[idx, t]
            
            # Count frames with significant harmonic content
            harmonic_count = 0
            
            # Check for overtones (integer multiples of f0)
            for n in range(2, 10):  # Check up to 9th harmonic
                target_freq = f0 * n
                
                # Find the closest frequency bin
                closest_idx = np.argmin(np.abs(frequencies - target_freq))
                
                # Check if there's significant energy at this harmonic
                if closest_idx < len(frequencies) and S[closest_idx, t] > threshold:
                    harmonic_count += 1
            
            overtone_counts.append(harmonic_count)
            fundamental_tones.append(f0)
    
    # Find the fundamental tone with the most overtones
    if overtone_counts:
        max_overtones_idx = np.argmax(overtone_counts)
        best_f0 = fundamental_tones[max_overtones_idx]
        overtone_count = overtone_counts[max_overtones_idx]
        print(f"Most timbrally colored fundamental tone: {best_f0:.2f} Hz (with {overtone_count} significant overtones)")
    else:
        best_f0 = 0
        print("No significant fundamental tones detected")
    
    # 5. Find the three strongest formants
    # Use time step Δt = 0.1s and frequency step Δf = 40-50 Hz
    
    # Time step of 0.1 seconds
    time_step = 0.1  # seconds
    frame_step = int(time_step * sr / hop_length)
    
    # Collect formants for all frames
    all_formants = []
    
    # For each time frame with step of 0.1s
    for t in range(0, S.shape[1], frame_step):
        if t >= S.shape[1]:
            break
            
        # Get the spectrum at this time
        spectrum = S[:, t]
        
        # Find peaks (formants)
        # Use a relatively small distance to get finer frequency resolution
        # distance≈50Hz in frequency bins
        min_distance = int(45 / (sr / n_fft))  # ~45 Hz in bin indices
        peaks, _ = signal.find_peaks(spectrum, distance=min_distance)
        
        # Get the frequencies and amplitudes of the peaks
        peak_freqs = frequencies[peaks]
        peak_amps = spectrum[peaks]
        
        # Keep only top 5 peaks by amplitude (to have some buffer for the top 3)
        if len(peak_freqs) > 5:
            top_indices = np.argsort(peak_amps)[-5:]
            peak_freqs = peak_freqs[top_indices]
            peak_amps = peak_amps[top_indices]
        
        # Sort by frequency
        sort_idx = np.argsort(peak_freqs)
        peak_freqs = peak_freqs[sort_idx]
        peak_amps = peak_amps[sort_idx]
        
        # Store time and formant data
        time_sec = t * hop_length / sr
        all_formants.append((time_sec, peak_freqs, peak_amps))
    
    # Create a plot showing formants over time
    plt.figure(figsize=(12, 6))
    
    # Plot the spectrogram as background
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', 
                            y_axis='log', cmap='gray_r', alpha=0.7)
    
    # Define colors for the three strongest formants
    colors = ['red', 'green', 'blue']
    
    # Track top 3 formants by frequency
    formant_tracks = [[] for _ in range(3)]
    
    # For each time frame where we computed formants
    for time_sec, peak_freqs, peak_amps in all_formants:
        # If we have at least 3 formants in this frame
        if len(peak_freqs) >= 3:
            # Sort by amplitude (descending)
            sorted_indices = np.argsort(peak_amps)[::-1]
            
            # Get top 3 peaks by amplitude
            for i in range(min(3, len(sorted_indices))):
                idx = sorted_indices[i]
                freq = peak_freqs[idx]
                amp = peak_amps[idx]
                
                # Store data for this formant track
                formant_tracks[i].append((time_sec, freq, amp))
    
    # Plot each formant track with a different color
    for i, track in enumerate(formant_tracks):
        if track:  # If we have data for this formant track
            times, freqs, _ = zip(*track)
            plt.plot(times, freqs, 'o-', color=colors[i], markersize=4, 
                    label=f'Формант {i+1}')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Форманты звука {label}')
    plt.xlabel('Время (с)')
    plt.ylabel('Частота (Гц)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/formants_{file_number}.png')
    plt.close()
    
    # Save formant data to file
    with open(f'results/formants_{file_number}.txt', 'w') as f:
        f.write(f"Форманты для звука {label} (файл {file_path}):\n")
        f.write("=" * 60 + "\n")
        f.write("Время (с) | Формант 1 (Гц) | Формант 2 (Гц) | Формант 3 (Гц)\n")
        f.write("-" * 60 + "\n")
        
        # For each time frame
        for t_idx, (time_sec, peak_freqs, peak_amps) in enumerate(all_formants):
            # Format the line with formant frequencies
            formant_str = f"{time_sec:.2f} | "
            
            # Get top 3 formants by amplitude
            if len(peak_amps) > 0:
                sorted_indices = np.argsort(peak_amps)[::-1]
                
                # Add up to 3 formant frequencies
                for i in range(3):
                    if i < len(sorted_indices):
                        idx = sorted_indices[i]
                        formant_str += f"{peak_freqs[idx]:.2f} | "
                    else:
                        formant_str += "N/A | "
            else:
                formant_str += "N/A | N/A | N/A | "
            
            # Remove the last separator
            formant_str = formant_str[:-3]
            
            # Write to file
            f.write(formant_str + "\n")
    
    # Return key findings for summary
    return {
        'file': file_path,
        'label': label,
        'duration': duration,
        'sample_rate': sr,
        'min_freq': min_freq,
        'max_freq': max_freq,
        'best_f0': best_f0
    }

# Analyze each audio file
results = []
for i, (audio_file, label) in enumerate(zip(audio_files, labels), 1):
    result = analyze_voice(audio_file, label, i)
    results.append(result)
    print("-" * 50)

# Create a summary plot comparing all three audio files
plt.figure(figsize=(12, 10))

# Subplot for min/max frequencies
plt.subplot(2, 1, 1)
x = np.arange(len(results))
width = 0.35

min_freqs = [r['min_freq'] for r in results]
max_freqs = [r['max_freq'] for r in results]

plt.bar(x - width/2, min_freqs, width, label='Минимальная частота (Гц)')
plt.bar(x + width/2, max_freqs, width, label='Максимальная частота (Гц)')

plt.ylabel('Частота (Гц)')
plt.title('Диапазон частот голоса')
plt.xticks(x, [r['label'] for r in results])
plt.legend()

# Subplot for fundamental tones
plt.subplot(2, 1, 2)
fundamental_tones = [r['best_f0'] for r in results]
plt.bar(x, fundamental_tones, width*1.5)

plt.ylabel('Частота (Гц)')
plt.title('Основной тон с наибольшим количеством обертонов')
plt.xticks(x, [r['label'] for r in results])

plt.tight_layout()
plt.savefig('results/voice_comparison.png')
plt.close()

print("Анализ завершен. Результаты сохранены в папке 'results'.") 