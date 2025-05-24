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
file_prefixes = ["AAA", "III", "GAF"]

# Function to analyze voice files
def analyze_voice(file_path, label, file_number, file_prefix):
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
    
    # Plot min/max frequency graph (Task 3)
    plt.figure(figsize=(12, 6))
    # Get average spectrum across all time frames
    avg_spectrum = np.mean(S, axis=1)
    plt.plot(frequencies, avg_spectrum, 'b-')
    plt.axvline(x=min_freq, color='g', linestyle='--', label=f'Мин: {min_freq:.1f} Гц')
    plt.axvline(x=max_freq, color='r', linestyle='--', label=f'Макс: {max_freq:.1f} Гц')
    plt.xscale('linear')
    plt.xlim(0, sr/2)  # Show full frequency range
    plt.xlabel('Частота [Гц]')
    plt.ylabel('Амплитуда')
    plt.title(f'Задание 3: Частоты сигнала {file_prefix}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/3_{file_prefix}_min_max_freq.png')
    plt.close()
    
    # 4. Find the most timbrally colored fundamental tone (with the most overtones)
    
    # Use harmonic-percussive source separation to enhance harmonic content
    y_harmonic, _ = librosa.effects.hpss(y)
    
    # Calculate pitch (f0) and harmonics using Spectral Harmonic-to-Noise Ratio
    pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # For each time frame, count overtones above significant threshold
    overtone_counts = []
    fundamental_tones = []
    overtone_energies = []
    
    # For each time frame
    for t in range(pitches.shape[1]):
        # Get the pitch with maximum magnitude in this frame
        idx = np.argmax(magnitudes[:, t])
        if magnitudes[idx, t] > 0:  # If we have a pitch
            f0 = pitches[idx, t]
            
            # Count frames with significant harmonic content
            harmonic_count = 0
            current_overtones = []
            
            # Check for overtones (integer multiples of f0)
            for n in range(1, 10):  # Check fundamental and up to 9th harmonic
                target_freq = f0 * n
                
                # Find the closest frequency bin
                closest_idx = np.argmin(np.abs(frequencies - target_freq))
                
                # Get energy at this frequency
                energy = S[closest_idx, t] if closest_idx < len(frequencies) else 0
                
                # Store overtone information (harmonic number, frequency, energy)
                current_overtones.append((n, target_freq, energy))
                
                # Check if there's significant energy at this harmonic
                if closest_idx < len(frequencies) and S[closest_idx, t] > threshold:
                    harmonic_count += 1
            
            overtone_counts.append(harmonic_count)
            fundamental_tones.append(f0)
            overtone_energies.append(current_overtones)
    
    # Find the fundamental tone with the most overtones
    if overtone_counts:
        max_overtones_idx = np.argmax(overtone_counts)
        best_f0 = fundamental_tones[max_overtones_idx]
        overtone_count = overtone_counts[max_overtones_idx]
        best_overtones = overtone_energies[max_overtones_idx]
        
        print(f"Most timbrally colored fundamental tone: {best_f0:.2f} Hz (with {overtone_count} significant overtones)")
        
        # Plot overtones graph (Task 4)
        plt.figure(figsize=(12, 6))
        harmonic_numbers = [o[0] for o in best_overtones]
        harmonic_freqs = [o[1] for o in best_overtones]
        harmonic_energies = [o[2] for o in best_overtones]
        
        plt.bar(harmonic_numbers, harmonic_energies, width=0.5, alpha=0.7)
        plt.xlabel('Номер гармоники')
        plt.ylabel('Энергия')
        plt.title(f'Задание 4: Обертоны сигнала {file_prefix}\nОсновной тон: {best_f0:.2f} Гц')
        plt.grid(True, axis='y')
        plt.xticks(harmonic_numbers)
        
        # Add frequency labels above bars
        for i, (num, freq, energy) in enumerate(best_overtones):
            plt.text(num, energy + max(harmonic_energies)*0.02, f'{freq:.1f} Гц', 
                    ha='center', va='bottom', rotation=45, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'results/4_{file_prefix}_harmonics.png')
        plt.close()
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
    all_formant_freqs = []  # To calculate average
    
    # For each time frame with step of 0.1s
    for t in range(0, S.shape[1], frame_step):
        if t >= S.shape[1]:
            break
            
        # Get the spectrum at this time
        spectrum = S[:, t]
        
        # Find peaks (formants)
        # Use a relatively small distance to get finer frequency resolution
        # distance≈45Hz in frequency bins
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
        
        # Sort by amplitude (descending) to get strongest formants
        sorted_indices = np.argsort(peak_amps)[::-1]
        peak_freqs = peak_freqs[sorted_indices]
        peak_amps = peak_amps[sorted_indices]
        
        # Store time and formant data
        time_sec = t * hop_length / sr
        all_formants.append((time_sec, peak_freqs, peak_amps))
        
        # Store formant frequencies for averaging
        all_formant_freqs.extend(peak_freqs[:3] if len(peak_freqs) >= 3 else peak_freqs)
    
    # Calculate average formant frequency
    avg_formant = np.mean(all_formant_freqs) if all_formant_freqs else 0
    
    # Create arrays to store specific formant data for plotting
    formant_times = []
    formant1_freqs = []
    formant2_freqs = []
    formant3_freqs = []
    
    # Extract specific formant data from all_formants
    for time_sec, peak_freqs, peak_amps in all_formants:
        formant_times.append(time_sec)
        
        # First formant (if available)
        if len(peak_freqs) > 0:
            formant1_freqs.append(peak_freqs[0])
        else:
            formant1_freqs.append(np.nan)
            
        # Second formant (if available)
        if len(peak_freqs) > 1:
            formant2_freqs.append(peak_freqs[1])
        else:
            formant2_freqs.append(np.nan)
            
        # Third formant (if available)
        if len(peak_freqs) > 2:
            formant3_freqs.append(peak_freqs[2])
        else:
            formant3_freqs.append(np.nan)
    
    # Plot formants with average value (Task 5)
    plt.figure(figsize=(12, 6))
    
    # Plot each formant track
    plt.plot(formant_times, formant1_freqs, 'b-', label='Форманта 1')
    plt.plot(formant_times, formant2_freqs, 'orange', label='Форманта 2')
    plt.plot(formant_times, formant3_freqs, 'g-', label='Форманта 3')
    
    # Add horizontal line for average
    plt.axhline(y=avg_formant, color='k', linestyle='--', 
               label=f'Общее среднее: {avg_formant:.1f} Гц')
    
    plt.xlabel('Время [с]')
    plt.ylabel('Частота [Гц]')
    plt.title(f'Задание 5: Форманты сигнала {file_prefix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/5_{file_prefix}_formants.png')
    plt.close()
    
    # Plot summed formants (Task 5.1)
    plt.figure(figsize=(12, 6))
    
    # Sum formant frequencies for each time point
    summed_formants = []
    for i, (time_sec, peak_freqs, peak_amps) in enumerate(all_formants):
        if len(peak_freqs) >= 3:
            # Sum top 3 formant frequencies
            formant_sum = formant1_freqs[i] + formant2_freqs[i] + formant3_freqs[i]
            summed_formants.append((time_sec, formant_sum))
    
    # Plot summed formants
    if summed_formants:
        times, sums = zip(*summed_formants)
        avg_sum = np.mean(sums)
        
        plt.plot(times, sums, 'b-', label='Сумма 3 формант')
        plt.axhline(y=avg_sum, color='k', linestyle='--', 
                   label=f'Среднее: {avg_sum:.1f} Гц')
        
        plt.xlabel('Время [с]')
        plt.ylabel('Суммарная частота [Гц]')
        plt.title(f'Задание 5.1: Суммарная форманта сигнала {file_prefix}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/5_1_{file_prefix}_formants_sum.png')
        plt.close()
    
    # Plot the spectrogram as background with formants overlay
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', 
                            y_axis='log', cmap='gray_r', alpha=0.7)
    
    # Plot each formant track with a different color
    plt.plot(formant_times, formant1_freqs, 'r-o', markersize=4, label='Формант 1')
    plt.plot(formant_times, formant2_freqs, 'g-o', markersize=4, label='Формант 2')
    plt.plot(formant_times, formant3_freqs, 'b-o', markersize=4, label='Формант 3')
    
    # Mark specific time points from the table for reference
    table_times = [3.0, 3.1, 3.2] if file_number == 1 else [4.0] if file_number == 2 else []
    for t in table_times:
        plt.axvline(x=t, color='yellow', linestyle='--', linewidth=1)
        plt.text(t, 100, f"{t} c", color='yellow', ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
    
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
        for i, (time_sec, peak_freqs, peak_amps) in enumerate(all_formants):
            # Format the line with formant frequencies
            formant_str = f"{time_sec:.2f} | "
            
            # Add up to 3 formant frequencies
            formant_str += f"{formant1_freqs[i]:.2f} | " if i < len(formant1_freqs) and not np.isnan(formant1_freqs[i]) else "N/A | "
            formant_str += f"{formant2_freqs[i]:.2f} | " if i < len(formant2_freqs) and not np.isnan(formant2_freqs[i]) else "N/A | "
            formant_str += f"{formant3_freqs[i]:.2f}" if i < len(formant3_freqs) and not np.isnan(formant3_freqs[i]) else "N/A"
            
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
        'best_f0': best_f0,
        'avg_formant': avg_formant,
        'formant1_range': [np.nanmin(formant1_freqs), np.nanmax(formant1_freqs)] if not all(np.isnan(x) for x in formant1_freqs) else [0, 0],
        'formant2_range': [np.nanmin(formant2_freqs), np.nanmax(formant2_freqs)] if not all(np.isnan(x) for x in formant2_freqs) else [0, 0],
        'formant3_range': [np.nanmin(formant3_freqs), np.nanmax(formant3_freqs)] if not all(np.isnan(x) for x in formant3_freqs) else [0, 0]
    }

# Analyze each audio file
results = []
for i, (audio_file, label, file_prefix) in enumerate(zip(audio_files, labels, file_prefixes), 1):
    result = analyze_voice(audio_file, label, i, file_prefix)
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

# Print formant ranges for each sound
for r in results:
    print(f"Звук {r['label']}:")
    print(f"Формант 1 диапазон: {r['formant1_range'][0]:.2f}-{r['formant1_range'][1]:.2f} Гц")
    print(f"Формант 2 диапазон: {r['formant2_range'][0]:.2f}-{r['formant2_range'][1]:.2f} Гц")
    print(f"Формант 3 диапазон: {r['formant3_range'][0]:.2f}-{r['formant3_range'][1]:.2f} Гц")
    print("-" * 50)

print("Анализ завершен. Результаты сохранены в папке 'results'.") 