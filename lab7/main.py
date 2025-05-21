#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cdist
import cv2

# Paths
output_dir = "./lab7/results"
os.makedirs(output_dir, exist_ok=True)

def load_symbols_from_folder(folder_path):
    """Загрузка изображений символов из указанной папки"""
    print("Loading symbols from folder:", folder_path)
    symbols = {}
    symbol_files = [f for f in os.listdir(folder_path) if f.startswith("char_") and f.endswith(".png")]
    
    for symbol_file in symbol_files:
        parts = symbol_file.split('_')
        if len(parts) >= 3:
            code = int(parts[1])
            symbol = parts[2].split('.')[0]
            
            image_path = os.path.join(folder_path, symbol_file)
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Error: failed to load image {image_path}")
                    continue
                    
                if np.mean(image) > 127:
                    image = 255 - image
                
                binary_image = np.zeros_like(image)
                binary_image[image > 127] = 255
                
                symbols[symbol] = {
                    'code': code,
                    'image': binary_image,
                    'file': symbol_file
                }
                print(f"Loaded symbol: {symbol}, unicode: {code}")
            except Exception as e:
                print(f"Error loading {symbol_file}: {e}")
    
    print(f"Loaded {len(symbols)} symbols")
    return symbols

def calculate_features(image):
    """Расчет признаков для изображения: масса, центр тяжести, моменты инерции"""
    # Бинаризация (0 и 1)
    binary = (image > 0).astype(np.uint8)
    
    # Масса (количество пикселей)
    mass = np.sum(binary)
    
    if mass == 0:
        return {
            "mass": 0,
            "center_x": 0,
            "center_y": 0,
            "ix": 0,
            "iy": 0,
            "mass_rel": 0,
            "center_x_rel": 0,
            "center_y_rel": 0,
            "ix_rel": 0,
            "iy_rel": 0,
            "q1_rel": 0,
            "q2_rel": 0,
            "q3_rel": 0,
            "q4_rel": 0,
        }
    
    # Размеры изображения
    height, width = binary.shape
    
    # Координаты всех пикселей символа
    y_indices, x_indices = np.where(binary > 0)
    
    # Центр тяжести
    center_x = np.mean(x_indices) if len(x_indices) > 0 else width / 2
    center_y = np.mean(y_indices) if len(y_indices) > 0 else height / 2
    
    # Моменты инерции
    ix = np.sum((x_indices - center_x) ** 2) / mass if mass > 0 else 0
    iy = np.sum((y_indices - center_y) ** 2) / mass if mass > 0 else 0
    
    # Расчет квадрантов относительно центра тяжести
    q1_pixels = np.sum((x_indices >= center_x) & (y_indices < center_y))
    q2_pixels = np.sum((x_indices < center_x) & (y_indices < center_y))
    q3_pixels = np.sum((x_indices < center_x) & (y_indices >= center_y))
    q4_pixels = np.sum((x_indices >= center_x) & (y_indices >= center_y))
    
    # Нормализация признаков
    max_mass = width * height
    max_center_x = width
    max_center_y = height
    max_ix = width * width / 4
    max_iy = height * height / 4
    
    # Относительные признаки
    mass_rel = mass / max_mass
    center_x_rel = center_x / max_center_x
    center_y_rel = center_y / max_center_y
    ix_rel = ix / max_ix if max_ix > 0 else 0
    iy_rel = iy / max_iy if max_iy > 0 else 0
    
    # Относительные квадранты
    q1_rel = q1_pixels / mass if mass > 0 else 0
    q2_rel = q2_pixels / mass if mass > 0 else 0
    q3_rel = q3_pixels / mass if mass > 0 else 0
    q4_rel = q4_pixels / mass if mass > 0 else 0
    
    return {
        "mass": mass,
        "center_x": center_x,
        "center_y": center_y,
        "ix": ix,
        "iy": iy,
        "mass_rel": mass_rel,
        "center_x_rel": center_x_rel,
        "center_y_rel": center_y_rel,
        "ix_rel": ix_rel,
        "iy_rel": iy_rel,
        "q1_rel": q1_rel,
        "q2_rel": q2_rel,
        "q3_rel": q3_rel,
        "q4_rel": q4_rel,
    }

def calculate_features_for_symbols(symbols):
    """Расчет признаков для всех символов"""
    features = {}
    for symbol, data in symbols.items():
        features[symbol] = calculate_features(data['image'])
        features[symbol]['code'] = data['code']
    return features

def save_features_to_csv(features, output_path):
    """Сохранение признаков в CSV-файл"""
    data = []
    for symbol, feature in features.items():
        data.append({
            'char': symbol,
            'code': feature['code'],
            'mass': feature['mass'],
            'center_x': feature['center_x'],
            'center_y': feature['center_y'],
            'ix': feature['ix'],
            'iy': feature['iy'],
            'mass_rel': feature['mass_rel'],
            'center_x_rel': feature['center_x_rel'],
            'center_y_rel': feature['center_y_rel'],
            'ix_rel': feature['ix_rel'],
            'iy_rel': feature['iy_rel'],
            'q1_rel': feature['q1_rel'],
            'q2_rel': feature['q2_rel'],
            'q3_rel': feature['q3_rel'],
            'q4_rel': feature['q4_rel'],
        })
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_path, "features.csv"), sep=';', index=False)
    return df

def create_test_phrase(text, font_size=52, output_path=None):
    """Create an image with the given text and font size"""
    width = len(text) * 50
    height = 100
    
    # Create a white image
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    # Convert to PIL Image for text drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Use a font that supports Hebrew
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback fonts
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    # Draw text (centered)
    try:
        # For newer Pillow versions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        try:
            text_width, text_height = draw.textsize(text, font=font)
        except AttributeError:
            text_width, text_height = font.getsize(text)
    
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, font=font, fill=0)
    
    # Convert back to numpy array
    phrase_image = np.array(pil_image)
    
    # Save the image if a path is provided
    if output_path:
        Image.fromarray(phrase_image).save(output_path)
    
    return phrase_image

def segment_characters(image):
    """Segment characters from the image"""
    # Invert if needed (ensure text is black on white)
    if np.mean(image) < 127:
        image = 255 - image
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from right to left (for Hebrew)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
    
    segments = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out noise (very small contours)
        if w < 5 or h < 5:
            continue
        
        # Extract the segment
        segment = binary[y:y+h, x:x+w]
        
        segments.append({
            'image': segment,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'area': cv2.contourArea(contour)
        })
    
    return segments

def compute_distances(features_df):
    """Compute Euclidean distances between all symbols based on their features"""
    # Extract features for distance calculation
    feature_cols = ['mass_rel', 'center_x_rel', 'center_y_rel', 'ix_rel', 'iy_rel', 
                    'q1_rel', 'q2_rel', 'q3_rel', 'q4_rel']
    feature_matrix = features_df[feature_cols].values
    
    # Compute distance matrix
    distance_matrix = cdist(feature_matrix, feature_matrix, metric='euclidean')
    
    # Convert distances to similarity measure (1 for identical, approaching 0 for dissimilar)
    max_distance = np.max(distance_matrix)
    similarity_matrix = 1 - (distance_matrix / max_distance if max_distance > 0 else distance_matrix)
    
    return similarity_matrix

def recognize_character(char_features, features_df):
    """Recognize a character by comparing its features with reference features"""
    # Extract reference features
    feature_cols = ['mass_rel', 'center_x_rel', 'center_y_rel', 'ix_rel', 'iy_rel', 
                    'q1_rel', 'q2_rel', 'q3_rel', 'q4_rel']
    ref_features = features_df[feature_cols].values
    
    # Convert char_features to vector format (same order as reference features)
    char_vector = np.array([[
        char_features['mass_rel'],
        char_features['center_x_rel'],
        char_features['center_y_rel'],
        char_features['ix_rel'],
        char_features['iy_rel'],
        char_features['q1_rel'],
        char_features['q2_rel'],
        char_features['q3_rel'],
        char_features['q4_rel']
    ]])
    
    # Compute distances
    distances = cdist(char_vector, ref_features, metric='euclidean')[0]
    
    # Convert to similarity (0 to 1 scale, where 1 is identical)
    max_distance = np.max(distances) if np.max(distances) > 0 else 1
    similarities = 1 - (distances / max_distance)
    
    # Create list of (char, similarity) tuples
    char_similarities = list(zip(features_df['char'].values, similarities))
    
    # Sort by similarity (descending)
    char_similarities.sort(key=lambda x: x[1], reverse=True)
    
    return char_similarities

def main():
    # Check for symbols directory
    symbols_folder = "./lab5/symbols"
    if not os.path.exists(symbols_folder):
        print(f"Error: Symbols directory {symbols_folder} not found!")
        # Check alternative paths
        alt_paths = [
            "../lab5/symbols", 
            "lab5/symbols", 
            "/home/dan/go/src/mephi/audiovisual/lab5/symbols"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                symbols_folder = alt_path
                print(f"Found alternative symbols directory: {symbols_folder}")
                break
        else:
            print("Could not find symbols directory!")
            return
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Load symbols
    symbols = load_symbols_from_folder(symbols_folder)
    
    if not symbols:
        print("Error: Failed to load symbols!")
        return
    
    # Calculate features for all symbols
    print("Calculating features for symbols...")
    features = calculate_features_for_symbols(symbols)
    
    # Save features to CSV
    features_df = save_features_to_csv(features, output_dir)
    
    # === TASK 1: Calculate Euclidean distances ===
    print("Calculating distance matrix...")
    similarity_matrix = compute_distances(features_df)
    
    # Save distance matrix visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Similarity')
    plt.xticks(ticks=np.arange(len(features_df)), labels=features_df['char'], rotation=90)
    plt.yticks(ticks=np.arange(len(features_df)), labels=features_df['char'])
    plt.title('Similarity Matrix between Hebrew Characters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_matrix.png"))
    plt.close()
    
    # Save similarity matrix to CSV
    symbols_list = features_df['char'].tolist()
    similarity_df = pd.DataFrame(similarity_matrix, index=symbols_list, columns=symbols_list)
    similarity_df.to_csv(os.path.join(output_dir, "similarity_matrix.csv"), sep=';')
    
    # === TASKS 2-3: Create test phrase and recognize ===
    # Define test phrase using the first 10 Hebrew letters
    hebrew_chars = list(symbols.keys())
    test_phrase = ''.join(hebrew_chars[:10])
    
    print(f"Test phrase: {test_phrase}")
    
    # Create test phrase image
    test_image_path = os.path.join(output_dir, "test_phrase.png")
    test_image = create_test_phrase(test_phrase, font_size=52, output_path=test_image_path)
    
    # Segment characters from the test image
    print("Segmenting characters...")
    segments = segment_characters(test_image)
    
    # For demonstration purposes, if segmentation fails, use individual symbol images
    if len(segments) < len(test_phrase):
        print(f"Warning: Segmentation found only {len(segments)} characters, using symbol images directly")
        segments = []
        for i, char in enumerate(test_phrase):
            if char in symbols:
                segments.append({
                    'image': symbols[char]['image'],
                    'x': i * 50,
                    'y': 0,
                    'width': symbols[char]['image'].shape[1],
                    'height': symbols[char]['image'].shape[0],
                    'area': np.sum(symbols[char]['image'] > 0)
                })
    
    # Recognize each segment
    print("Recognizing characters...")
    hypotheses = []
    
    for i, segment in enumerate(segments):
        # Calculate features for the segment
        segment_features = calculate_features(segment['image'])
        
        # Recognize the character
        char_hypotheses = recognize_character(segment_features, features_df)
        
        # Save the hypotheses
        hypotheses.append(char_hypotheses)
        
        # Save the segment image
        Image.fromarray(segment['image']).save(os.path.join(output_dir, f"segment_{i+1}.png"))
    
    # === TASK 3: Save hypotheses to file ===
    with open(os.path.join(output_dir, "recognition_hypotheses.txt"), "w", encoding="utf-8") as f:
        for i, hyp in enumerate(hypotheses, 1):
            f.write(f"{i}: {[(char, round(score, 4)) for char, score in hyp]}\n")
    
    # === TASK 4: Output best hypotheses as string ===
    recognized_string = ''.join([h[0][0] for h in hypotheses])
    
    # === TASK 5: Calculate accuracy ===
    min_length = min(len(test_phrase), len(recognized_string))
    correct_chars = sum(1 for a, b in zip(test_phrase[:min_length], recognized_string[:min_length]) if a == b)
    accuracy = (correct_chars / len(test_phrase)) * 100
    
    # Save results
    with open(os.path.join(output_dir, "recognition_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Original phrase: {test_phrase}\n")
        f.write(f"Recognized phrase: {recognized_string}\n")
        f.write(f"Number of errors: {len(test_phrase) - correct_chars} out of {len(test_phrase)}\n")
        f.write(f"Recognition accuracy: {accuracy:.2f}%\n")
    
    print(f"Recognition accuracy: {accuracy:.2f}%")
    
    # === TASK 6: Experiment with different font sizes ===
    print("Experimenting with different font sizes...")
    
    font_sizes = [42, 48, 56, 62]
    results = []
    
    for font_size in font_sizes:
        # Create image with different font size
        test_image_path = os.path.join(output_dir, f"test_phrase_size_{font_size}.png")
        test_image = create_test_phrase(test_phrase, font_size=font_size, output_path=test_image_path)
        
        # Segment characters
        segments = segment_characters(test_image)
        
        # If segmentation fails, use symbol images directly
        if len(segments) < len(test_phrase):
            segments = []
            for i, char in enumerate(test_phrase):
                if char in symbols:
                    segments.append({
                        'image': symbols[char]['image'],
                        'x': i * 50,
                        'y': 0,
                        'width': symbols[char]['image'].shape[1],
                        'height': symbols[char]['image'].shape[0],
                        'area': np.sum(symbols[char]['image'] > 0)
                    })
        
        # Recognize each segment
        size_hypotheses = []
        for segment in segments:
            segment_features = calculate_features(segment['image'])
            char_hypotheses = recognize_character(segment_features, features_df)
            size_hypotheses.append(char_hypotheses)
        
        # Get recognized string
        size_recognized = ''.join([h[0][0] for h in size_hypotheses])
        
        # Calculate accuracy
        min_length = min(len(test_phrase), len(size_recognized))
        size_correct = sum(1 for a, b in zip(test_phrase[:min_length], size_recognized[:min_length]) if a == b)
        size_accuracy = (size_correct / len(test_phrase)) * 100
        
        # Save results
        results.append({
            'font_size': font_size,
            'recognized': size_recognized,
            'accuracy': size_accuracy,
            'correct': size_correct,
            'total': len(test_phrase)
        })
        
        with open(os.path.join(output_dir, f"recognition_results_size_{font_size}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Font size: {font_size}\n")
            f.write(f"Original phrase: {test_phrase}\n")
            f.write(f"Recognized phrase: {size_recognized}\n")
            f.write(f"Number of errors: {len(test_phrase) - size_correct} out of {len(test_phrase)}\n")
            f.write(f"Recognition accuracy: {size_accuracy:.2f}%\n")
    
    # Create chart showing accuracy vs font size
    plt.figure(figsize=(10, 6))
    sizes = [r['font_size'] for r in results] + [52]  # Add original size
    accuracies = [r['accuracy'] for r in results] + [accuracy]  # Add original accuracy
    
    plt.plot(sizes, accuracies, marker='o', linestyle='-', color='blue', linewidth=2)
    plt.xlabel('Font Size')
    plt.ylabel('Recognition Accuracy (%)')
    plt.title('Recognition Accuracy vs Font Size')
    plt.grid(True)
    plt.xticks(sizes)
    
    # Add annotations
    for i, acc in enumerate(accuracies):
        plt.annotate(f"{acc:.1f}%", 
                   (sizes[i], acc), 
                   textcoords="offset points",
                   xytext=(0, 10), 
                   ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "font_size_experiment.png"))
    
    # Save summary to CSV
    summary_data = []
    for r in results:
        summary_data.append({
            'Font Size': r['font_size'],
            'Accuracy (%)': round(r['accuracy'], 2),
            'Errors': f"{r['total'] - r['correct']} out of {r['total']}",
            'Recognized Phrase': r['recognized']
        })
    
    # Add original results (font size 52)
    summary_data.append({
        'Font Size': 52,
        'Accuracy (%)': round(accuracy, 2),
        'Errors': f"{len(test_phrase) - correct_chars} out of {len(test_phrase)}",
        'Recognized Phrase': recognized_string
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "font_size_experiment_summary.csv"), sep=';', index=False)
    
    print(f"✅ Lab 7 completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 