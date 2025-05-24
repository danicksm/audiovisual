#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import List, Tuple, Dict
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FONT_PATH = "../lab5/Hebrew.ttf"
FEATURES_CSV_PATH = "../lab5/results/features.csv"
TEST_IMAGE_PATH = "../lab6/phrase.bmp"
FONT_SIZE = 52
OUTPUT_DIR_ORIGINALS = "symbols/originals"
RESULTS_DIR = "results"

# Ensure directories exist
os.makedirs(OUTPUT_DIR_ORIGINALS, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load features from lab5
def load_features() -> pd.DataFrame:
    """Load character features from CSV file generated in lab5."""
    try:
        features_df = pd.read_csv(FEATURES_CSV_PATH, sep=';')
        logger.info(f"Loaded features for {len(features_df)} characters")
        return features_df
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        sys.exit(1)

def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess an image."""
    try:
        img = np.array(Image.open(image_path))
        logger.info(f"Loaded image with shape {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(1)

def binarize_image(img: np.ndarray) -> np.ndarray:
    """Binarize the image."""
    # Ensure image is grayscale
    if len(img.shape) == 3:
        # Convert RGB to grayscale
        img_gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        img_gray = img
    
    # Binarize (invert if text is dark on light background)
    binary_threshold = 128
    img_bin = np.zeros_like(img_gray)
    img_bin[img_gray < binary_threshold] = 255
    
    logger.info("Image binarized")
    return img_bin

def find_intervals(profile: np.ndarray, threshold: int = 5) -> List[Tuple[int, int]]:
    """Find intervals in a profile where values are above threshold."""
    intervals = []
    start = None
    
    for i, value in enumerate(profile):
        if value > threshold and start is None:
            start = i
        elif value <= threshold and start is not None:
            # Filter out too small intervals
            if i - start >= 2:
                intervals.append((start, i))
            start = None
    
    # Don't forget the last interval if it ends at the profile's end
    if start is not None and len(profile) - start >= 2:
        intervals.append((start, len(profile)))
        
    return intervals

def segment_text(img_bin: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Segment text into character bounding boxes."""
    # Calculate horizontal profile (sum along rows)
    horizontal_profile = np.sum(img_bin, axis=1)
    
    # Find rows containing text
    row_intervals = find_intervals(horizontal_profile)
    logger.info(f"Found {len(row_intervals)} text rows")
    
    # For each text row, find character intervals
    char_boxes = []
    
    for row_idx, (y1, y2) in enumerate(row_intervals):
        # Extract the row and calculate its vertical profile
        row = img_bin[y1:y2, :]
        vertical_profile = np.sum(row, axis=0)
        
        # Find character intervals in this row
        char_intervals = find_intervals(vertical_profile)
        logger.info(f"Found {len(char_intervals)} characters in row {row_idx+1}")
        
        # Create bounding boxes for each character
        for x1, x2 in char_intervals:
            char_boxes.append((x1, y1, x2, y2))
    
    return char_boxes

def extract_char_features(img_bin: np.ndarray, box: Tuple[int, int, int, int]) -> Dict:
    """Extract features for a character defined by its bounding box."""
    x1, y1, x2, y2 = box
    
    # Extract the character
    char_img = img_bin[y1:y2, x1:x2]
    height, width = char_img.shape
    
    # Invert if necessary (ensure character is 1, background is 0)
    if np.mean(char_img) > 127:  # If mean is high, image is likely inverted
        binary = (char_img < 128).astype(np.uint8)
    else:
        binary = (char_img >= 128).astype(np.uint8)
    
    # Calculate total weight (mass)
    total_weight = np.sum(binary)
    
    # If character has no weight (empty area), return default values
    if total_weight == 0:
        return {
            'total_weight': 0,
            'norm_cog_x': 0.5,
            'norm_cog_y': 0.5,
            'norm_moment_x': 0,
            'norm_moment_y': 0
        }
    
    # Calculate center of gravity
    y_indices, x_indices = np.indices(binary.shape)
    cog_x = np.sum(x_indices * binary) / total_weight
    cog_y = np.sum(y_indices * binary) / total_weight
    
    # Normalize center of gravity
    norm_cog_x = cog_x / width
    norm_cog_y = cog_y / height
    
    # Calculate axial moments of inertia
    moment_x = np.sum(((y_indices - cog_y) ** 2) * binary)
    moment_y = np.sum(((x_indices - cog_x) ** 2) * binary)
    
    # Normalize moments
    norm_moment_x = moment_x / (total_weight * height ** 2) if total_weight > 0 else 0
    norm_moment_y = moment_y / (total_weight * width ** 2) if total_weight > 0 else 0
    
    return {
        'total_weight': total_weight,
        'norm_cog_x': norm_cog_x,
        'norm_cog_y': norm_cog_y,
        'norm_moment_x': norm_moment_x,
        'norm_moment_y': norm_moment_y
    }

def calculate_similarity(char_features: Dict, reference_features: pd.DataFrame) -> List[Tuple[str, float]]:
    """Calculate similarity between a character and all reference characters."""
    similarities = []
    
    # Features to compare
    feature_keys = ['total_weight', 'norm_cog_x', 'norm_cog_y', 'norm_moment_x', 'norm_moment_y']
    
    # Normalize the weight by getting an average reference weight
    avg_reference_weight = reference_features['Weight_Q1'].sum() + reference_features['Weight_Q2'].sum() + \
                           reference_features['Weight_Q3'].sum() + reference_features['Weight_Q4'].sum()
    avg_reference_weight /= len(reference_features)
    
    # Create a normalized feature vector for the character
    char_vector = np.array([
        char_features['total_weight'] / avg_reference_weight,
        char_features['norm_cog_x'],
        char_features['norm_cog_y'],
        char_features['norm_moment_x'],
        char_features['norm_moment_y']
    ])
    
    for _, row in reference_features.iterrows():
        char = row['Character']
        
        # Create a normalized feature vector for the reference character
        ref_total_weight = row['Weight_Q1'] + row['Weight_Q2'] + row['Weight_Q3'] + row['Weight_Q4']
        ref_vector = np.array([
            ref_total_weight / avg_reference_weight,
            row['NormCoG_X'],
            row['NormCoG_Y'],
            row['NormMomentOfInertia_X'],
            row['NormMomentOfInertia_Y']
        ])
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(char_vector - ref_vector)
        
        # Convert distance to similarity (1 for identical, approaching 0 for very different)
        # Using exponential decay to map distance to [0,1]
        similarity = math.exp(-distance)
        
        similarities.append((char, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities

def save_comparison_results(char_similarities: List[List[Tuple[str, float]]], original_text: str = ""):
    """Save the comparison results to a file."""
    result_path = os.path.join(RESULTS_DIR, "comparison_results.txt")
    
    with open(result_path, 'w', encoding='utf-8') as f:
        for i, similarities in enumerate(char_similarities):
            f.write(f"{i+1}: {similarities}\n")
        
        # Extract best guesses
        best_guesses = [sim[0][0] for sim in char_similarities]
        recognized_text = ''.join(best_guesses)
        
        f.write("\nRecognized text: " + recognized_text + "\n")
        
        if original_text:
            # Calculate accuracy
            correct = sum(1 for a, b in zip(original_text, recognized_text) if a == b)
            total = max(len(original_text), len(recognized_text))
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            f.write(f"\nOriginal text: {original_text}\n")
            f.write(f"Correctly recognized: {correct}/{total} characters ({accuracy:.2f}%)\n")
    
    logger.info(f"Comparison results saved to {result_path}")
    return recognized_text, best_guesses

def save_recognition_visualization(img: np.ndarray, char_boxes: List[Tuple[int, int, int, int]], 
                                  best_guesses: List[str]):
    """Save a visualization of the recognition results."""
    # Convert to RGB for visualization
    if len(img.shape) == 2:
        img_rgb = np.stack([img] * 3, axis=2)
    else:
        img_rgb = img.copy()
    
    # Draw bounding boxes and best guesses
    for i, ((x1, y1, x2, y2), char) in enumerate(zip(char_boxes, best_guesses)):
        # Draw bounding box
        # Top horizontal line
        img_rgb[y1:y1+2, x1:x2, 0] = 255  # Red
        img_rgb[y1:y1+2, x1:x2, 1] = 0
        img_rgb[y1:y1+2, x1:x2, 2] = 0
        
        # Bottom horizontal line
        img_rgb[y2-2:y2, x1:x2, 0] = 255
        img_rgb[y2-2:y2, x1:x2, 1] = 0
        img_rgb[y2-2:y2, x1:x2, 2] = 0
        
        # Left vertical line
        img_rgb[y1:y2, x1:x1+2, 0] = 255
        img_rgb[y1:y2, x1:x1+2, 1] = 0
        img_rgb[y1:y2, x1:x1+2, 2] = 0
        
        # Right vertical line
        img_rgb[y1:y2, x2-2:x2, 0] = 255
        img_rgb[y1:y2, x2-2:x2, 1] = 0
        img_rgb[y1:y2, x2-2:x2, 2] = 0
        
        # Add character label on top of the box
        # This is a simple implementation; for production, you might want to use
        # a proper text rendering library that handles different languages
        y_text = max(0, y1 - 15)
        x_text = x1
        
        # Create a small white background for text
        text_height = 15
        text_width = 15
        img_rgb[y_text:y_text+text_height, x_text:x_text+text_width, :] = [255, 255, 255]
        
        # Draw the character (approximation - in real application use proper text rendering)
        img_rgb[y_text+2:y_text+text_height-2, x_text+2:x_text+text_width-2, 0] = 0
        img_rgb[y_text+2:y_text+text_height-2, x_text+2:x_text+text_width-2, 1] = 0
        img_rgb[y_text+2:y_text+text_height-2, x_text+2:x_text+text_width-2, 2] = 0
    
    # Save visualization
    result_path = os.path.join(RESULTS_DIR, "recognition_visualization.png")
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Character Recognition Results')
    plt.tight_layout()
    plt.savefig(result_path, dpi=150)
    plt.close()
    
    logger.info(f"Recognition visualization saved to {result_path}")

def generate_test_image(text: str, font_size: int) -> str:
    """Generate a test image with the given text and font size."""
    output_path = os.path.join(RESULTS_DIR, f"test_image_size_{font_size}.png")
    
    try:
        # Create a larger image to ensure text fits
        img_width, img_height = 800, 200
        img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Load font
        font = ImageFont.truetype(FONT_PATH, font_size)
        
        # Calculate position to center the text
        text_width = draw.textlength(text, font=font)
        text_height = font_size  # Approximate
        
        # For right-to-left text like Hebrew, position from right side
        position = (img_width - text_width - 50, (img_height - text_height) // 2)
        
        # Draw the text
        draw.text(position, text, font=font, fill=(0, 0, 0))
        
        # Save the image
        img.save(output_path)
        logger.info(f"Generated test image with font size {font_size} at {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate test image: {e}")
        return ""

def main():
    """Main function to execute all tasks."""
    logger.info("Starting Lab 7 - Classification based on features")
    
    # Load features from lab5
    reference_features = load_features()
    
    # Task 1-5: Classify characters in the text from lab6
    # Load and preprocess the image
    img = load_image(TEST_IMAGE_PATH)
    img_bin = binarize_image(img)
    
    # Segment text into characters
    char_boxes = segment_text(img_bin)
    logger.info(f"Segmented {len(char_boxes)} characters")
    
    # Calculate features and similarities for each character
    char_similarities = []
    
    for i, box in enumerate(char_boxes):
        # Extract features
        char_features = extract_char_features(img_bin, box)
        
        # Calculate similarity with all reference characters
        similarities = calculate_similarity(char_features, reference_features)
        char_similarities.append(similarities)
        
        logger.info(f"Character {i+1}: Best match is '{similarities[0][0]}' with similarity {similarities[0][1]:.4f}")
    
    # Save comparison results
    recognized_text, best_guesses = save_comparison_results(char_similarities)
    
    # Save visualization
    save_recognition_visualization(img, char_boxes, best_guesses)
    
    # Task 6: Generate test image with different font size and recognize it
    # Extract the first row of text from the recognized text (to use as test text)
    # For simplicity, we'll use a small portion of the recognized text
    test_text = recognized_text[:10] if len(recognized_text) > 10 else recognized_text
    
    # Try with a different font size
    different_font_size = FONT_SIZE + 10  # 10 points larger
    test_image_path = generate_test_image(test_text, different_font_size)
    
    if test_image_path:
        # Recognize the generated test image
        test_img = load_image(test_image_path)
        test_img_bin = binarize_image(test_img)
        
        # Segment text into characters
        test_char_boxes = segment_text(test_img_bin)
        logger.info(f"Segmented {len(test_char_boxes)} characters in test image")
        
        # Calculate features and similarities for each character
        test_char_similarities = []
        
        for i, box in enumerate(test_char_boxes):
            # Extract features
            char_features = extract_char_features(test_img_bin, box)
            
            # Calculate similarity with all reference characters
            similarities = calculate_similarity(char_features, reference_features)
            test_char_similarities.append(similarities)
            
            logger.info(f"Test Character {i+1}: Best match is '{similarities[0][0]}' with similarity {similarities[0][1]:.4f}")
        
        # Save comparison results for the test image
        test_result_path = os.path.join(RESULTS_DIR, "test_comparison_results.txt")
        test_recognized_text, test_best_guesses = save_comparison_results(test_char_similarities, test_text)
        
        # Save visualization for the test image
        test_result_viz_path = os.path.join(RESULTS_DIR, "test_recognition_visualization.png")
        save_recognition_visualization(test_img, test_char_boxes, test_best_guesses)
        
        # Compare results
        logger.info(f"Original text: {test_text}")
        logger.info(f"Recognized text (original font size): {recognized_text}")
        logger.info(f"Recognized text (different font size): {test_recognized_text}")
    
    logger.info("Lab 7 completed successfully!")

if __name__ == "__main__":
    main()
