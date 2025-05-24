#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from scipy.spatial.distance import euclidean
import pickle
import shutil
from PIL import Image, ImageDraw, ImageFont

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
LAB5_DIR = "../lab5"
LAB6_DIR = "../lab6"
RESULTS_DIR = "results"
FEATURES_CSV = os.path.join(LAB5_DIR, "results", "features.csv")
PHRASE_BMP = os.path.join(LAB6_DIR, "phrase.bmp")
LAB6_CHARACTERS_DIR = os.path.join(LAB6_DIR, "results")
FONT_PATH = os.path.join(LAB5_DIR, "Hebrew.ttf")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_features():
    """
    Load features of reference symbols from lab5.
    
    Returns:
        pd.DataFrame: DataFrame with symbol features
    """
    logger.info(f"Loading features from {FEATURES_CSV}")
    
    # Load features CSV
    features_df = pd.read_csv(FEATURES_CSV, sep=';')
    
    logger.info(f"Loaded features for {len(features_df)} symbols")
    return features_df

def load_segmented_characters():
    """
    Load segmented characters from lab6.
    
    Returns:
        list: List of character images
    """
    logger.info(f"Loading segmented characters from {LAB6_CHARACTERS_DIR}")
    
    characters = []
    char_filenames = []
    
    # Find all character files in lab6/results
    for filename in sorted(os.listdir(LAB6_CHARACTERS_DIR)):
        if filename.startswith("char_") and filename.endswith(".png") and not "_profiles" in filename:
            char_filenames.append(filename)
    
    # Sort character files by number
    char_filenames.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    
    # Load each character image
    for filename in char_filenames:
        filepath = os.path.join(LAB6_CHARACTERS_DIR, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Convert to binary (0 for background, 1 for character)
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            # Invert so character is 1, background is 0
            binary = cv2.bitwise_not(binary)
            binary = binary / 255  # Normalize to 0-1
            characters.append(binary)
            logger.debug(f"Loaded character from {filename}")
    
    logger.info(f"Loaded {len(characters)} segmented characters")
    return characters

def calculate_character_features(char_img):
    """
    Calculate features for a character image.
    
    Args:
        char_img (numpy.ndarray): Binary character image (0-1 values)
        
    Returns:
        dict: Dictionary of features
    """
    height, width = char_img.shape
    
    # Split the image into four quarters
    h_mid = height // 2
    w_mid = width // 2
    
    q1 = char_img[:h_mid, :w_mid]  # Top-left
    q2 = char_img[:h_mid, w_mid:]  # Top-right
    q3 = char_img[h_mid:, :w_mid]  # Bottom-left
    q4 = char_img[h_mid:, w_mid:]  # Bottom-right
    
    # Calculate weight (mass) of each quarter
    weight_q1 = np.sum(q1)
    weight_q2 = np.sum(q2)
    weight_q3 = np.sum(q3)
    weight_q4 = np.sum(q4)
    total_weight = weight_q1 + weight_q2 + weight_q3 + weight_q4
    
    # Calculate center of gravity
    y_indices, x_indices = np.indices(char_img.shape)
    cog_x = np.sum(x_indices * char_img) / total_weight if total_weight > 0 else width / 2
    cog_y = np.sum(y_indices * char_img) / total_weight if total_weight > 0 else height / 2
    
    # Normalize center of gravity
    norm_cog_x = cog_x / width
    norm_cog_y = cog_y / height
    
    # Calculate axial moments of inertia
    moment_x = np.sum(((y_indices - cog_y) ** 2) * char_img)
    moment_y = np.sum(((x_indices - cog_x) ** 2) * char_img)
    
    # Normalize moments
    norm_moment_x = moment_x / (total_weight * height ** 2) if total_weight > 0 else 0
    norm_moment_y = moment_y / (total_weight * width ** 2) if total_weight > 0 else 0
    
    # Return dictionary of features
    return {
        'Weight_Q1': weight_q1,
        'Weight_Q2': weight_q2,
        'Weight_Q3': weight_q3,
        'Weight_Q4': weight_q4,
        'CoG_X': cog_x,
        'CoG_Y': cog_y,
        'NormCoG_X': norm_cog_x,
        'NormCoG_Y': norm_cog_y,
        'MomentOfInertia_X': moment_x,
        'MomentOfInertia_Y': moment_y,
        'NormMomentOfInertia_X': norm_moment_x,
        'NormMomentOfInertia_Y': norm_moment_y
    }

def compute_similarity(char_features, reference_features):
    """
    Compute similarity between character and reference based on Euclidean distance.
    
    Args:
        char_features (dict): Features of the character to recognize
        reference_features (pd.Series): Features of the reference character
        
    Returns:
        float: Similarity measure (1 for exact match, decreasing for less similar)
    """
    # Define the features to use for comparison
    feature_keys = [
        'NormCoG_X', 'NormCoG_Y', 
        'NormMomentOfInertia_X', 'NormMomentOfInertia_Y'
    ]
    
    # Extract feature vectors
    char_vector = np.array([char_features[key] for key in feature_keys])
    ref_vector = np.array([reference_features[key] for key in feature_keys])
    
    # Calculate Euclidean distance
    distance = euclidean(char_vector, ref_vector)
    
    # Convert distance to similarity (1 for exact match, decreasing for less similar)
    # Using exponential decay to ensure positive values
    similarity = np.exp(-distance)
    
    return similarity

def recognize_character(char_img, reference_features):
    """
    Recognize a character by comparing with reference features.
    
    Args:
        char_img (numpy.ndarray): Binary character image
        reference_features (pd.DataFrame): DataFrame with reference features
        
    Returns:
        list: List of tuples (char, similarity) sorted by similarity
    """
    # Calculate features for the character
    char_features = calculate_character_features(char_img)
    
    # Calculate similarity with each reference character
    similarities = []
    
    for _, reference in reference_features.iterrows():
        char = reference['Character']
        unicode_val = reference['Unicode']
        similarity = compute_similarity(char_features, reference)
        similarities.append((char, similarity, unicode_val))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return list of tuples (char, similarity)
    return [(char, similarity) for char, similarity, _ in similarities]

def recognize_characters(characters, reference_features):
    """
    Recognize all segmented characters.
    
    Args:
        characters (list): List of character images
        reference_features (pd.DataFrame): DataFrame with reference features
        
    Returns:
        list: List of recognition results, where each result is a list of tuples (char, similarity)
    """
    logger.info("Recognizing characters...")
    
    results = []
    
    for i, char_img in enumerate(characters):
        logger.info(f"Recognizing character {i+1}/{len(characters)}")
        result = recognize_character(char_img, reference_features)
        results.append(result)
    
    return results

def save_recognition_results(results):
    """
    Save recognition results to a file.
    
    Args:
        results (list): List of recognition results
    """
    logger.info("Saving recognition results...")
    
    # Create output file
    output_path = os.path.join(RESULTS_DIR, "recognition_results.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results):
            f.write(f"{i+1}: {result}\n")
    
    logger.info(f"Results saved to {output_path}")

def extract_best_hypothesis(results):
    """
    Extract best hypothesis (highest similarity) for each character.
    
    Args:
        results (list): List of recognition results
        
    Returns:
        str: String of best hypotheses
    """
    return ''.join([result[0][0] for result in results])

def count_correct_recognitions(hypothesis, reference):
    """
    Count correct recognitions by comparing with reference.
    
    Args:
        hypothesis (str): String of best hypotheses
        reference (str): Reference string
        
    Returns:
        tuple: Number of correct recognitions, total characters, percentage
    """
    # Ensure the strings have the same length
    min_len = min(len(hypothesis), len(reference))
    
    # Count correct recognitions
    correct = sum(1 for i in range(min_len) if hypothesis[i] == reference[i])
    
    # Calculate percentage
    percentage = 100 * correct / len(reference) if len(reference) > 0 else 0
    
    return correct, len(reference), percentage

def generate_different_size_image(reference_string, font_size_change):
    """
    Generate an image of the reference string with a different font size.
    
    Args:
        reference_string (str): Reference string
        font_size_change (int): Change in font size (positive or negative)
        
    Returns:
        numpy.ndarray: Generated image
    """
    logger.info(f"Generating image with font size change of {font_size_change}")
    
    # Base font size from lab5
    base_font_size = 52
    new_font_size = base_font_size + font_size_change
    
    # Create a larger image to ensure text fits
    img = Image.new('L', (800, 100), color=255)
    
    try:
        font = ImageFont.truetype(FONT_PATH, new_font_size)
    except Exception as e:
        logger.error(f"Failed to load font: {e}")
        return None
    
    draw = ImageDraw.Draw(img)
    
    # Draw the text
    # Hebrew is right-to-left, so position accordingly
    text_width = draw.textlength(reference_string, font=font)
    position = (img.width - text_width - 10, 10)
    draw.text(position, reference_string, font=font, fill=0)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Save generated image
    output_path = os.path.join(RESULTS_DIR, f"generated_size_{new_font_size}.png")
    cv2.imwrite(output_path, img_array)
    
    return img_array

def segment_characters_from_image(image):
    """
    Segment characters from an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        list: List of segmented character images
    """
    logger.info("Segmenting characters from generated image")
    
    # Binarize image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Invert so text is white, background is black
    binary_inv = cv2.bitwise_not(binary)
    
    # Calculate profiles
    horizontal_profile = np.sum(binary_inv, axis=1)
    vertical_profile = np.sum(binary_inv, axis=0)
    
    # Identify text lines
    line_threshold = max(horizontal_profile) * 0.05
    rows_with_text = np.where(horizontal_profile > line_threshold)[0]
    
    if len(rows_with_text) == 0:
        logger.warning("No text lines detected")
        return []
    
    # Group consecutive rows to identify lines
    line_ranges = []
    line_start = rows_with_text[0]
    
    for i in range(1, len(rows_with_text)):
        if rows_with_text[i] - rows_with_text[i-1] > 1:  # Gap between text lines
            line_end = rows_with_text[i-1]
            line_ranges.append((line_start, line_end))
            line_start = rows_with_text[i]
    
    # Add the last line
    line_ranges.append((line_start, rows_with_text[-1]))
    
    # For each line, identify characters using vertical profile
    bounding_boxes = []
    
    for line_start, line_end in line_ranges:
        # Extract vertical profile for this line only
        line_image = binary_inv[line_start:line_end+1, :]
        line_vertical_profile = np.sum(line_image, axis=0)
        
        # Threshold for character detection
        char_threshold = max(line_vertical_profile) * 0.05
        
        # Find columns where profile exceeds threshold
        cols_with_text = np.where(line_vertical_profile > char_threshold)[0]
        
        if len(cols_with_text) == 0:
            continue
        
        # Group consecutive columns to identify characters
        char_start = cols_with_text[0]
        
        for i in range(1, len(cols_with_text)):
            if cols_with_text[i] - cols_with_text[i-1] > 1:  # Gap between characters
                char_end = cols_with_text[i-1]
                
                # Create bounding box (x, y, width, height)
                x = char_start
                y = line_start
                w = char_end - char_start + 1
                h = line_end - line_start + 1
                
                bounding_boxes.append((x, y, w, h))
                
                char_start = cols_with_text[i]
        
        # Add the last character in the line
        char_end = cols_with_text[-1]
        x = char_start
        y = line_start
        w = char_end - char_start + 1
        h = line_end - line_start + 1
        
        bounding_boxes.append((x, y, w, h))
    
    # Sort bounding boxes from right to left (for Hebrew)
    bounding_boxes.sort(key=lambda box: -box[0])
    
    # Extract character images
    characters = []
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        char_img = binary_inv[y:y+h, x:x+w]
        
        # Normalize to 0-1
        char_img = char_img / 255
        
        characters.append(char_img)
        
        # Save character image
        output_path = os.path.join(RESULTS_DIR, f"gen_char_{i+1}.png")
        cv2.imwrite(output_path, char_img * 255)
    
    logger.info(f"Segmented {len(characters)} characters from generated image")
    return characters

def main():
    """Main function."""
    logger.info("Starting recognition process")
    
    # Load reference features from lab5
    reference_features = load_features()
    
    # Load segmented characters from lab6
    segmented_chars = load_segmented_characters()
    
    # Задание 1-2: Recognize characters based on features
    recognition_results = recognize_characters(segmented_chars, reference_features)
    
    # Задание 3: Save recognition results
    save_recognition_results(recognition_results)
    
    # Задание 4: Extract best hypothesis
    best_hypothesis = extract_best_hypothesis(recognition_results)
    logger.info(f"Best hypothesis: {best_hypothesis}")
    
    # Задание 5: Count correct recognitions
    # The original phrase is not known, so we need to get it
    # Since Hebrew is right-to-left, we need to manually input the reference string
    # This would be the phrase from lab6/phrase.bmp
    reference_string = "מילים עבריות לדוגמה"  # Replace with actual phrase
    
    correct, total, percentage = count_correct_recognitions(best_hypothesis, reference_string)
    logger.info(f"Correct recognitions: {correct}/{total} ({percentage:.2f}%)")
    
    # Save recognition accuracy
    with open(os.path.join(RESULTS_DIR, "recognition_accuracy.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Reference string: {reference_string}\n")
        f.write(f"Recognized string: {best_hypothesis}\n")
        f.write(f"Correct recognitions: {correct}/{total} ({percentage:.2f}%)\n")
    
    # Задание 6: Generate a different size image and recognize it
    font_size_change = 4  # Increase font size by 4 points
    different_size_image = generate_different_size_image(reference_string, font_size_change)
    
    if different_size_image is not None:
        # Segment characters from the generated image
        different_size_chars = segment_characters_from_image(different_size_image)
        
        # Recognize the segmented characters
        different_size_results = recognize_characters(different_size_chars, reference_features)
        
        # Save recognition results
        output_path = os.path.join(RESULTS_DIR, "different_size_results.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, result in enumerate(different_size_results):
                f.write(f"{i+1}: {result}\n")
        
        # Extract best hypothesis
        different_size_hypothesis = extract_best_hypothesis(different_size_results)
        
        # Count correct recognitions
        correct_diff, total_diff, percentage_diff = count_correct_recognitions(
            different_size_hypothesis, reference_string
        )
        
        logger.info(f"Different size recognition: {correct_diff}/{total_diff} ({percentage_diff:.2f}%)")
        
        # Save different size accuracy
        with open(os.path.join(RESULTS_DIR, "different_size_accuracy.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Reference string: {reference_string}\n")
            f.write(f"Font size change: {font_size_change}\n")
            f.write(f"Recognized string: {different_size_hypothesis}\n")
            f.write(f"Correct recognitions: {correct_diff}/{total_diff} ({percentage_diff:.2f}%)\n")
        
        # Compare the two recognitions
        with open(os.path.join(RESULTS_DIR, "comparison.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Original recognition: {percentage:.2f}%\n")
            f.write(f"Different size recognition: {percentage_diff:.2f}%\n")
            f.write(f"Difference: {percentage_diff - percentage:.2f}%\n")
    
    logger.info("Recognition process completed")

if __name__ == "__main__":
    main()
