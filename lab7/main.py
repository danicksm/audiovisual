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
import seaborn as sns

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

# New reference string with symbols from lab5
# Составляем фразу из всех доступных символов иврита (для лучшего распознавания)
REFERENCE_STRING = "אבגדהוזחטיכלמנסעפצקרשת"

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
            
            # Normalize image size for better comparison
            binary = normalize_image_size(binary)
            
            characters.append(binary)
            logger.debug(f"Loaded character from {filename}")
    
    logger.info(f"Loaded {len(characters)} segmented characters")
    return characters

def normalize_image_size(img, target_size=(32, 32)):
    """
    Normalize image size to a standard size.
    
    Args:
        img (numpy.ndarray): Input image
        target_size (tuple): Target size (width, height)
        
    Returns:
        numpy.ndarray: Normalized image
    """
    # Add padding to make the image square
    height, width = img.shape
    max_dim = max(height, width)
    
    # Create a square black image
    square_img = np.zeros((max_dim, max_dim), dtype=img.dtype)
    
    # Calculate offsets to center the original image
    y_offset = (max_dim - height) // 2
    x_offset = (max_dim - width) // 2
    
    # Copy the original image to the center of the square image
    square_img[y_offset:y_offset + height, x_offset:x_offset + width] = img
    
    # Resize to target size
    normalized = cv2.resize(square_img, target_size, interpolation=cv2.INTER_AREA)
    
    return normalized

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
    
    # Calculate specific weight (normalized by area)
    area_q1 = q1.size
    area_q2 = q2.size
    area_q3 = q3.size
    area_q4 = q4.size
    
    specific_weight_q1 = weight_q1 / area_q1 if area_q1 > 0 else 0
    specific_weight_q2 = weight_q2 / area_q2 if area_q2 > 0 else 0
    specific_weight_q3 = weight_q3 / area_q3 if area_q3 > 0 else 0
    specific_weight_q4 = weight_q4 / area_q4 if area_q4 > 0 else 0
    
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
    
    # Calculate profiles
    x_profile = np.sum(char_img, axis=0)  # Vertical profile
    y_profile = np.sum(char_img, axis=1)  # Horizontal profile
    
    # Normalize profiles
    norm_x_profile = x_profile / height if np.max(x_profile) > 0 else x_profile
    norm_y_profile = y_profile / width if np.max(y_profile) > 0 else y_profile
    
    # Return dictionary of features
    return {
        'TotalWeight': total_weight,
        'Weight_Q1': weight_q1,
        'Weight_Q2': weight_q2,
        'Weight_Q3': weight_q3,
        'Weight_Q4': weight_q4,
        'SpecificWeight_Q1': specific_weight_q1,
        'SpecificWeight_Q2': specific_weight_q2,
        'SpecificWeight_Q3': specific_weight_q3,
        'SpecificWeight_Q4': specific_weight_q4,
        'CoG_X': cog_x,
        'CoG_Y': cog_y,
        'NormCoG_X': norm_cog_x,
        'NormCoG_Y': norm_cog_y,
        'MomentOfInertia_X': moment_x,
        'MomentOfInertia_Y': moment_y,
        'NormMomentOfInertia_X': norm_moment_x,
        'NormMomentOfInertia_Y': norm_moment_y,
        'XProfile': norm_x_profile,
        'YProfile': norm_y_profile
    }

def compute_distance(char_features, reference_features):
    """
    Compute Euclidean distance between character and reference based on features.
    
    Args:
        char_features (dict): Features of the character
        reference_features (pd.Series): Features of the reference character
        
    Returns:
        float: Euclidean distance between the characters
    """
    # Define the features to use for comparison
    scalar_feature_keys = [
        'Weight_Q1', 'Weight_Q2', 'Weight_Q3', 'Weight_Q4',  # Масса символа по четвертям
        'SpecificWeight_Q1', 'SpecificWeight_Q2', 'SpecificWeight_Q3', 'SpecificWeight_Q4',
        'NormCoG_X', 'NormCoG_Y',  # Нормализованные координаты центра тяжести
        'NormMomentOfInertia_X', 'NormMomentOfInertia_Y'  # Нормализованные осевые моменты инерции
    ]
    
    # Define weights for each feature
    weights = {
        'Weight_Q1': 0.1, 'Weight_Q2': 0.1, 'Weight_Q3': 0.1, 'Weight_Q4': 0.1,
        'SpecificWeight_Q1': 0.05, 'SpecificWeight_Q2': 0.05, 'SpecificWeight_Q3': 0.05, 'SpecificWeight_Q4': 0.05,
        'NormCoG_X': 0.15, 'NormCoG_Y': 0.15,
        'NormMomentOfInertia_X': 0.15, 'NormMomentOfInertia_Y': 0.15
    }
    
    # Calculate weighted squared differences for scalar features
    weighted_squared_diff = 0
    for key in scalar_feature_keys:
        # Skip if key is missing in either features dictionary
        if key not in char_features or key not in reference_features:
            continue
            
        # Normalize feature values to [0, 1] range if needed
        char_value = float(char_features[key])
        ref_value = float(reference_features[key])
        
        # Add weighted squared difference
        weighted_squared_diff += weights[key] * (char_value - ref_value)**2
    
    # Calculate weighted Euclidean distance
    distance = np.sqrt(weighted_squared_diff)
    
    return distance

def compute_similarity(char_features, reference_features):
    """
    Compute similarity between character and reference based on Euclidean distance.
    
    Args:
        char_features (dict): Features of the character to recognize
        reference_features (pd.Series): Features of the reference character
        
    Returns:
        float: Similarity measure (1 for exact match, decreasing for less similar)
    """
    # Calculate distance
    distance = compute_distance(char_features, reference_features)
    
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

def save_recognition_results(results, filename="recognition_results.txt"):
    """
    Save recognition results to a file.
    
    Args:
        results (list): List of recognition results
        filename (str): Output filename
    """
    logger.info(f"Saving recognition results to {filename}...")
    
    # Create output file
    output_path = os.path.join(RESULTS_DIR, filename)
    
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

def generate_different_size_image(reference_string, font_size):
    """
    Generate an image of the reference string with the specified font size.
    
    Args:
        reference_string (str): Reference string
        font_size (int): Font size to use
        
    Returns:
        numpy.ndarray: Generated image
    """
    logger.info(f"Generating image with font size {font_size}")
    
    # Create a larger image to ensure text fits
    img = Image.new('L', (1200, 100), color=255)
    
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
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
    output_path = os.path.join(RESULTS_DIR, f"generated_size_{font_size}.png")
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
        
        # Normalize image size
        char_img = normalize_image_size(char_img)
        
        characters.append(char_img)
        
        # Save character image
        output_path = os.path.join(RESULTS_DIR, f"gen_char_{i+1}.png")
        cv2.imwrite(output_path, char_img * 255)
    
    logger.info(f"Segmented {len(characters)} characters from generated image")
    return characters

def load_phrase_image():
    """
    Load and preprocess the phrase image from lab6.
    
    Returns:
        numpy.ndarray: Preprocessed image
    """
    logger.info(f"Loading phrase image from {PHRASE_BMP}")
    
    # Read the image
    image = cv2.imread(PHRASE_BMP, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        logger.error(f"Failed to load image {PHRASE_BMP}")
        return None
    
    # Save a copy to results directory
    output_path = os.path.join(RESULTS_DIR, "original_phrase.png")
    cv2.imwrite(output_path, image)
    
    return image

def run_recognition_experiment(reference_features, image_source, font_size=None, source_name=None):
    """
    Run a recognition experiment on the given image source.
    
    Args:
        reference_features (pd.DataFrame): Reference features
        image_source: Either a numpy.ndarray image or a list of character images
        font_size (int, optional): Font size for the experiment (for logging)
        source_name (str, optional): Name of the source (for logging)
        
    Returns:
        tuple: Recognition results, best hypothesis, accuracy metrics
    """
    if source_name:
        logger.info(f"Running recognition experiment on {source_name}")
    
    # Segment characters if image_source is an image
    if isinstance(image_source, np.ndarray):
        characters = segment_characters_from_image(image_source)
    else:
        characters = image_source
    
    # Skip if no characters found
    if not characters:
        logger.warning("No characters found in the image")
        return None, "", (0, 0, 0)
    
    # Recognize characters
    recognition_results = recognize_characters(characters, reference_features)
    
    # Generate output filename
    if font_size:
        results_filename = f"recognition_results_size_{font_size}.txt"
    elif source_name:
        results_filename = f"recognition_results_{source_name}.txt"
    else:
        results_filename = "recognition_results.txt"
    
    # Save recognition results
    save_recognition_results(recognition_results, results_filename)
    
    # Extract best hypothesis
    best_hypothesis = extract_best_hypothesis(recognition_results)
    logger.info(f"Best hypothesis: {best_hypothesis}")
    
    # Count correct recognitions
    correct, total, percentage = count_correct_recognitions(best_hypothesis, REFERENCE_STRING)
    logger.info(f"Correct recognitions: {correct}/{total} ({percentage:.2f}%)")
    
    return recognition_results, best_hypothesis, (correct, total, percentage)

def save_experiment_results(experiment_name, hypothesis, accuracy, font_size=None):
    """
    Save experiment results to a file.
    
    Args:
        experiment_name (str): Name of the experiment
        hypothesis (str): Recognized string
        accuracy (tuple): Accuracy metrics (correct, total, percentage)
        font_size (int, optional): Font size used
    """
    correct, total, percentage = accuracy
    
    # Generate output filename
    if font_size:
        filename = f"{experiment_name}_size_{font_size}.txt"
    else:
        filename = f"{experiment_name}.txt"
    
    output_path = os.path.join(RESULTS_DIR, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Reference string: {REFERENCE_STRING}\n")
        if font_size:
            f.write(f"Font size: {font_size}\n")
        f.write(f"Recognized string: {hypothesis}\n")
        f.write(f"Correct recognitions: {correct}/{total} ({percentage:.2f}%)\n")
    
    logger.info(f"Experiment results saved to {output_path}")

def compare_experiments(experiments):
    """
    Compare the results of multiple experiments.
    
    Args:
        experiments (dict): Dictionary mapping experiment names to accuracy metrics
    """
    output_path = os.path.join(RESULTS_DIR, "comparison.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Comparison of recognition experiments:\n\n")
        
        # Write individual experiment results
        for name, (_, _, percentage) in experiments.items():
            f.write(f"{name}: {percentage:.2f}%\n")
        
        f.write("\nDifferences between experiments:\n\n")
        
        # Calculate differences between experiments
        experiment_names = list(experiments.keys())
        for i in range(len(experiment_names)):
            for j in range(i+1, len(experiment_names)):
                name_i = experiment_names[i]
                name_j = experiment_names[j]
                
                percentage_i = experiments[name_i][2]
                percentage_j = experiments[name_j][2]
                
                difference = percentage_j - percentage_i
                
                f.write(f"{name_j} - {name_i}: {difference:.2f}%\n")
    
    logger.info(f"Experiment comparison saved to {output_path}")

def visualize_experiments(experiments):
    """
    Create a bar chart comparing the results of multiple experiments.
    
    Args:
        experiments (dict): Dictionary mapping experiment names to accuracy metrics
    """
    # Extract experiment names and percentages
    names = []
    percentages = []
    
    for name, (_, _, percentage) in experiments.items():
        names.append(name)
        percentages.append(percentage)
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(names, percentages)
    plt.xlabel('Experiment')
    plt.ylabel('Recognition Accuracy (%)')
    plt.title('Comparison of Recognition Experiments')
    plt.ylim(0, 100)  # Set y-axis limits from 0 to 100%
    
    # Add value labels on top of each bar
    for i, v in enumerate(percentages):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    # Save the chart
    output_path = os.path.join(RESULTS_DIR, "experiment_comparison.png")
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Experiment visualization saved to {output_path}")

def create_distance_matrix(reference_features):
    """
    Create a matrix of Euclidean distances between all reference symbols.
    
    Args:
        reference_features (pd.DataFrame): DataFrame with reference features
        
    Returns:
        tuple: Distance matrix, list of character labels
    """
    logger.info("Creating distance matrix between reference symbols")
    
    # Get the number of reference symbols
    n_symbols = len(reference_features)
    
    # Initialize distance matrix
    distance_matrix = np.zeros((n_symbols, n_symbols))
    
    # Get list of characters
    char_labels = reference_features['Character'].tolist()
    
    # Calculate distances between all pairs of symbols
    for i in range(n_symbols):
        for j in range(n_symbols):
            char_i = reference_features.iloc[i]
            char_j = reference_features.iloc[j]
            
            # Calculate distance using our compute_distance function
            distance = compute_distance(char_i, char_j)
            
            # Store in matrix
            distance_matrix[i, j] = distance
    
    # Save distance matrix to file
    np.savetxt(os.path.join(RESULTS_DIR, "distance_matrix.csv"), distance_matrix, delimiter=';')
    
    return distance_matrix, char_labels

def visualize_distance_matrix(distance_matrix, char_labels):
    """
    Visualize the distance matrix as a heatmap.
    
    Args:
        distance_matrix (np.ndarray): Matrix of distances between symbols
        char_labels (list): List of character labels
    """
    logger.info("Visualizing distance matrix")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    ax = sns.heatmap(distance_matrix, cmap='viridis_r', 
                    xticklabels=char_labels, yticklabels=char_labels,
                    cbar_kws={'label': 'Евклидово расстояние'})
    
    # Set title and labels
    plt.title('Матрица евклидовых расстояний между эталонами')
    
    # Save the figure
    output_path = os.path.join(RESULTS_DIR, "distance_matrix.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Distance matrix visualization saved to {output_path}")

def main():
    """Main function."""
    logger.info("Starting recognition process")
    
    # Load reference features from lab5
    reference_features = load_features()

    # Create and visualize distance matrix between reference symbols
    distance_matrix, char_labels = create_distance_matrix(reference_features)
    visualize_distance_matrix(distance_matrix, char_labels)
    
    # Dictionary to store experiment results
    experiments = {}
    
    # 1. Experiment with original segmented characters from lab6
    logger.info("Experiment 1: Recognizing segmented characters from lab6")
    segmented_chars = load_segmented_characters()
    original_results, original_hypothesis, original_accuracy = run_recognition_experiment(
        reference_features, segmented_chars, source_name="original"
    )
    save_experiment_results("recognition_accuracy", original_hypothesis, original_accuracy)
    experiments["Original Lab6 Chars"] = original_accuracy
    
    # 2. Experiment with phrase.bmp from lab6
    logger.info("Experiment 2: Recognizing phrase.bmp from lab6")
    phrase_image = load_phrase_image()
    if phrase_image is not None:
        phrase_results, phrase_hypothesis, phrase_accuracy = run_recognition_experiment(
            reference_features, phrase_image, source_name="phrase_bmp"
        )
        save_experiment_results("phrase_recognition", phrase_hypothesis, phrase_accuracy)
        experiments["Phrase.bmp"] = phrase_accuracy
    
    # 3. Experiment with different font sizes using new reference string with all Hebrew symbols
    # Base font size from lab5
    base_font_size = 52
    
    # Smaller font size
    smaller_font_size = base_font_size - 8
    logger.info(f"Experiment 3: Recognizing text with smaller font size ({smaller_font_size})")
    smaller_image = generate_different_size_image(REFERENCE_STRING, smaller_font_size)
    if smaller_image is not None:
        smaller_results, smaller_hypothesis, smaller_accuracy = run_recognition_experiment(
            reference_features, smaller_image, smaller_font_size, "smaller_font"
        )
        save_experiment_results("smaller_font_accuracy", smaller_hypothesis, smaller_accuracy, smaller_font_size)
        experiments[f"Smaller Font ({smaller_font_size})"] = smaller_accuracy
    
    # Original font size (for comparison)
    logger.info(f"Experiment 4: Recognizing text with original font size ({base_font_size})")
    original_size_image = generate_different_size_image(REFERENCE_STRING, base_font_size)
    if original_size_image is not None:
        original_size_results, original_size_hypothesis, original_size_accuracy = run_recognition_experiment(
            reference_features, original_size_image, base_font_size, "original_font"
        )
        save_experiment_results("original_font_accuracy", original_size_hypothesis, original_size_accuracy, base_font_size)
        experiments[f"Original Font ({base_font_size})"] = original_size_accuracy
    
    # Larger font size
    larger_font_size = base_font_size + 8
    logger.info(f"Experiment 5: Recognizing text with larger font size ({larger_font_size})")
    larger_image = generate_different_size_image(REFERENCE_STRING, larger_font_size)
    if larger_image is not None:
        larger_results, larger_hypothesis, larger_accuracy = run_recognition_experiment(
            reference_features, larger_image, larger_font_size, "larger_font"
        )
        save_experiment_results("larger_font_accuracy", larger_hypothesis, larger_accuracy, larger_font_size)
        experiments[f"Larger Font ({larger_font_size})"] = larger_accuracy
    
    # Very large font size for better recognition
    very_large_font_size = base_font_size + 20
    logger.info(f"Experiment 6: Recognizing text with very large font size ({very_large_font_size})")
    very_large_image = generate_different_size_image(REFERENCE_STRING, very_large_font_size)
    if very_large_image is not None:
        very_large_results, very_large_hypothesis, very_large_accuracy = run_recognition_experiment(
            reference_features, very_large_image, very_large_font_size, "very_large_font"
        )
        save_experiment_results("very_large_font_accuracy", very_large_hypothesis, very_large_accuracy, very_large_font_size)
        experiments[f"Very Large Font ({very_large_font_size})"] = very_large_accuracy
    
    # Compare and visualize experiment results
    compare_experiments(experiments)
    visualize_experiments(experiments)
    
    logger.info("Recognition process completed")

if __name__ == "__main__":
    main()
