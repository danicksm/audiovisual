#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FONT_PATH = "Hebrew.ttf"
FONT_SIZE = 52
OUTPUT_DIR_ORIGINALS = "symbols/originals"
OUTPUT_DIR_PROFILES = "symbols/profiles"
RESULTS_DIR = "results"
CSV_OUTPUT = os.path.join(RESULTS_DIR, "features.csv")

# Ensure directories exist
os.makedirs(OUTPUT_DIR_ORIGINALS, exist_ok=True)
os.makedirs(OUTPUT_DIR_PROFILES, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hebrew alphabet (first 22 letters)
hebrew_chars = [chr(i) for i in range(0x05D0, 0x05EA + 1)]

def crop_white_borders(img):
    """Crop white borders from an image."""
    # Convert to numpy array and get non-white pixel coordinates
    img_array = np.array(img)
    if len(img_array.shape) == 3:  # RGB image
        # Convert to grayscale for processing
        gray_array = np.mean(img_array, axis=2)
    else:  # Already grayscale
        gray_array = img_array
    
    # Find the bounding box of non-white pixels
    non_white = np.where(gray_array < 245)  # Threshold to identify non-white pixels
    if len(non_white[0]) == 0:  # If image is all white
        return img
    
    # Get the bounding box
    min_y, max_y = np.min(non_white[0]), np.max(non_white[0])
    min_x, max_x = np.min(non_white[1]), np.max(non_white[1])
    
    # Add a small padding (5 pixels)
    padding = 5
    min_y = max(0, min_y - padding)
    min_x = max(0, min_x - padding)
    max_y = min(img_array.shape[0], max_y + padding)
    max_x = min(img_array.shape[1], max_x + padding)
    
    # Crop and return
    return img.crop((min_x, min_y, max_x, max_y))

def generate_character_images():
    """Generate reference images for each Hebrew character."""
    logger.info("Generating character images")
    
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception as e:
        logger.error(f"Failed to load font: {e}")
        return {}
    
    char_images = {}
    
    for char in hebrew_chars:
        # Create a larger image to ensure character fits
        img = Image.new('L', (FONT_SIZE * 2, FONT_SIZE * 2), color=255)
        draw = ImageDraw.Draw(img)
        
        # Calculate position to center the character
        # Hebrew is right-to-left, so we position accordingly
        text_width = draw.textlength(char, font=font)
        text_height = FONT_SIZE  # Approximate
        position = ((img.width - text_width) // 2, (img.height - text_height) // 2)
        
        # Draw the character
        draw.text(position, char, font=font, fill=0)
        
        # Crop white borders
        img = crop_white_borders(img)
        
        # Save the image
        filename = f"{ord(char):d}.png"
        filepath = os.path.join(OUTPUT_DIR_ORIGINALS, filename)
        img.save(filepath)
        
        char_images[char] = {
            'image': img,
            'filename': filename,
            'filepath': filepath
        }
        
        logger.info(f"Generated image for character {char} (Unicode: {ord(char)})")
    
    return char_images

def calculate_features(char_images):
    """Calculate features for each character image."""
    logger.info("Calculating features for all characters")
    
    features = []
    
    for char, data in char_images.items():
        img = data['image']
        img_array = np.array(img)
        
        # Ensure the image is binary (0 and 255)
        # For simplicity, we'll threshold: < 128 -> 0 (black), >= 128 -> 1 (white)
        binary = (img_array >= 128).astype(np.uint8)
        
        # Invert so that character pixels are 1 and background is 0
        binary = 1 - binary
        
        height, width = binary.shape
        
        # Split the image into four quarters
        h_mid = height // 2
        w_mid = width // 2
        
        q1 = binary[:h_mid, :w_mid]  # Top-left
        q2 = binary[:h_mid, w_mid:]  # Top-right
        q3 = binary[h_mid:, :w_mid]  # Bottom-left
        q4 = binary[h_mid:, w_mid:]  # Bottom-right
        
        # a) Weight (mass of black) of each quarter
        weight_q1 = np.sum(q1)
        weight_q2 = np.sum(q2)
        weight_q3 = np.sum(q3)
        weight_q4 = np.sum(q4)
        total_weight = weight_q1 + weight_q2 + weight_q3 + weight_q4
        
        # b) Specific weight (normalized to quarter area)
        area_q1 = q1.size
        area_q2 = q2.size
        area_q3 = q3.size
        area_q4 = q4.size
        
        specific_weight_q1 = weight_q1 / area_q1 if area_q1 > 0 else 0
        specific_weight_q2 = weight_q2 / area_q2 if area_q2 > 0 else 0
        specific_weight_q3 = weight_q3 / area_q3 if area_q3 > 0 else 0
        specific_weight_q4 = weight_q4 / area_q4 if area_q4 > 0 else 0
        
        # c) Center of gravity coordinates
        y_indices, x_indices = np.indices(binary.shape)
        cog_x = np.sum(x_indices * binary) / total_weight if total_weight > 0 else width / 2
        cog_y = np.sum(y_indices * binary) / total_weight if total_weight > 0 else height / 2
        
        # d) Normalized center of gravity coordinates
        norm_cog_x = cog_x / width
        norm_cog_y = cog_y / height
        
        # e) Axial moments of inertia (horizontal and vertical)
        # Horizontal moment (around x-axis)
        moment_x = np.sum(((y_indices - cog_y) ** 2) * binary)
        # Vertical moment (around y-axis)
        moment_y = np.sum(((x_indices - cog_x) ** 2) * binary)
        
        # f) Normalized axial moments of inertia
        # Normalize by dividing by the product of total mass and square of dimension
        norm_moment_x = moment_x / (total_weight * height ** 2) if total_weight > 0 else 0
        norm_moment_y = moment_y / (total_weight * width ** 2) if total_weight > 0 else 0
        
        # g) X and Y profiles
        x_profile = np.sum(binary, axis=0)  # Sum along rows (vertical profile)
        y_profile = np.sum(binary, axis=1)  # Sum along columns (horizontal profile)
        
        # Save profiles as PNG
        save_profile(x_profile, 'X', char, data['filename'])
        save_profile(y_profile, 'Y', char, data['filename'])
        
        # Store all features
        char_features = {
            'Character': char,
            'Unicode': ord(char),
            'Filename': data['filename'],
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
            'NormMomentOfInertia_Y': norm_moment_y
        }
        
        features.append(char_features)
        logger.info(f"Calculated features for character {char}")
    
    return pd.DataFrame(features)

def save_profile(profile, axis, char, filename):
    """Save profile as a bar chart."""
    plt.figure(figsize=(10, 6))
    
    # Create bar chart with integer labels
    x = np.arange(len(profile))
    plt.bar(x, profile)
    
    # Set integer labels on axes
    plt.xticks(np.arange(0, len(profile), step=max(1, len(profile)//10)))
    plt.yticks(np.arange(0, max(profile)+1, step=max(1, int(max(profile)/10))))
    
    if axis == 'X':
        plt.xlabel('Горизонтальная позиция (пиксели)')
        plt.ylabel('Количество черных пикселей')
        plt.title(f'Профиль X для символа {char}')
    else:  # Y profile
        plt.xlabel('Вертикальная позиция (пиксели)')
        plt.ylabel('Количество черных пикселей')
        plt.title(f'Профиль Y для символа {char}')
    
    # Save figure
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR_PROFILES, f"{base_filename}_{axis}_profile.png")
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    # Also save a copy to results for the report
    results_path = os.path.join(RESULTS_DIR, f"{base_filename}_{axis}_profile.png")
    plt.figure(figsize=(10, 6))
    plt.bar(x, profile)
    plt.xticks(np.arange(0, len(profile), step=max(1, len(profile)//10)))
    plt.yticks(np.arange(0, max(profile)+1, step=max(1, int(max(profile)/10))))
    
    if axis == 'X':
        plt.xlabel('Горизонтальная позиция (пиксели)')
        plt.ylabel('Количество черных пикселей')
        plt.title(f'Профиль X для символа {char}')
    else:  # Y profile
        plt.xlabel('Вертикальная позиция (пиксели)')
        plt.ylabel('Количество черных пикселей')
        plt.title(f'Профиль Y для символа {char}')
    
    plt.savefig(results_path, dpi=100)
    plt.close()

def save_features_csv(features_df):
    """Save features to CSV file."""
    features_df.to_csv(CSV_OUTPUT, sep=';', index=False)
    logger.info(f"Features saved to {CSV_OUTPUT}")

def main():
    """Main function to execute all tasks."""
    logger.info("Starting Lab 5 - Character Features Extraction")
    
    # 1. Generate reference character images
    char_images = generate_character_images()
    
    if not char_images:
        logger.error("Failed to generate character images. Exiting.")
        return
    
    # 2. Calculate features for each image
    features_df = calculate_features(char_images)
    
    # 3. Save features to CSV
    save_features_csv(features_df)
    
    # Generate a sample image with all characters for the report
    generate_sample_image(char_images)
    
    logger.info("Lab 5 completed successfully!")

def generate_sample_image(char_images):
    """Generate a sample image with all characters for the report."""
    # Calculate grid dimensions
    n_chars = len(char_images)
    cols = min(5, n_chars)
    rows = (n_chars + cols - 1) // cols
    
    # Create a blank image
    cell_size = 100
    img = Image.new('RGB', (cols * cell_size, rows * cell_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Place each character image in the grid
    for i, (char, data) in enumerate(char_images.items()):
        row = i // cols
        col = i % cols
        
        char_img = data['image']
        
        # Calculate position to center the character image in its cell
        x = col * cell_size + (cell_size - char_img.width) // 2
        y = row * cell_size + (cell_size - char_img.height) // 2
        
        # Paste the character image
        if char_img.mode == 'L':
            # Convert grayscale to RGB
            char_img_rgb = Image.new('RGB', char_img.size)
            char_img_rgb.paste(char_img)
            img.paste(char_img_rgb, (x, y))
        else:
            img.paste(char_img, (x, y))
        
        # Add character info
        text_y = y + char_img.height + 5
        if text_y < (row + 1) * cell_size - 15:
            draw.text((x, text_y), f"{char} ({ord(char)})", fill=(0, 0, 0))
    
    # Save the image
    output_path = os.path.join(RESULTS_DIR, "all_characters.png")
    img.save(output_path)
    logger.info(f"Sample image with all characters saved to {output_path}")

if __name__ == "__main__":
    main()
