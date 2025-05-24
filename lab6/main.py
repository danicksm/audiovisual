#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_IMAGE = "phrase.bmp"
RESULTS_DIR = "results"
THRESHOLD = 127  # Threshold for binarization
MIN_CHAR_WIDTH = 5  # Минимальная ширина символа в пикселях
GAP_SIZE = 1  # Уменьшаем размер промежутка между символами (было 2)
OVERLAP_THRESHOLD = 0.3  # Максимальное допустимое перекрытие прямоугольников (было 0.5)

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_image(image_path):
    """
    Load and preprocess the image (convert to grayscale and binarize).
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        tuple: Original image, grayscale image, and binary image
    """
    logger.info(f"Loading image from {image_path}")
    
    # Read the image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image {image_path} not found")
    
    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image (text is black, background is white)
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Invert binary image so text is white (1) and background is black (0)
    # This is more convenient for calculating profiles
    binary_inv = cv2.bitwise_not(binary)
    
    return original, gray, binary_inv

def calculate_profiles(binary_image):
    """
    Calculate horizontal and vertical profiles of the image.
    
    Args:
        binary_image (numpy.ndarray): Binary image with text as white (1) and background as black (0)
        
    Returns:
        tuple: Horizontal profile and vertical profile
    """
    logger.info("Calculating horizontal and vertical profiles")
    
    # Calculate horizontal profile (sum of white pixels in each row)
    horizontal_profile = np.sum(binary_image, axis=1)
    
    # Calculate vertical profile (sum of white pixels in each column)
    vertical_profile = np.sum(binary_image, axis=0)
    
    return horizontal_profile, vertical_profile

def plot_profiles(original_image, binary_image, horizontal_profile, vertical_profile):
    """
    Plot the image and its profiles.
    
    Args:
        original_image (numpy.ndarray): Original image
        binary_image (numpy.ndarray): Binary image
        horizontal_profile (numpy.ndarray): Horizontal profile
        vertical_profile (numpy.ndarray): Vertical profile
    """
    logger.info("Plotting profiles")
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot original image
    axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Исходное изображение')
    axs[0, 0].axis('off')
    
    # Plot binary image
    axs[0, 1].imshow(binary_image, cmap='gray')
    axs[0, 1].set_title('Бинарное изображение')
    axs[0, 1].axis('off')
    
    # Plot horizontal profile
    axs[1, 0].plot(horizontal_profile, np.arange(len(horizontal_profile)), 'b-')
    axs[1, 0].set_ylabel('Строка (пиксели)')
    axs[1, 0].set_xlabel('Сумма белых пикселей')
    axs[1, 0].set_title('Горизонтальный профиль')
    axs[1, 0].invert_yaxis()  # Invert y-axis to match image coordinates
    
    # Plot vertical profile
    axs[1, 1].plot(np.arange(len(vertical_profile)), vertical_profile, 'r-')
    axs[1, 1].set_xlabel('Столбец (пиксели)')
    axs[1, 1].set_ylabel('Сумма белых пикселей')
    axs[1, 1].set_title('Вертикальный профиль')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(RESULTS_DIR, "profiles.png")
    plt.savefig(output_path, dpi=150)
    logger.info(f"Profiles saved to {output_path}")
    
    plt.close()

def segment_characters(binary_image, vertical_profile, horizontal_profile):
    """
    Segment characters based on profiles with thinning.
    
    Args:
        binary_image (numpy.ndarray): Binary image
        vertical_profile (numpy.ndarray): Vertical profile
        horizontal_profile (numpy.ndarray): Horizontal profile
        
    Returns:
        list: List of bounding boxes for characters (x, y, w, h)
    """
    logger.info("Segmenting characters")
    
    # 1. Identify text lines using horizontal profile
    # Use a lower threshold to capture all parts of characters
    line_threshold = max(horizontal_profile) * 0.02  # Снижен порог для захвата высоких элементов
    
    # Find rows where profile exceeds threshold
    rows_with_text = np.where(horizontal_profile > line_threshold)[0]
    
    if len(rows_with_text) == 0:
        logger.warning("No text lines detected")
        return []
    
    # Get the full text line range
    line_start = np.min(rows_with_text)
    line_end = np.max(rows_with_text)
    
    # 2. Segment characters using vertical profile and connected components
    # Extract the text line
    line_image = binary_image[line_start:line_end+1, :]
    
    # Use a lower threshold for character detection
    char_threshold = max(vertical_profile) * 0.005  # Еще ниже порог для захвата всех символов (было 0.01)
    
    # Find columns where profile exceeds threshold
    cols_with_text = np.where(vertical_profile > char_threshold)[0]
    
    if len(cols_with_text) == 0:
        logger.warning("No characters detected")
        return []
    
    # Метод 1: Используем анализ вертикального профиля для нахождения разрывов между символами
    # Находим разрывы (нулевые области) в вертикальном профиле
    
    # Сначала сгладим профиль для устранения шума
    smoothed_profile = vertical_profile.copy()
    window_size = 3
    for i in range(window_size, len(vertical_profile) - window_size):
        smoothed_profile[i] = np.mean(vertical_profile[i-window_size:i+window_size+1])
    
    # Находим локальные минимумы в профиле
    local_mins = []
    for i in range(1, len(smoothed_profile)-1):
        if (smoothed_profile[i] < smoothed_profile[i-1] and 
            smoothed_profile[i] <= smoothed_profile[i+1] and
            smoothed_profile[i] < max(smoothed_profile) * 0.3):  # Только значимые минимумы
            local_mins.append(i)
    
    # Находим разрывы на основе локальных минимумов и порога
    gaps = []
    i = 0
    while i < len(cols_with_text) - 1:
        if cols_with_text[i+1] - cols_with_text[i] > GAP_SIZE:
            # Проверяем, есть ли локальный минимум в этом месте
            gap_center = (cols_with_text[i] + cols_with_text[i+1]) // 2
            is_local_min = any(abs(lm - gap_center) < 3 for lm in local_mins)
            
            if is_local_min or cols_with_text[i+1] - cols_with_text[i] > 2 * GAP_SIZE:
                gaps.append((cols_with_text[i], cols_with_text[i+1]))
        i += 1
    
    # Создаем границы символов на основе разрывов
    char_boundaries = []
    if len(gaps) == 0:  # Если разрывов нет, считаем всё одним символом
        char_boundaries.append((cols_with_text[0], cols_with_text[-1]))
    else:
        # Добавляем начало первого символа
        char_boundaries.append((cols_with_text[0], gaps[0][0]))
        
        # Добавляем символы между разрывами
        for i in range(len(gaps)-1):
            char_boundaries.append((gaps[i][1], gaps[i+1][0]))
        
        # Добавляем последний символ
        char_boundaries.append((gaps[-1][1], cols_with_text[-1]))
    
    # Метод 2: Дополнительно используем анализ связных компонент
    # Создаем копию изображения для обнаружения связных компонент
    contours, _ = cv2.findContours(line_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Создаем прямоугольники для каждой связной компоненты
    contour_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= MIN_CHAR_WIDTH:  # Фильтруем слишком маленькие компоненты
            contour_boxes.append((x, y + line_start, w, h))
    
    # Объединяем результаты обоих методов
    bounding_boxes = []
    
    # Добавляем прямоугольники на основе разрывов в вертикальном профиле
    for x_start, x_end in char_boundaries:
        width = x_end - x_start + 1
        if width >= MIN_CHAR_WIDTH:  # Проверяем минимальную ширину
            bounding_boxes.append((x_start, line_start, width, line_end - line_start + 1))
    
    # Добавляем прямоугольники на основе связных компонент
    bounding_boxes.extend(contour_boxes)
    
    # Удаляем дубликаты и перекрывающиеся прямоугольники
    filtered_boxes = []
    for box in bounding_boxes:
        add_box = True
        x1, y1, w1, h1 = box
        
        # Проверяем, не перекрывается ли текущий прямоугольник с уже добавленными
        for existing_box in filtered_boxes:
            x2, y2, w2, h2 = existing_box
            
            # Вычисляем площадь перекрытия
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            # Вычисляем площадь текущего прямоугольника
            box_area = w1 * h1
            
            # Если перекрытие слишком большое, не добавляем текущий прямоугольник
            if overlap_area > OVERLAP_THRESHOLD * box_area:  # Используем константу вместо хардкода
                add_box = False
                break
        
        if add_box:
            filtered_boxes.append(box)
    
    # Проверяем, нет ли слишком широких прямоугольников, которые могут содержать несколько символов
    max_char_width = np.median([box[2] for box in filtered_boxes]) * 1.5
    refined_boxes = []
    
    for box in filtered_boxes:
        x, y, w, h = box
        if w > max_char_width and w > 20:  # Прямоугольник слишком широкий
            # Пытаемся разделить его на несколько символов
            num_splits = max(2, int(w / (max_char_width * 0.8)))
            split_width = w // num_splits
            
            for i in range(num_splits):
                new_x = x + i * split_width
                new_w = split_width if i < num_splits - 1 else w - (i * split_width)
                refined_boxes.append((new_x, y, new_w, h))
        else:
            refined_boxes.append(box)
    
    # Сортируем прямоугольники слева направо (для иврита - справа налево при чтении, но в коде мы сохраняем слева направо)
    refined_boxes.sort(key=lambda box: box[0])
    
    logger.info(f"Detected {len(refined_boxes)} characters")
    return refined_boxes

def visualize_segmentation(original_image, binary_image, bounding_boxes):
    """
    Visualize character segmentation by drawing bounding boxes.
    
    Args:
        original_image (numpy.ndarray): Original image
        binary_image (numpy.ndarray): Binary image
        bounding_boxes (list): List of bounding boxes (x, y, w, h)
    """
    logger.info("Visualizing character segmentation")
    
    # Create a copy of the original image for drawing
    result_image = original_image.copy()
    
    # Draw bounding boxes
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Отключаем номера символов, чтобы не загромождать изображение
        # cv2.putText(result_image, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Save result image
    output_path = os.path.join(RESULTS_DIR, "segmentation.png")
    cv2.imwrite(output_path, result_image)
    logger.info(f"Segmentation visualization saved to {output_path}")
    
    # Also create a figure with both original and segmented images
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Сегментация символов')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save comparison figure
    output_path = os.path.join(RESULTS_DIR, "segmentation_comparison.png")
    plt.savefig(output_path, dpi=150)
    logger.info(f"Segmentation comparison saved to {output_path}")
    
    plt.close()

def extract_characters(original_image, binary_image, bounding_boxes):
    """
    Extract individual characters based on bounding boxes.
    
    Args:
        original_image (numpy.ndarray): Original image
        binary_image (numpy.ndarray): Binary image
        bounding_boxes (list): List of bounding boxes (x, y, w, h)
    """
    logger.info("Extracting individual characters")
    
    characters = []
    
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Проверяем, не выходит ли область за границы изображения
        x = max(0, x)
        y = max(0, y)
        w = min(w, binary_image.shape[1] - x)
        h = min(h, binary_image.shape[0] - y)
        
        # Extract character from binary image
        char_img = binary_image[y:y+h, x:x+w]
        
        # Add to characters list
        characters.append(char_img)
        
        # Save character image
        output_path = os.path.join(RESULTS_DIR, f"char_{i+1}.png")
        cv2.imwrite(output_path, char_img)
    
    logger.info(f"Extracted {len(characters)} characters")
    return characters

def calculate_character_profiles(characters):
    """
    Calculate profiles for each extracted character.
    
    Args:
        characters (list): List of character images
    """
    logger.info("Calculating character profiles")
    
    for i, char_img in enumerate(characters):
        # Calculate horizontal and vertical profiles
        h_profile = np.sum(char_img, axis=1)
        v_profile = np.sum(char_img, axis=0)
        
        # Create a figure with the character and its profiles
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot character
        axs[0].imshow(char_img, cmap='gray')
        axs[0].set_title(f'Символ {i+1}')
        axs[0].axis('off')
        
        # Plot horizontal profile
        axs[1].plot(h_profile, np.arange(len(h_profile)), 'b-')
        axs[1].set_ylabel('Строка (пиксели)')
        axs[1].set_xlabel('Сумма белых пикселей')
        axs[1].set_title('Горизонтальный профиль')
        axs[1].invert_yaxis()  # Invert y-axis to match image coordinates
        
        # Plot vertical profile
        axs[2].plot(np.arange(len(v_profile)), v_profile, 'r-')
        axs[2].set_xlabel('Столбец (пиксели)')
        axs[2].set_ylabel('Сумма белых пикселей')
        axs[2].set_title('Вертикальный профиль')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(RESULTS_DIR, f"char_{i+1}_profiles.png")
        plt.savefig(output_path, dpi=150)
        
        plt.close()
    
    logger.info("Character profiles calculated and saved")

def create_character_grid(characters):
    """
    Create a grid with all extracted characters for easier visualization.
    
    Args:
        characters (list): List of character images
    """
    logger.info("Creating character grid")
    
    num_chars = len(characters)
    
    if num_chars == 0:
        logger.warning("No characters to display")
        return
    
    # Determine grid dimensions
    cols = min(8, num_chars)
    rows = (num_chars + cols - 1) // cols
    
    # Create figure
    plt.figure(figsize=(15, rows * 2))
    
    for i, char_img in enumerate(characters):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(char_img, cmap='gray')
        plt.title(f'Символ {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save grid
    output_path = os.path.join(RESULTS_DIR, "character_grid.png")
    plt.savefig(output_path, dpi=150)
    logger.info(f"Character grid saved to {output_path}")
    
    plt.close()

def detect_text_block(binary_image, horizontal_profile, vertical_profile):
    """
    Detect the bounding rectangle for the text block as a whole.
    
    Args:
        binary_image (numpy.ndarray): Binary image
        horizontal_profile (numpy.ndarray): Horizontal profile
        vertical_profile (numpy.ndarray): Vertical profile
        
    Returns:
        tuple: Bounding box for the entire text block (x, y, w, h)
    """
    logger.info("Detecting text block")
    
    # Threshold for detection
    h_threshold = max(horizontal_profile) * 0.02  # Снижен порог для захвата всех частей символов
    v_threshold = max(vertical_profile) * 0.02  # Снижен порог для захвата всех частей символов
    
    # Find rows and columns with text
    rows_with_text = np.where(horizontal_profile > h_threshold)[0]
    cols_with_text = np.where(vertical_profile > v_threshold)[0]
    
    if len(rows_with_text) == 0 or len(cols_with_text) == 0:
        logger.warning("No text block detected")
        return None
    
    # Create bounding box
    x = cols_with_text[0]
    y = rows_with_text[0]
    w = cols_with_text[-1] - x + 1
    h = rows_with_text[-1] - y + 1
    
    logger.info(f"Text block detected at ({x}, {y}) with size {w}x{h}")
    return (x, y, w, h)

def visualize_text_block(original_image, text_block):
    """
    Visualize the text block by drawing a bounding box.
    
    Args:
        original_image (numpy.ndarray): Original image
        text_block (tuple): Bounding box for the text block (x, y, w, h)
    """
    if text_block is None:
        logger.warning("No text block to visualize")
        return
    
    logger.info("Visualizing text block")
    
    # Create a copy of the original image for drawing
    result_image = original_image.copy()
    
    # Extract bounding box coordinates
    x, y, w, h = text_block
    
    # Draw bounding box
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Save result image
    output_path = os.path.join(RESULTS_DIR, "text_block.png")
    cv2.imwrite(output_path, result_image)
    logger.info(f"Text block visualization saved to {output_path}")

def main():
    """Main function to execute all tasks."""
    logger.info("Starting Lab 6 - Text Segmentation")
    
    try:
        # 1. Load and preprocess image
        original, gray, binary = load_and_preprocess_image(INPUT_IMAGE)
        
        # 2. Calculate horizontal and vertical profiles
        horizontal_profile, vertical_profile = calculate_profiles(binary)
        plot_profiles(original, binary, horizontal_profile, vertical_profile)
        
        # 3. Segment characters
        bounding_boxes = segment_characters(binary, vertical_profile, horizontal_profile)
        visualize_segmentation(original, binary, bounding_boxes)
        
        # Extract individual characters
        characters = extract_characters(original, binary, bounding_boxes)
        
        # 4. Calculate profiles for each character
        calculate_character_profiles(characters)
        
        # Create a grid with all characters
        create_character_grid(characters)
        
        # 5. (Optional) Detect text block
        text_block = detect_text_block(binary, horizontal_profile, vertical_profile)
        visualize_text_block(original, text_block)
        
        logger.info("Lab 6 completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()


