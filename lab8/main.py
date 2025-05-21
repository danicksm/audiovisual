import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Create results directory
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

def rgb_to_hsl(image):
    """Convert RGB image to HSL color space."""
    # OpenCV loads images in BGR format
    # Convert to RGB for easier processing
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to HSV (OpenCV doesn't have direct HSL conversion)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    # Scale the V channel to L for HSL-like representation
    return hsv_img

def apply_power_transform(img, gamma=2.0):
    """Apply power-law (gamma) transformation to the brightness channel."""
    # Normalize pixel values
    normalized = img / 255.0
    # Apply power law transformation
    transformed = np.power(normalized, gamma) * 255.0
    # Convert back to uint8
    return np.uint8(transformed)

def compute_ngldm(image, d=2, a=1, bins=256):
    """
    Compute Neighborhood Gray-Level Dependence Matrix (NGLDM)
    
    Parameters:
    - image: grayscale image
    - d: distance
    - a: range of gray level 
    - bins: number of gray levels
    
    Returns:
    - ngldm: Neighborhood Gray-Level Dependence Matrix
    """
    if bins < 256:
        # Quantize gray levels
        image = np.floor(image / (256 / bins)).astype(np.uint8)
    
    height, width = image.shape
    ngldm = np.zeros((bins, 9), dtype=np.int32)
    
    # For each pixel, compute number of neighbors within range a
    for i in range(d, height - d):
        for j in range(d, width - d):
            center = image[i, j]
            s = 0  # Number of neighbors with similar gray level
            
            # Check 8-neighborhood at distance d
            for di in [-d, 0, d]:
                for dj in [-d, 0, d]:
                    if di == 0 and dj == 0:
                        continue
                    
                    neighbor = image[i + di, j + dj]
                    if abs(int(neighbor) - int(center)) <= a:
                        s += 1
            
            # Update NGLDM
            ngldm[center, s] += 1
    
    return ngldm

def calculate_features(ngldm):
    """
    Calculate texture features from NGLDM
    
    Features:
    - NN: Number Nonuniformity
    - SM: Small Number Emphasis
    - ENT: Entropy
    """
    # Filter out zero entries to avoid log(0)
    mask = ngldm > 0
    
    # Total number of entries
    N = np.sum(ngldm)
    
    # Small Number Emphasis (SM)
    s_values = np.arange(ngldm.shape[1])
    s_squared = s_values ** 2
    sm = np.sum(ngldm * (1 / (s_squared + 1e-10))) / N
    
    # Number Nonuniformity (NN)
    g_sum = np.sum(ngldm, axis=1)
    nn = np.sum(g_sum ** 2) / N
    
    # Entropy (ENT)
    p = ngldm[mask] / N  # Probability
    ent = -np.sum(p * np.log2(p + 1e-10))
    
    return {
        "Number Nonuniformity (NN)": nn,
        "Small Number Emphasis (SM)": sm,
        "Entropy (ENT)": ent
    }

def process_image(image_path):
    """Process an image to extract NGLDM features before and after power transform."""
    # Generate unique timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"result_{timestamp}"
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Не удалось открыть файл: {image_path}")
        return
    
    # Convert to HSL-like space
    hsl_image = rgb_to_hsl(image)
    
    # Extract luminance channel (V in HSV)
    l_channel = hsl_image[:, :, 2]
    
    # Apply power transform to luminance
    gamma = 0.5  # Gamma < 1 increases brightness, Gamma > 1 decreases brightness
    l_channel_transformed = apply_power_transform(l_channel, gamma)
    
    # Compute NGLDM for original and transformed images
    ngldm_original = compute_ngldm(l_channel, d=2, a=1, bins=64)
    ngldm_transformed = compute_ngldm(l_channel_transformed, d=2, a=1, bins=64)
    
    # Calculate features
    features_original = calculate_features(ngldm_original)
    features_transformed = calculate_features(ngldm_transformed)
    
    # Create visualization
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle("ЛР8. NGLDM анализ и степенное контрастирование", fontsize=16)
    
    # 1. Original HSL image
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Исходное изображение")
    axs[0, 0].axis('off')
    
    # 2. Original luminance channel
    axs[0, 1].imshow(l_channel, cmap='gray')
    axs[0, 1].set_title("Канал яркости (L)")
    axs[0, 1].axis('off')
    
    # 3. Transformed luminance channel
    axs[1, 0].imshow(l_channel_transformed, cmap='gray')
    axs[1, 0].set_title(f"После степенного преобразования (gamma={gamma})")
    axs[1, 0].axis('off')
    
    # 4. Histogram of original luminance
    axs[1, 1].hist(l_channel.ravel(), bins=256, color='gray')
    axs[1, 1].set_title("Гистограмма яркости (до преобразования)")
    axs[1, 1].set_xlabel("Яркость")
    axs[1, 1].set_ylabel("Частота")
    
    # 5. Histogram of transformed luminance
    axs[2, 0].hist(l_channel_transformed.ravel(), bins=256, color='gray')
    axs[2, 0].set_title("Гистограмма яркости (после преобразования)")
    axs[2, 0].set_xlabel("Яркость")
    axs[2, 0].set_ylabel("Частота")
    
    # 6. Visualize NGLDM as bar chart for original image
    s_values = np.arange(ngldm_original.shape[1])
    ngldm_sum_by_s_orig = np.sum(ngldm_original, axis=0)
    axs[2, 1].bar(s_values, ngldm_sum_by_s_orig, color='blue', alpha=0.7)
    axs[2, 1].set_title("NGLDM распределение (до преобразования)")
    axs[2, 1].set_xlabel("Число соседей")
    axs[2, 1].set_ylabel("Частота")
    
    # 7. Visualize NGLDM as bar chart for transformed image
    ngldm_sum_by_s_transform = np.sum(ngldm_transformed, axis=0)
    axs[3, 0].bar(s_values, ngldm_sum_by_s_transform, color='red', alpha=0.7)
    axs[3, 0].set_title("NGLDM распределение (после преобразования)")
    axs[3, 0].set_xlabel("Число соседей")
    axs[3, 0].set_ylabel("Частота")
    
    # 8. Features comparison
    axs[3, 1].axis('off')
    feature_text = "Признаки NGLDM:\n\n"
    
    for feature_name in features_original:
        orig_val = features_original[feature_name]
        trans_val = features_transformed[feature_name]
        diff = ((trans_val - orig_val) / orig_val) * 100 if orig_val != 0 else 0
        
        feature_text += f"{feature_name}:\n"
        feature_text += f"  До:     {orig_val:.4f}\n"
        feature_text += f"  После:  {trans_val:.4f}\n"
        feature_text += f"  Δ:      {diff:.2f}%\n\n"
    
    axs[3, 1].text(0.05, 0.95, feature_text, fontsize=12, 
                   verticalalignment='top', transform=axs[3, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{prefix}_full_report.png"))
    
    # Create a more detailed visualization of NGLDM matrix
    plt.figure(figsize=(14, 7))
    
    # Plot heatmap of NGLDM for the original image
    plt.subplot(1, 2, 1)
    log_ngldm_orig = np.log1p(ngldm_original[:64, :])  # log for better visibility
    plt.imshow(log_ngldm_orig, cmap='viridis', aspect='auto')
    plt.colorbar(label='Log(Count + 1)')
    plt.title('NGLDM матрица (до преобразования)')
    plt.xlabel('Число соседей')
    plt.ylabel('Уровень серого')
    
    # Plot heatmap of NGLDM for the transformed image
    plt.subplot(1, 2, 2)
    log_ngldm_transform = np.log1p(ngldm_transformed[:64, :])  # log for better visibility
    plt.imshow(log_ngldm_transform, cmap='viridis', aspect='auto')
    plt.colorbar(label='Log(Count + 1)')
    plt.title('NGLDM матрица (после преобразования)')
    plt.xlabel('Число соседей')
    plt.ylabel('Уровень серого')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{prefix}_ngldm_matrix.png"))
    
    print(f"[✓] Готово! Результаты сохранены в папку '{results_dir}'.")

# Process both images
for image_file in os.listdir("images"):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Обработка изображения: {image_file}")
        process_image(os.path.join("images", image_file))
