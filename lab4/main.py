#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor

# Функция для создания директории, если она не существует
def ensure_dir(directory):
    """
    Creates directory if it doesn't exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# Функция для преобразования цветного изображения в полутоновое
def convert_to_grayscale(image):
    """
    Converts color image to grayscale using NumPy
    """
    # Используем формулу Y = 0.299*R + 0.587*G + 0.114*B
    if len(image.shape) == 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return image

# Функция для применения свертки
def apply_convolution(image, kernel):
    """
    Applies convolution using NumPy
    """
    # Получаем размеры изображения и ядра
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Получаем смещения для ядра
    pad_h = k_height // 2
    pad_w = k_width // 2
    
    # Создаем выходное изображение
    output = np.zeros_like(image, dtype=np.float32)
    
    # Дополняем изображение нулями для корректной свертки на границах
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Применяем свертку
    for i in range(i_height):
        for j in range(i_width):
            # Извлекаем область, соответствующую размеру ядра
            region = padded_image[i:i+k_height, j:j+k_width]
            # Применяем свертку
            output[i, j] = np.sum(region * kernel)
    
    return output

# Функция для вычисления градиентов по оператору Робертса
def roberts_operator(image):
    """
    Applies Roberts operator to the image and returns Gx, Gy, and G matrices
    """
    # Оператор Робертса 3x3 из условия задачи
    # Ядро для градиента по X
    roberts_x = np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Ядро для градиента по Y
    roberts_y = np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Вычисление градиентов с использованием NumPy вместо OpenCV
    Gx = apply_convolution(image, roberts_x)
    Gy = apply_convolution(image, roberts_y)
    
    # Вычисление итогового градиента G = sqrt(Gx^2 + Gy^2)
    G = np.sqrt(Gx**2 + Gy**2)
    
    return Gx, Gy, G

# Функция для нормализации матрицы градиента в диапазон [0, 255]
def normalize_gradient(gradient):
    """
    Normalizes gradient matrix to [0, 255] range using NumPy
    """
    min_val = np.min(gradient)
    max_val = np.max(gradient)
    
    # Избегаем деления на ноль
    if max_val == min_val:
        return np.zeros_like(gradient, dtype=np.uint8)
    
    # Нормализация в диапазон [0, 255]
    normalized = 255.0 * (gradient - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)

# Функция для бинаризации градиентного изображения
def binarize_gradient(gradient, threshold=50):
    """
    Binarizes gradient image with given threshold using NumPy
    """
    return np.where(gradient > threshold, 255, 0).astype(np.uint8)

# Функция для обработки одного изображения
def process_image(image_path, threshold=50):
    """
    Processes a single image with Roberts operator
    """
    # Получаем имя файла без расширения для создания директории результатов
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    results_dir = f"./results/{image_name}"
    ensure_dir(results_dir)
    
    # Загружаем изображение (используем OpenCV только для загрузки)
    print(f"Обработка изображения: {image_name}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return
    
    # Сохраняем исходное цветное изображение
    cv2.imwrite(f"{results_dir}/original_color.png", image)
    
    # Преобразуем в полутоновое и сохраняем
    gray = convert_to_grayscale(image)
    cv2.imwrite(f"{results_dir}/grayscale.png", gray)
    
    # Вычисляем градиенты
    Gx, Gy, G = roberts_operator(gray)
    
    # Нормализуем градиенты
    Gx_norm = normalize_gradient(Gx)
    Gy_norm = normalize_gradient(Gy)
    G_norm = normalize_gradient(G)
    
    # Сохраняем нормализованные градиенты
    cv2.imwrite(f"{results_dir}/gradient_x.png", Gx_norm)
    cv2.imwrite(f"{results_dir}/gradient_y.png", Gy_norm)
    cv2.imwrite(f"{results_dir}/gradient.png", G_norm)
    
    # Бинаризуем итоговый градиент и сохраняем
    G_binary = binarize_gradient(G_norm, threshold)
    cv2.imwrite(f"{results_dir}/gradient_binary.png", G_binary)
    
    print(f"Результаты обработки сохранены в: {results_dir}")
    
    # Возвращаем пути для использования в README
    return {
        "name": image_name,
        "original": f"{results_dir}/original_color.png",
        "grayscale": f"{results_dir}/grayscale.png",
        "gradient_x": f"{results_dir}/gradient_x.png",
        "gradient_y": f"{results_dir}/gradient_y.png",
        "gradient": f"{results_dir}/gradient.png",
        "binary": f"{results_dir}/gradient_binary.png"
    }

# Функция для визуализации оператора Робертса
def visualize_roberts_operator():
    """
    Creates a visualization of the Roberts operator
    """
    ensure_dir("./results")
    
    # Создаем изображения для визуализации ядер оператора Робертса из условия
    roberts_x = np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    roberts_y = np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Масштабируем для лучшей визуализации
    visualization_size = 100
    roberts_x_visual = np.zeros((3*visualization_size, 3*visualization_size), dtype=np.uint8)
    roberts_y_visual = np.zeros((3*visualization_size, 3*visualization_size), dtype=np.uint8)
    
    # Заполняем визуализации значениями (масштабированными)
    for i in range(3):
        for j in range(3):
            # Для Gx
            value_x = roberts_x[i, j]
            color_x = 255 if value_x > 0 else (0 if value_x < 0 else 128)
            y1_x, y2_x = i*visualization_size, (i+1)*visualization_size
            x1_x, x2_x = j*visualization_size, (j+1)*visualization_size
            roberts_x_visual[y1_x:y2_x, x1_x:x2_x] = color_x
            
            # Для Gy
            value_y = roberts_y[i, j]
            color_y = 255 if value_y > 0 else (0 if value_y < 0 else 128)
            y1_y, y2_y = i*visualization_size, (i+1)*visualization_size
            x1_y, x2_y = j*visualization_size, (j+1)*visualization_size
            roberts_y_visual[y1_y:y2_y, x1_y:x2_y] = color_y
    
    # Сохраняем визуализации
    cv2.imwrite("./results/roberts_operator_x.png", roberts_x_visual)
    cv2.imwrite("./results/roberts_operator_y.png", roberts_y_visual)

# Основная функция
def main():
    """
    Main function that processes all images in the images directory
    """
    # Создаем директорию для результатов
    ensure_dir("./results")
    
    # Визуализируем оператор Робертса
    visualize_roberts_operator()
    
    # Получаем список всех изображений в директории images
    image_paths = glob.glob("./images/*.png") + glob.glob("./images/*.jpg") + glob.glob("./images/*.jpeg")
    
    if not image_paths:
        print("Изображения не найдены в директории ./images/")
        return
    
    # Порог для бинаризации (можно изменить)
    threshold = 50
    
    # Обрабатываем все изображения параллельно
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_path, threshold) for image_path in image_paths]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    
    print(f"Обработано изображений: {len(results)}")

if __name__ == "__main__":
    main()
