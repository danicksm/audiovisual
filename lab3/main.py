import cv2
import numpy as np
import os
import time
import glob

def get_straight_cross_neighbors(img, x, y):
    """
    Функция возвращает отсортированный список значений пикселей в маске "прямой крест".
    В прямом кресте 5x5 включены только пиксели по вертикали и горизонтали от центрального.
    """
    neighbors = []
    height, width = img.shape  # Получаем размеры изображения
    
    # Определяем позиции для прямого креста 5x5
    # Горизонтальная линия креста
    for dx in [-2, -1, 0, 1, 2]:
        nx, ny = x + dx, y  # Горизонтальные соседи
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append(img[ny, nx])
    
    # Вертикальная линия креста (без центра, т.к. он уже добавлен)
    for dy in [-2, -1, 1, 2]:
        nx, ny = x, y + dy  # Вертикальные соседи
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append(img[ny, nx])
    
    return sorted(neighbors)  # Сортируем список значений соседей

def rank_filter(image, rank=7):
    """
    Фильтрует изображение, применяя ранговый фильтр с маской "прямой крест" 5x5.
    Для каждого пикселя выбирается значение, соответствующее заданному рангу в отсортированном списке соседей.
    """
    height, width = image.shape  # Получаем размеры изображения
    filtered_image = np.zeros((height, width), dtype=np.uint8)  # Создаем новое изображение для результата
    
    # Проходим по каждому пикселю изображения
    for y in range(height):
        for x in range(width):
            neighbors = get_straight_cross_neighbors(image, x, y)  # Получаем соседей для текущего пикселя
            # Выбираем пиксель с позицией, соответствующей рангу
            # В прямом кресте 5x5 может быть до 9 пикселей
            filtered_image[y, x] = neighbors[min(rank - 1, len(neighbors) - 1)]  # Чтобы избежать выхода за пределы списка
    
    return filtered_image  # Возвращаем фильтрованное изображение

def difference_image(img1, img2):
    """
    Вычисляет разностное изображение как модуль разности между исходным и фильтрованным изображением.
    """
    return cv2.absdiff(img1, img2)  # Абсолютная разность между двумя изображениями

def convert_to_monochrome(gray_image, threshold=127):
    """
    Преобразует полутоновое изображение в монохромное с помощью бинаризации
    """
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_filter_to_color_image(color_image, rank=7):
    """
    Применяет ранговый фильтр к каждому каналу цветного изображения отдельно.
    """
    # Разделяем изображение на цветовые каналы
    b, g, r = cv2.split(color_image)
    
    # Применяем ранговый фильтр к каждому каналу
    filtered_b = rank_filter(b, rank)
    filtered_g = rank_filter(g, rank)
    filtered_r = rank_filter(r, rank)
    
    # Объединяем фильтрованные каналы обратно в цветное изображение
    return cv2.merge([filtered_b, filtered_g, filtered_r])

def process_image(image_path, output_dir, is_color=True, rank=7):
    """
    Обрабатывает изображение с помощью рангового фильтра и сохраняет результаты.
    """
    # Создаем имя файла и директорию для результатов
    image_name = os.path.basename(image_path)
    base_name, ext = os.path.splitext(image_name)
    
    # Создаем отдельную директорию для результатов обработки текущего изображения
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Читаем изображение
    if is_color:
        input_image = cv2.imread(image_path)
        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        input_gray = input_image
    
    # Преобразуем в монохромное изображение
    print(f"  Преобразование в монохромное изображение...")
    input_mono = convert_to_monochrome(input_gray)
    
    print(f"  Обработка в монохромном режиме...")
    # Обрабатываем монохромное изображение
    start_time = time.time()
    filtered_mono = rank_filter(input_mono, rank)
    end_time = time.time()
    processing_time = end_time - start_time
    
    diff_mono = difference_image(input_mono, filtered_mono)
    
    # Сохраняем результаты для монохромного изображения
    cv2.imwrite(os.path.join(image_output_dir, f"original_monochrome.png"), input_mono)
    cv2.imwrite(os.path.join(image_output_dir, f"filtered_monochrome.png"), filtered_mono)
    cv2.imwrite(os.path.join(image_output_dir, f"difference_monochrome.png"), diff_mono)
    
    print(f"  Обработка в градациях серого...")
    # Обрабатываем изображение в градациях серого
    start_time = time.time()
    filtered_gray = rank_filter(input_gray, rank)
    end_time = time.time()
    gray_processing_time = end_time - start_time
    
    diff_gray = difference_image(input_gray, filtered_gray)
    
    # Сохраняем результаты для изображения в градациях серого
    cv2.imwrite(os.path.join(image_output_dir, f"original_grayscale.png"), input_gray)
    cv2.imwrite(os.path.join(image_output_dir, f"filtered_grayscale.png"), filtered_gray)
    cv2.imwrite(os.path.join(image_output_dir, f"difference_grayscale.png"), diff_gray)
    
    color_processing_time = None
    
    # Обрабатываем цветное изображение, если запрошено
    if is_color:
        print(f"  Обработка цветного изображения...")
        start_time = time.time()
        filtered_color = apply_filter_to_color_image(input_image, rank)
        end_time = time.time()
        color_processing_time = end_time - start_time
        
        diff_color = cv2.absdiff(input_image, filtered_color)
        
        # Сохраняем результаты для цветного изображения
        cv2.imwrite(os.path.join(image_output_dir, f"original_color.png"), input_image)
        cv2.imwrite(os.path.join(image_output_dir, f"filtered_color.png"), filtered_color)
        cv2.imwrite(os.path.join(image_output_dir, f"difference_color.png"), diff_color)
    
    return processing_time, gray_processing_time, color_processing_time

def main():
    # Директории для входных и выходных данных
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "images")
    output_dir = os.path.join(current_dir, "results")
    
    print(f"Поиск изображений в директории: {input_dir}")
    
    # Получаем список всех изображений в директории images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_paths:
        print("Ошибка: в директории images не найдено изображений.")
        return
    
    print(f"Найдено изображений: {len(image_paths)}")
    
    rank = 7  # Ранг 7 для прямого креста (до 9 элементов)
    
    print(f"Начинаем обработку изображений с помощью рангового фильтра (ранг {rank})")
    print(f"Используется маска 'прямой крест' 5x5 (до 9 соседей)")
    
    # Обрабатываем каждое изображение
    for image_path in image_paths:
        print(f"\nОбработка изображения: {image_path}")
        
        try:
            mono_time, gray_time, color_time = process_image(image_path, output_dir, is_color=True, rank=rank)
            
            print(f"  Время обработки монохромного изображения: {mono_time:.3f} секунд")
            print(f"  Время обработки в градациях серого: {gray_time:.3f} секунд")
            if color_time is not None:
                print(f"  Время обработки цветного изображения: {color_time:.3f} секунд")
            
            image_name = os.path.basename(image_path)
            base_name, _ = os.path.splitext(image_name)
            print(f"  Результаты сохранены в: {os.path.join(output_dir, base_name)}")
            
        except Exception as e:
            print(f"  Ошибка при обработке {image_path}: {e}")
    
    print("\nОбработка завершена!")

if __name__ == "__main__":
    main() 