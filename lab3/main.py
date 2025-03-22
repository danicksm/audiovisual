import numpy as np
import os
import time
import glob
from PIL import Image
import concurrent.futures

def load_image(image_path):
    """
    Загрузка изображения с использованием Pillow вместо OpenCV
    """
    return np.array(Image.open(image_path))

def save_image(image, output_path):
    """
    Сохранение изображения с использованием Pillow вместо OpenCV
    """
    # Убедимся, что изображение в формате uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Сохраняем изображение
    Image.fromarray(image).save(output_path)

def rgb_to_gray(image):
    """
    Преобразование RGB изображения в градации серого без использования cv2
    Формула: Y = 0.299*R + 0.587*G + 0.114*B
    """
    if len(image.shape) == 2:  # Если изображение уже в градациях серого
        return image
    
    # Используем взвешенную сумму каналов для преобразования в градации серого
    return np.round(0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)

def convert_to_monochrome(gray_image, threshold=127):
    """
    Преобразует полутоновое изображение в монохромное с помощью простой бинаризации
    """
    return (gray_image > threshold).astype(np.uint8) * 255

def difference_image(img1, img2):
    """
    Вычисляет абсолютную разность между двумя изображениями без использования cv2
    """
    return np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)

def split_channels(color_image):
    """
    Разделяет цветное изображение на каналы (аналог cv2.split)
    """
    if len(color_image.shape) < 3:
        return [color_image]
    
    # Изображение в формате RGB, разделяем по каналам
    return [color_image[:, :, i] for i in range(color_image.shape[2])]

def merge_channels(channels):
    """
    Объединяет каналы в цветное изображение (аналог cv2.merge)
    """
    if len(channels) == 1:
        return channels[0]
    
    # Преобразуем каналы в массивы и соединяем их
    return np.stack(channels, axis=2)

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

def rank_filter_optimized(image, rank=7):
    """
    Оптимизированная версия рангового фильтра, использующая numpy для ускорения операций.
    Фильтрует изображение, применяя ранговый фильтр с маской "прямой крест" 5x5.
    """
    height, width = image.shape
    filtered_image = np.zeros_like(image)
    
    # Предварительно создаем маски для всех пикселей
    # Создаем маски для горизонтальной и вертикальной линий креста
    offset_x = [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)]
    offset_y = [(0, -2), (0, -1), (0, 1), (0, 2)]  # Центр исключен, т.к. он уже в offset_x
    
    # Оптимизация для внутренних пикселей (где не нужно проверять границы)
    for y in range(2, height - 2):
        for x in range(2, width - 2):
            # Собираем значения соседей напрямую без проверки границ
            neighbors = [image[y + dy, x + dx] for dx, dy in offset_x + offset_y]
            neighbors.sort()
            filtered_image[y, x] = neighbors[min(rank - 1, len(neighbors) - 1)]
    
    # Обработка граничных пикселей (с проверкой границ)
    edge_coords = []
    for y in range(height):
        for x in range(width):
            if (y < 2 or y >= height - 2) or (x < 2 or x >= width - 2):
                edge_coords.append((y, x))
    
    for y, x in edge_coords:
        neighbors = []
        # Горизонтальные соседи
        for dx, dy in offset_x:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append(image[ny, nx])
        
        # Вертикальные соседи
        for dx, dy in offset_y:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append(image[ny, nx])
        
        neighbors.sort()
        filtered_image[y, x] = neighbors[min(rank - 1, len(neighbors) - 1)]
    
    return filtered_image

def visualize_straight_cross_mask(size=7):
    """
    Создает визуализацию маски "прямой крест" и сохраняет её как изображение.
    """
    if size % 2 == 0:
        size += 1  # Убедимся, что размер нечетный
    
    # Создаем пустую маску
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    # Заполняем горизонтальную линию
    mask[center, :] = 128
    # Заполняем вертикальную линию
    mask[:, center] = 128
    # Центральный пиксель отмечаем ярче
    mask[center, center] = 255
    
    # Сохраняем маску как изображение
    mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "straight_cross_mask.png")
    save_image(mask, mask_path)
    
    return mask_path

def create_progress_bar(total, current, bar_length=30):
    """
    Создает текстовую полосу прогресса для отображения в консоли.
    """
    progress = current / total
    arrow = '=' * int(round(progress * bar_length - 1)) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    return f"[{arrow}{spaces}] {int(progress * 100)}%"

def process_batch(image_batch, rank, y_start, y_end, width, offsets):
    """
    Обрабатывает пакет строк изображения.
    Используется для параллельной обработки частей изображения.
    """
    result = np.zeros((y_end - y_start, width), dtype=np.uint8)
    
    for y_local, y in enumerate(range(y_start, y_end)):
        for x in range(width):
            # Собираем значения из окрестности
            neighbors = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                # Проверяем границы
                if 0 <= ny < len(image_batch) + y_start and 0 <= nx < width:
                    if ny < y_start:
                        neighbors.append(image_batch[0, nx])
                    elif ny >= y_end:
                        neighbors.append(image_batch[-1, nx])
                    else:
                        neighbors.append(image_batch[ny - y_start, nx])
            
            neighbors.sort()
            result[y_local, x] = neighbors[min(rank - 1, len(neighbors) - 1)]
    
    return result

def rank_filter_parallel(image, rank=7, num_workers=None):
    """
    Параллельная версия рангового фильтра.
    Разделяет изображение на пакеты строк и обрабатывает их параллельно.
    """
    height, width = image.shape
    
    # Если число рабочих потоков не задано, используем количество доступных процессоров
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    
    # Создаем смещения для маски "прямой крест"
    offsets = [(0, dx) for dx in range(-2, 3)] + [(dy, 0) for dy in range(-2, 3) if dy != 0]
    
    # Разделяем изображение на пакеты строк
    batch_size = max(1, height // num_workers)
    batches = []
    
    for i in range(0, height, batch_size):
        y_start = i
        y_end = min(i + batch_size, height)
        batches.append((y_start, y_end))
    
    results = []
    
    # Обрабатываем пакеты параллельно
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, 
                           image[max(0, y_start-2):min(height, y_end+2)], 
                           rank, y_start, y_end, width, offsets): (y_start, y_end) 
            for y_start, y_end in batches
        }
        
        completed = 0
        print("  [Прогресс обработки]:", end="", flush=True)
        
        for future in concurrent.futures.as_completed(future_to_batch):
            y_start, y_end = future_to_batch[future]
            try:
                result = future.result()
                results.append((y_start, result))
                
                # Отображаем прогресс
                completed += 1
                progress = create_progress_bar(len(batches), completed)
                print(f"\r  [Прогресс обработки]: {progress}", end="", flush=True)
            except Exception as e:
                print(f"\nОшибка при обработке строк {y_start}-{y_end}: {e}")
    
    print()  # Новая строка после полосы прогресса
    
    # Объединяем результаты
    filtered_image = np.zeros_like(image)
    for y_start, result in sorted(results, key=lambda x: x[0]):
        filtered_image[y_start:y_start+result.shape[0]] = result
    
    return filtered_image

# Заменяем текущую функцию рангового фильтра на параллельную версию
rank_filter = rank_filter_parallel

def apply_filter_to_color_image(color_image, rank=7, use_parallel=True):
    """
    Применяет ранговый фильтр к каждому каналу цветного изображения отдельно.
    Параметр use_parallel позволяет включить параллельную обработку каналов.
    """
    # Разделяем изображение на цветовые каналы
    channels = split_channels(color_image)
    
    if use_parallel and len(channels) > 1:
        # Параллельная обработка каналов
        with concurrent.futures.ThreadPoolExecutor() as executor:
            filtered_channels = list(executor.map(lambda ch: rank_filter(ch, rank), channels))
    else:
        # Последовательная обработка каналов
        filtered_channels = [rank_filter(channel, rank) for channel in channels]
    
    # Объединяем фильтрованные каналы обратно в цветное изображение
    return merge_channels(filtered_channels)

def process_image(image_path, output_dir, is_color=True, rank=7, use_parallel=True):
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
    input_image = load_image(image_path)
    
    if is_color and len(input_image.shape) == 3:
        input_gray = rgb_to_gray(input_image)
    else:
        # Если изображение уже в градациях серого или загружено как таковое
        input_image = input_image if len(input_image.shape) == 2 else rgb_to_gray(input_image)
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
    save_image(input_mono, os.path.join(image_output_dir, f"original_monochrome.png"))
    save_image(filtered_mono, os.path.join(image_output_dir, f"filtered_monochrome.png"))
    save_image(diff_mono, os.path.join(image_output_dir, f"difference_monochrome.png"))
    
    print(f"  Обработка в градациях серого...")
    # Обрабатываем изображение в градациях серого
    start_time = time.time()
    filtered_gray = rank_filter(input_gray, rank)
    end_time = time.time()
    gray_processing_time = end_time - start_time
    
    diff_gray = difference_image(input_gray, filtered_gray)
    
    # Сохраняем результаты для изображения в градациях серого
    save_image(input_gray, os.path.join(image_output_dir, f"original_grayscale.png"))
    save_image(filtered_gray, os.path.join(image_output_dir, f"filtered_grayscale.png"))
    save_image(diff_gray, os.path.join(image_output_dir, f"difference_grayscale.png"))
    
    color_processing_time = None
    
    # Обрабатываем цветное изображение, если запрошено и изображение цветное
    if is_color and len(input_image.shape) == 3:
        print(f"  Обработка цветного изображения...")
        start_time = time.time()
        filtered_color = apply_filter_to_color_image(input_image, rank, use_parallel)
        end_time = time.time()
        color_processing_time = end_time - start_time
        
        diff_color = difference_image(input_image, filtered_color)
        
        # Сохраняем результаты для цветного изображения
        save_image(input_image, os.path.join(image_output_dir, f"original_color.png"))
        save_image(filtered_color, os.path.join(image_output_dir, f"filtered_color.png"))
        save_image(diff_color, os.path.join(image_output_dir, f"difference_color.png"))
    
    return processing_time, gray_processing_time, color_processing_time

def main():
    # Директории для входных и выходных данных
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "images")
    output_dir = os.path.join(current_dir, "results")
    
    # Убедимся, что выходная директория существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем и сохраняем визуализацию маски "прямой крест"
    mask_path = visualize_straight_cross_mask(7)
    print(f"Визуализация маски 'прямой крест' сохранена в: {mask_path}")
    
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
    use_parallel = True  # Включаем параллельную обработку для цветных изображений
    
    print(f"Начинаем обработку изображений с помощью рангового фильтра (ранг {rank})")
    print(f"Используется маска 'прямой крест' 5x5 (до 9 соседей)")
    if use_parallel:
        print(f"Включена параллельная обработка каналов для цветных изображений")
    
    # Обрабатываем каждое изображение
    for image_path in image_paths:
        print(f"\nОбработка изображения: {image_path}")
        
        try:
            mono_time, gray_time, color_time = process_image(image_path, output_dir, is_color=True, rank=rank, use_parallel=use_parallel)
            
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