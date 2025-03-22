import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def rgb_to_grayscale(image_path, output_path):
    """
    Конвертирует цветное изображение в градации серого методом взвешенного усреднения.
    Использует стандартные веса ITU-R BT.601: 0.299*R + 0.587*G + 0.114*B
    
    Аргументы:
        image_path (str): Путь к исходному цветному изображению
        output_path (str): Путь для сохранения полутонового изображения
        
    Возвращает:
        numpy.ndarray: Полутоновое изображение в виде 2D массива
    """
    # Открываем изображение и конвертируем в RGB для обеспечения 3 каналов
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img)
    
    # Применяем взвешенное усреднение согласно стандарту ITU-R BT.601
    grayscale = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
    grayscale = grayscale.astype(np.uint8)
    
    # Сохраняем полутоновое изображение в формате BMP
    Image.fromarray(grayscale).save(output_path)
    
    return grayscale

def singh_thresholding(image, window_size=7, k=0.5):
    """
    Адаптивная бинаризация Сингха для полутонового изображения.
    
    Алгоритм использует локальное среднее и стандартное отклонение в окне
    для определения порога бинаризации для каждого пикселя.
    
    Аргументы:
        image (numpy.ndarray): Полутоновое изображение в виде 2D массива
        window_size (int): Размер локального окна (должен быть нечетным)
        k (float): Константа Сингха, влияющая на чувствительность алгоритма
        
    Возвращает:
        numpy.ndarray: Бинарное изображение в виде 2D массива
    """
    print(f"Бинаризация Сингха с размером окна={window_size}, k={k}")
    
    # Получаем размеры изображения
    height, width = image.shape
    
    # Создаем выходное бинарное изображение
    binary_image = np.zeros_like(image)
    
    # Вычисляем глобальное стандартное отклонение для нормализации
    global_std = np.std(image)
    
    # Вычисляем половину размера окна для отступа
    half_window = window_size // 2
    
    # Дополняем изображение для обработки краевых пикселей
    padded_image = np.pad(image, half_window, mode='reflect')
    
    # Обрабатываем каждый пиксель изображения
    for y in range(height):
        for x in range(width):
            # Извлекаем локальное окно вокруг пикселя
            local_window = padded_image[y:y+window_size, x:x+window_size]
            
            # Вычисляем локальные статистики
            local_mean = np.mean(local_window)
            local_std = np.std(local_window)
            
            # Формула порога Сингха: 
            # T(x,y) = mean(x,y) * [1 + k * ((std(x,y)/global_std) - 1)]
            threshold = local_mean * (1 + k * ((local_std / global_std) - 1))
            
            # Применяем порог к пикселю
            binary_image[y, x] = 255 if image[y, x] > threshold else 0
    
    return binary_image

def process_image(image_path, output_dir):
    """
    Обрабатывает одно изображение: конвертирует в градации серого и затем бинаризует.
    
    Аргументы:
        image_path (str): Путь к исходному цветному изображению
        output_dir (str): Директория для сохранения результатов
    """
    # Извлекаем имя файла из пути
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Создаем отдельную директорию для каждого изображения
    image_output_dir = os.path.join(output_dir, filename)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Определяем пути для сохранения результатов
    gray_path = os.path.join(image_output_dir, f"{filename}_gray.bmp")
    binary_path = os.path.join(image_output_dir, f"{filename}_binary.bmp")
    
    # Конвертируем в градации серого
    gray_image = rgb_to_grayscale(image_path, gray_path)
    
    # Параметры алгоритма Сингха
    window_size = 7
    k = 0.5
    
    # Конвертируем в бинарное изображение методом Сингха
    binary_image = singh_thresholding(gray_image, window_size=window_size, k=k)
    Image.fromarray(binary_image).save(binary_path)
    
    # Сохраняем оригинальное изображение в директорию результатов
    original_copy_path = os.path.join(image_output_dir, f"{filename}_original.png")
    Image.open(image_path).save(original_copy_path)
    
    # Создаем и сохраняем сравнительное изображение
    comparison_path = os.path.join(image_output_dir, f"{filename}_comparison.png")
    create_comparison_image(image_path, gray_image, binary_image, comparison_path, window_size, k)
    
    return gray_image, binary_image

def create_comparison_image(original_path, gray_image, binary_image, output_path, window_size, k):
    """
    Создает сравнительное изображение с оригиналом, полутоновым и бинарным изображениями
    
    Аргументы:
        original_path (str): Путь к оригинальному изображению
        gray_image (numpy.ndarray): Полутоновое изображение
        binary_image (numpy.ndarray): Бинарное изображение
        output_path (str): Путь для сохранения сравнительного изображения
        window_size (int): Размер окна для алгоритма Сингха
        k (float): Константа Сингха
    """
    # Создаем фигуру с тремя подграфиками
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Показываем оригинальное изображение
    original_img = Image.open(original_path)
    axs[0].imshow(original_img)
    axs[0].set_title('Оригинал')
    axs[0].axis('off')
    
    # Показываем полутоновое изображение
    axs[1].imshow(gray_image, cmap='gray')
    axs[1].set_title('Градации серого')
    axs[1].axis('off')
    
    # Показываем бинарное изображение
    axs[2].imshow(binary_image, cmap='gray')
    axs[2].set_title(f'Бинарное (Сингх, окно={window_size}, k={k})')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Основная функция программы"""
    # Базовые директории
    input_dir = "images"
    output_dir = "results"
    
    # Создаем директории, если они не существуют
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Проверяем наличие изображений
    if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
        # Пытаемся скопировать изображения из lab1/src_img, если существует
        lab1_path = os.path.join("..", "lab1", "src_img")
        if os.path.exists(lab1_path):
            print(f"Директория {input_dir} пуста. Копирую тестовые изображения из {lab1_path}...")
            for img_file in os.listdir(lab1_path):
                if img_file.lower().endswith(('.png', '.bmp')):
                    import shutil
                    shutil.copy2(os.path.join(lab1_path, img_file), input_dir)
            print("Изображения скопированы.")
        else:
            print(f"Директория {input_dir} пуста. Добавьте изображения перед запуском обработки.")
            return
    
    # Список всех PNG и BMP изображений в исходной директории
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.bmp'))]
    
    if not images:
        print(f"В директории {input_dir} не найдены изображения (PNG, BMP).")
        return
    
    # Обрабатываем каждое изображение
    for image_file in images:
        image_path = os.path.join(input_dir, image_file)
        print(f"Обработка {image_file}...")
        process_image(image_path, output_dir)
    
    print("Все изображения успешно обработаны.")
    print(f"Результаты сохранены в директории {output_dir}")

if __name__ == "__main__":
    main() 