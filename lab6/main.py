import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Создаем директорию для результатов
output_dir = "lab6/results"
os.makedirs(output_dir, exist_ok=True)

# Путь к изображению
img_path = "lab6/images/Твои глаза самые красивые - с фоном.bmp"

# Функция для обработки изображения
def process_image(img_path):
    # Загрузка изображения - используем OpenCV только для загрузки
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Файл не найден: {img_path}")
    
    # Сохраняем оригинальное изображение
    cv2.imwrite(os.path.join(output_dir, "original.png"), img)
    
    # Бинаризация с использованием NumPy вместо OpenCV
    binary_threshold = 128
    img_bin = np.zeros_like(img)
    img_bin[img < binary_threshold] = 255  # Инверсная бинаризация (текст темный на светлом фоне)
    
    # Задание 2: Построение горизонтального и вертикального профилей
    horizontal_profile = np.sum(img_bin, axis=1)
    vertical_profile = np.sum(img_bin, axis=0)
    
    # Сохранение профилей
    plt.figure(figsize=(12, 4))
    plt.plot(horizontal_profile)
    plt.title('Горизонтальный профиль изображения')
    plt.xlabel('Строка')
    plt.ylabel('Сумма пикселей')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "horizontal_profile.png"))
    plt.close()
    
    plt.figure(figsize=(12, 4))
    plt.plot(vertical_profile)
    plt.title('Вертикальный профиль изображения')
    plt.xlabel('Столбец')
    plt.ylabel('Сумма пикселей')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "vertical_profile.png"))
    plt.close()
    
    # Задание 3: Сегментация символов на основе профилей с прореживанием
    def find_intervals(profile, threshold=5):
        intervals = []
        start = None
        for i, value in enumerate(profile):
            if value > threshold and start is None:
                start = i
            elif value <= threshold and start is not None:
                # Отфильтровываем слишком маленькие интервалы
                if i - start >= 2:
                    intervals.append((start, i))
                start = None
        if start is not None and len(profile) - start >= 2:
            intervals.append((start, len(profile)))
        return intervals
    
    # Обнаружение строк текста
    lines = find_intervals(horizontal_profile)
    print(f"Обнаружено строк: {len(lines)}")
    
    # Создаем цветное изображение для отображения сегментации с использованием NumPy
    img_segmented = np.stack([img, img, img], axis=2)  # Преобразуем в RGB с помощью NumPy
    
    # Массив для хранения координат всех символов
    all_chars = []
    
    # Для каждой строки находим символы
    for line_idx, (y1, y2) in enumerate(lines):
        line_img = img_bin[y1:y2, :]
        vertical_profile_line = np.sum(line_img, axis=0)
        chars = find_intervals(vertical_profile_line)
        
        print(f"В строке {line_idx+1} обнаружено символов: {len(chars)}")
        
        # Проверяем найденные символы
        for x1, x2 in chars:
            # Проверяем размер и отношение сторон
            width = x2 - x1
            height = y2 - y1
            
            # Отфильтровываем слишком маленькие области
            if width >= 3 and height >= 3:
                # Добавляем координаты в общий список
                all_chars.append((x1, y1, x2, y2))
                # Рисуем прямоугольник вокруг символа - реализуем через NumPy
                # Вертикальные линии
                img_segmented[y1:y2, x1:x1+1, 0] = 0    # R = 0
                img_segmented[y1:y2, x1:x1+1, 1] = 0    # G = 0
                img_segmented[y1:y2, x1:x1+1, 2] = 255  # B = 255
                
                img_segmented[y1:y2, x2:x2+1, 0] = 0
                img_segmented[y1:y2, x2:x2+1, 1] = 0
                img_segmented[y1:y2, x2:x2+1, 2] = 255
                
                # Горизонтальные линии
                img_segmented[y1:y1+1, x1:x2+1, 0] = 0
                img_segmented[y1:y1+1, x1:x2+1, 1] = 0
                img_segmented[y1:y1+1, x1:x2+1, 2] = 255
                
                img_segmented[y2:y2+1, x1:x2+1, 0] = 0
                img_segmented[y2:y2+1, x1:x2+1, 1] = 0
                img_segmented[y2:y2+1, x1:x2+1, 2] = 255
    
    # Сохраняем изображение с обведенными символами
    cv2.imwrite(os.path.join(output_dir, "segmented.png"), img_segmented)
    
    # Задание 4: Построение профилей символов
    if all_chars:
        # Выбираем первый символ для примера
        x1, y1, x2, y2 = all_chars[0]
        char_img = img_bin[y1:y2, x1:x2]
        
        # Сохраняем вырезанный символ
        cv2.imwrite(os.path.join(output_dir, "char_sample.png"), char_img)
        
        # Строим профили символа
        char_horizontal_profile = np.sum(char_img, axis=1)
        char_vertical_profile = np.sum(char_img, axis=0)
        
        plt.figure(figsize=(8, 4))
        plt.plot(char_horizontal_profile)
        plt.title('Горизонтальный профиль символа')
        plt.xlabel('Строка')
        plt.ylabel('Сумма пикселей')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "char_horizontal_profile.png"))
        plt.close()
        
        plt.figure(figsize=(8, 4))
        plt.plot(char_vertical_profile)
        plt.title('Вертикальный профиль символа')
        plt.xlabel('Столбец')
        plt.ylabel('Сумма пикселей')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "char_vertical_profile.png"))
        plt.close()
    
    # Задание 5: Выделение обрамляющего прямоугольника для всего текста
    ys, xs = np.nonzero(img_bin)
    if len(ys) > 0 and len(xs) > 0:
        y_min, y_max = np.min(ys), np.max(ys)
        x_min, x_max = np.min(xs), np.max(xs)
        
        # Создаем копию изображения для обрамляющего прямоугольника
        img_bounding = img_segmented.copy()
        
        # Рисуем зеленый прямоугольник с помощью NumPy
        # Вертикальные линии
        img_bounding[y_min:y_max+1, x_min:x_min+2, 0] = 0     # R = 0
        img_bounding[y_min:y_max+1, x_min:x_min+2, 1] = 255   # G = 255
        img_bounding[y_min:y_max+1, x_min:x_min+2, 2] = 0     # B = 0
        
        img_bounding[y_min:y_max+1, x_max:x_max+2, 0] = 0
        img_bounding[y_min:y_max+1, x_max:x_max+2, 1] = 255
        img_bounding[y_min:y_max+1, x_max:x_max+2, 2] = 0
        
        # Горизонтальные линии
        img_bounding[y_min:y_min+2, x_min:x_max+2, 0] = 0
        img_bounding[y_min:y_min+2, x_min:x_max+2, 1] = 255
        img_bounding[y_min:y_min+2, x_min:x_max+2, 2] = 0
        
        img_bounding[y_max:y_max+2, x_min:x_max+2, 0] = 0
        img_bounding[y_max:y_max+2, x_min:x_max+2, 1] = 255
        img_bounding[y_max:y_max+2, x_min:x_max+2, 2] = 0
        
        # Сохраняем изображение с обрамляющим прямоугольником
        cv2.imwrite(os.path.join(output_dir, "bounding_rect.png"), img_bounding)
    
    # Возвращаем координаты всех обнаруженных символов
    return all_chars

# Обрабатываем изображение
print("Обработка изображения...")
chars = process_image(img_path)
print(f"Обнаружено {len(chars)} символов на изображении.")

print("✅ Лабораторная работа №6 выполнена. Результаты сохранены в директории:", output_dir)
