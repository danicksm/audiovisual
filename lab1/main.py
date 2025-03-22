from PIL import Image
import numpy as np
from numpy.ma.core import arccos
import os
import sys

# Создаем директории, если они не существуют
os.makedirs('lab1/src_img', exist_ok=True)

# Список доступных изображений
AVAILABLE_IMAGES = ['berry', 'buisness', 'qiwi']

def process_image(filename):
    try:
        with Image.open(f'lab1/src_img/{filename}.png') as img:
            img.load()
    except FileNotFoundError:
        print(f"Файл 'lab1/src_img/{filename}.png' не найден.")
        return False

    # Создаем директорию для текущего изображения
    output_dir = f'lab1/new_img/{filename}'
    os.makedirs(output_dir, exist_ok=True)
    
    img = img.convert("RGB")
    image = np.array(img)
    
    print(f"Обработка изображения {filename}.png...")

    # 1.1 Выделение компонентов R, G, B
    print("Разделение RGB каналов...")
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Создание одноканальных RGB изображений
    r_colored = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=2)
    g_colored = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=2)
    b_colored = np.stack([np.zeros_like(b), np.zeros_like(b), b], axis=2)

    # Сохранение каналов как отдельные изображения
    Image.fromarray(r_colored.astype(np.uint8)).save(f'{output_dir}/red_channel.png')
    Image.fromarray(g_colored.astype(np.uint8)).save(f'{output_dir}/green_channel.png')
    Image.fromarray(b_colored.astype(np.uint8)).save(f'{output_dir}/blue_channel.png')

    # 1.2 Преобразование RGB -> HSI
    def rgb_to_hsi(image):
        # Нормализация RGB значений
        new_image = np.array(image) / 255.0
        r, g, b = new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2]

        # Вычисление интенсивности (I)
        i = (r + g + b) / 3.0

        # Вычисление насыщенности (S)
        min_rgb = np.minimum(np.minimum(r, g), b)
        s = 1.0 - (3.0 * min_rgb / (r + g + b + 1e-8))

        # Вычисление оттенка (H)
        numerator = 0.5 * (2.0 * r - g - b)
        denominator = np.sqrt(np.maximum((r - g) ** 2 + (r - b) * (g - b), 0.0)) + 1e-8
        theta = arccos(np.clip(numerator / denominator, -1.0, 1.0))
        h = np.where(b <= g, theta, 2.0 * np.pi - theta)
        h = h / (2.0 * np.pi)  # Нормализация в [0, 1]

        # Обрезка значений до [0, 1]
        h = np.clip(h, 0.0, 1.0)
        s = np.clip(s, 0.0, 1.0)
        i = np.clip(i, 0.0, 1.0)

        # Сохранение HSI изображения
        hsi_image = np.stack([h, s, i], axis=2) * 255.0
        hsi_image = Image.fromarray(hsi_image.astype('uint8'))
        hsi_image.save(f'{output_dir}/rgb_to_hsi.png')

        # Сохранение яркостной компоненты
        i_image = Image.fromarray((i * 255.0).astype('uint8'))
        i_image.save(f'{output_dir}/intensity.png')
        
        return h, s, i

    print("Преобразование из RGB в HSI...")
    h, s, i = rgb_to_hsi(img)

    # 1.3 Инвертирование яркостной компоненты
    def invert_intensity(src_img):
        # Нормализация RGB значений
        new_image = np.array(src_img) / 255.0
        r, g, b = new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2]
        
        # Вычисление интенсивности
        i = (r + g + b) / 3.0
        
        # Инвертирование интенсивности
        i_inv = 1.0 - i
        
        # Коррекция RGB каналов с учетом новой интенсивности
        r_new = r * (i_inv / (i + 1e-10))
        g_new = g * (i_inv / (i + 1e-10))
        b_new = b * (i_inv / (i + 1e-10))
        
        # Обрезка значений
        r_new = np.clip(r_new, 0.0, 1.0)
        g_new = np.clip(g_new, 0.0, 1.0)
        b_new = np.clip(b_new, 0.0, 1.0)
        
        # Сохранение изображения с инвертированной интенсивностью
        inv_img = np.stack([r_new, g_new, b_new], axis=2) * 255.0
        inv_img = Image.fromarray(inv_img.astype('uint8'))
        inv_img.save(f'{output_dir}/inverted_intensity.png')

    print("Инвертирование интенсивности...")
    invert_intensity(img)

    # Параметры передискретизации
    M = 2  # коэффициент увеличения
    N = 3  # коэффициент уменьшения
    K = M / N  # коэффициент комбинированной передискретизации

    # 2.1 Интерполяция изображения в M раз (билинейная)
    def bilinear_resize(src_image, m):
        src_width, src_height = src_image.size
        pixels = np.array(src_image)
        new_width, new_height = int(m * src_width), int(m * src_height)
        
        # Создание нового изображения
        new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
        # Билинейная интерполяция
        for y in range(new_height):
            for x in range(new_width):
                # Вычисление соответствующих координат в исходном изображении
                gx = (x / new_width) * src_width
                gy = (y / new_height) * src_height
                
                # Получение соседних пикселей
                x1, y1 = int(gx), int(gy)
                x2, y2 = min(x1 + 1, src_width - 1), min(y1 + 1, src_height - 1)
                
                # Вычисление весов
                dx = gx - x1
                dy = gy - y1
                
                # Получение значений пикселей
                f11 = pixels[y1, x1]
                f21 = pixels[y1, x2]
                f12 = pixels[y2, x1]
                f22 = pixels[y2, x2]
                
                # Интерполяция
                fx1 = (1 - dx) * f11 + dx * f21
                fx2 = (1 - dx) * f12 + dx * f22
                new_p = (1 - dy) * fx1 + dy * fx2
                
                new_image[y, x] = new_p
        
        return Image.fromarray(new_image)

    print(f"Выполнение билинейного растяжения (M={M})...")
    resized_img = bilinear_resize(img, M)
    resized_img.save(f'{output_dir}/resized_M_{M}.png')

    # 2.2 Децимация изображения в N раз (сжатие методом усреднения)
    def mean_resize(src_img, n):
        pixels = np.array(src_img)
        src_height, src_width, C = pixels.shape
        
        # Вычисление новых размеров
        new_width, new_height = src_width // n, src_height // n
        new_img = np.zeros((new_height, new_width, C), dtype=np.uint8)
        
        # Усреднение блоков пикселей
        for y in range(new_height):
            for x in range(new_width):
                y_start, y_end = y * n, min((y + 1) * n, src_height)
                x_start, x_end = x * n, min((x + 1) * n, src_width)
                
                block = pixels[y_start:y_end, x_start:x_end]
                
                for c in range(C):
                    new_img[y, x, c] = np.mean(block[:, :, c])
        
        return Image.fromarray(new_img)

    print(f"Выполнение сжатия методом усреднения (N={N})...")
    mean_img = mean_resize(img, N)
    mean_img.save(f'{output_dir}/resized_N_{N}.png')

    # 2.3 Двухпроходная передискретизация (растяжение и последующее сжатие)
    def two_pass_resampling(src_img, m, n):
        # Сначала интерполяция в M раз
        resized_img_m = bilinear_resize(src_img, m)
        
        # Затем децимация в N раз
        result_img = mean_resize(resized_img_m, n)
        
        # Сохранение результата
        result_img.save(f'{output_dir}/two_pass_resampling_M{m}_N{n}.png')
        
        return result_img

    print(f"Выполнение двухпроходной передискретизации (M={M}, N={N})...")
    two_pass_img = two_pass_resampling(img, M, N)

    # 2.4 Передискретизация изображения в K раз за один проход
    def one_pass_resampling(src_img, K):
        w, h = src_img.size
        
        # Вычисление новых размеров
        new_w = max(1, int(round(w * K)))
        new_h = max(1, int(round(h * K)))
        
        # Создание нового изображения
        new_img = Image.new(img.mode, (new_w, new_h))
        old_pixels = src_img.load()
        new_pixels = new_img.load()
        
        # Прямое отображение пикселей
        for y_new in range(new_h):
            for x_new in range(new_w):
                old_x = min(int(x_new / K), w - 1)
                old_y = min(int(y_new / K), h - 1)
                new_pixels[x_new, y_new] = old_pixels[old_x, old_y]
        
        # Сохранение результата
        new_img.save(f'{output_dir}/one_pass_resampling_K{K}.png')
        
        return new_img

    print(f"Выполнение однопроходной передискретизации (K={K})...")
    one_pass_img = one_pass_resampling(img, K)

    # Сохранение оригинального изображения в папку результатов
    img.save(f'{output_dir}/original.png')

    print(f"Обработка изображения {filename}.png завершена!")
    return True

def main():
    if len(sys.argv) > 1:
        # Если указан аргумент командной строки, обрабатываем только указанное изображение
        if sys.argv[1] in AVAILABLE_IMAGES:
            process_image(sys.argv[1])
        else:
            print(f"Неизвестное изображение: {sys.argv[1]}")
            print(f"Доступные изображения: {', '.join(AVAILABLE_IMAGES)}")
    else:
        # Обрабатываем все доступные изображения
        print("Доступные изображения:")
        for i, img_name in enumerate(AVAILABLE_IMAGES, 1):
            print(f"{i}. {img_name}")
        
        choice = input("Введите номер изображения для обработки (или 'all' для обработки всех): ")
        
        if choice.lower() == 'all':
            for img_name in AVAILABLE_IMAGES:
                process_image(img_name)
                print("=" * 50)
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(AVAILABLE_IMAGES):
                    process_image(AVAILABLE_IMAGES[index])
                else:
                    print("Неверный номер изображения")
            except ValueError:
                print("Некорректный ввод. Введите число или 'all'")

    print("Программа завершена.")

if __name__ == "__main__":
    main() 