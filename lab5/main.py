from PIL import Image, ImageDraw, ImageFont 
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Создаем директорию для хранения изображений символов
output_folder = "./symbols"
shared_output_folder = "./../hebrew"
os.makedirs(output_folder, exist_ok=True)

# Путь к шрифту иврита
font_path = "./Hebrew.ttf"
if not os.path.exists(font_path):
    # Альтернативные пути к шрифтам
    font_paths = [
        "Hebrew.ttf",
        "./Hebrew.ttf",
    ]
    
    # Поиск доступного шрифта
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break
    
    if not os.path.exists(font_path):
        print("Не найден ни один подходящий шрифт с поддержкой иврита.")
        print("Установите шрифт с поддержкой иврита или укажите правильный путь.")
        sys.exit(1)

print(f"Используем шрифт: {font_path}")

# Параметры шрифта
font_size = 52
try:
    font = ImageFont.truetype(font_path, font_size)
except Exception as e:
    print(f"Ошибка при загрузке шрифта: {e}")
    sys.exit(1)

# Символы алфавита иврит в порядке, указанном в задании
hebrew_chars = "א ב ג ד ה ו ז ח ט י כ ך ל מ ם נ ן ס ע פ ף צ ץ ק ר ש ת אל"
# Удаляем пробелы для обработки только символов
hebrew_chars = hebrew_chars.replace(" ", "")

print(f"Начинаем генерацию изображений символов иврита...")

for char in hebrew_chars:
    # Создаем изображение с запасом по размеру
    image_size = (300, 300)  
    image = Image.new("L", image_size, 255)  # Белый фон, одноканальное (grayscale) изображение
    draw = ImageDraw.Draw(image)

    # Определяем размер текста
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Создаем новое изображение с размером, достаточным для текста, с небольшими полями
    image = Image.new("L", (text_width + 40, text_height + 40), 255)
    draw = ImageDraw.Draw(image)
    
    # Рисуем символ
    draw.text((20, 20), char, font=font, fill=0)  # Черный текст на белом фоне

    # Обрезаем белые поля
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    
    # Имя файла на основе кода символа и самого символа
    image_path = os.path.join(output_folder, f"char_{ord(char)}_{char}.png")
    image.save(image_path)

    # Сохраняем в общую папку
    shared_image_path = os.path.join(shared_output_folder, f"char_{ord(char)}_{char}.png")
    image.save(shared_image_path)

    print(f"Символ '{char}' сохранен как {image_path}")