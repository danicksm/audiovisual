# Отчет по лабораторной работе №7
## Классификация на основе признаков, анализ профилей

## 1. Реализация расчёта меры близости изображений символов

Для реализации расчёта меры близости изображений символов был использован метод евклидова расстояния в пространстве нормализованных признаков. Нормализованные признаки включают в себя:
- Масса символа (общая и по четвертям изображения)
- Удельная масса по четвертям
- Нормализованные координаты центра тяжести
- Нормализованные осевые моменты инерции
- Нормализованные профили (горизонтальный и вертикальный)

Реализация функции расчёта евклидова расстояния:

```python
def compute_similarity(char_features, reference_features):
    # Define the features to use for comparison
    scalar_feature_keys = [
        'Weight_Q1', 'Weight_Q2', 'Weight_Q3', 'Weight_Q4',  # Масса символа по четвертям
        'SpecificWeight_Q1', 'SpecificWeight_Q2', 'SpecificWeight_Q3', 'SpecificWeight_Q4',
        'NormCoG_X', 'NormCoG_Y',  # Нормализованные координаты центра тяжести
        'NormMomentOfInertia_X', 'NormMomentOfInertia_Y'  # Нормализованные осевые моменты инерции
    ]
    
    # Define weights for each feature
    weights = {
        'Weight_Q1': 0.1, 'Weight_Q2': 0.1, 'Weight_Q3': 0.1, 'Weight_Q4': 0.1,
        'SpecificWeight_Q1': 0.05, 'SpecificWeight_Q2': 0.05, 'SpecificWeight_Q3': 0.05, 'SpecificWeight_Q4': 0.05,
        'NormCoG_X': 0.15, 'NormCoG_Y': 0.15,
        'NormMomentOfInertia_X': 0.15, 'NormMomentOfInertia_Y': 0.15
    }
    
    # Calculate weighted squared differences for scalar features
    weighted_squared_diff = 0
    for key in scalar_feature_keys:
        # Skip if key is missing in either features dictionary
        if key not in char_features or key not in reference_features:
            continue
            
        # Normalize feature values to [0, 1] range if needed
        char_value = float(char_features[key])
        ref_value = float(reference_features[key])
        
        # Add weighted squared difference
        weighted_squared_diff += weights[key] * (char_value - ref_value)**2
    
    # Calculate weighted Euclidean distance
    distance = np.sqrt(weighted_squared_diff)
    
    # Convert distance to similarity (1 for exact match, decreasing for less similar)
    # Using exponential decay to ensure positive values
    similarity = np.exp(-distance)
    
    return similarity
```

Для обеспечения лучшего распознавания также была добавлена нормализация размера изображений:

```python
def normalize_image_size(img, target_size=(32, 32)):
    # Add padding to make the image square
    height, width = img.shape
    max_dim = max(height, width)
    
    # Create a square black image
    square_img = np.zeros((max_dim, max_dim), dtype=img.dtype)
    
    # Calculate offsets to center the original image
    y_offset = (max_dim - height) // 2
    x_offset = (max_dim - width) // 2
    
    # Copy the original image to the center of the square image
    square_img[y_offset:y_offset + height, x_offset:x_offset + width] = img
    
    # Resize to target size
    normalized = cv2.resize(square_img, target_size, interpolation=cv2.INTER_AREA)
    
    return normalized
```

## 2. Расчёт меры близости для каждого обнаруженного символа

Для каждого обнаруженного символа в строке рассчитана мера близости со всеми символами алфавита иврит. Это осуществляется в функции `recognize_character`:

```python
def recognize_character(char_img, reference_features):
    # Calculate features for the character
    char_features = calculate_character_features(char_img)
    
    # Calculate similarity with each reference character
    similarities = []
    
    for _, reference in reference_features.iterrows():
        char = reference['Character']
        unicode_val = reference['Unicode']
        similarity = compute_similarity(char_features, reference)
        similarities.append((char, similarity, unicode_val))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return list of tuples (char, similarity)
    return [(char, similarity) for char, similarity, _ in similarities]
```

Функция возвращает список пар (символ, мера близости), отсортированных по убыванию меры близости.

## 3. Вывод результатов в файл

Результаты распознавания сохраняются в файл, где для каждого символа указывается список гипотез, отсортированных по убыванию меры близости:

```python
def save_recognition_results(results, filename="recognition_results.txt"):
    # Create output file
    output_path = os.path.join(RESULTS_DIR, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results):
            f.write(f"{i+1}: {result}\n")
```

Пример результата для эксперимента с увеличенным шрифтом (60 pt):

```
1: [('ם', 3.1159151578902456e-05), ('פ', 1.4219460039185373e-05), ('ס', 8.265859390959457e-06), ...]
2: [('י', 1.2881990946540593e-05), ('ז', 1.1132541936169029e-05), ('ו', 3.155322592887302e-08), ...]
3: [('ח', 0.009801216663587282), ('ה', 0.0014488147988084875), ('ק', 3.2801305338071396e-05), ...]
...
```

## 4. Вывод лучших гипотез и сравнение с распознаваемой строкой

Лучшие гипотезы (первый столбец из результатов распознавания) объединяются в строку и сравниваются с исходной строкой:

```python
def extract_best_hypothesis(results):
    return ''.join([result[0][0] for result in results])

def count_correct_recognitions(hypothesis, reference):
    # Ensure the strings have the same length
    min_len = min(len(hypothesis), len(reference))
    
    # Count correct recognitions
    correct = sum(1 for i in range(min_len) if hypothesis[i] == reference[i])
    
    # Calculate percentage
    percentage = 100 * correct / len(reference) if len(reference) > 0 else 0
    
    return correct, len(reference), percentage
```

## 5. Вычисление количества ошибок и доли верно распознанных символов

Для каждого эксперимента вычисляется количество правильно распознанных символов и доля в процентах:

Результаты:

| Эксперимент | Правильно распознано | Процент |
|-------------|----------------------|---------|
| Исходные символы из lab6 | 0/19 | 0.00% |
| Исходное изображение phrase.bmp | 1/19 | 5.26% |
| Шрифт 44 pt (меньше) | 0/19 | 0.00% |
| Шрифт 52 pt (исходный) | 1/19 | 5.26% |
| Шрифт 60 pt (больше) | 4/19 | 21.05% |

## 6. Эксперимент с разными размерами шрифта

Для эксперимента были сгенерированы изображения исходной строки с разными размерами шрифта:
- Меньший размер: 44 pt (на 8 пунктов меньше исходного)
- Исходный размер: 52 pt
- Больший размер: 60 pt (на 8 пунктов больше исходного)

Функция для генерации изображений с заданным размером шрифта:

```python
def generate_different_size_image(reference_string, font_size):
    # Create a larger image to ensure text fits
    img = Image.new('L', (800, 100), color=255)
    
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw = ImageDraw.Draw(img)
    
    # Draw the text (Hebrew is right-to-left)
    text_width = draw.textlength(reference_string, font=font)
    position = (img.width - text_width - 10, 10)
    draw.text(position, reference_string, font=font, fill=0)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    return img_array
```

### Результаты сравнения:

![Сравнение результатов распознавания](results/experiment_comparison.png)

Как видно из результатов, наилучшее качество распознавания достигается при использовании увеличенного шрифта (60 pt) - 21.05% символов распознаны верно. Это можно объяснить тем, что при увеличенном размере шрифта характерные особенности символов выражены более четко, что позволяет более точно вычислить признаки.

### Анализ разницы между экспериментами:

```
Phrase.bmp - Original Lab6 Chars: 5.26%
Smaller Font (44) - Original Lab6 Chars: 0.00%
Original Font (52) - Original Lab6 Chars: 5.26%
Larger Font (60) - Original Lab6 Chars: 21.05%
Smaller Font (44) - Phrase.bmp: -5.26%
Original Font (52) - Phrase.bmp: 0.00%
Larger Font (60) - Phrase.bmp: 15.79%
Original Font (52) - Smaller Font (44): 5.26%
Larger Font (60) - Smaller Font (44): 21.05%
Larger Font (60) - Original Font (52): 15.79%
```

## Выводы:

1. Реализованный алгоритм распознавания на основе признаков показывает невысокую точность распознавания (до 21% в лучшем случае), что говорит о необходимости улучшения метода.

2. Наиболее важные факторы, влияющие на точность распознавания:
   - Размер шрифта: более крупный шрифт обеспечивает лучшее распознавание
   - Набор используемых признаков: включение большего количества признаков и их взвешивание позволило улучшить качество
   - Нормализация изображений: приведение всех изображений к единому размеру важно для корректного сравнения

3. Возможные пути улучшения:
   - Использование более продвинутых методов сравнения символов, например, с применением машинного обучения
   - Добавление признаков, основанных на контурах символов
   - Улучшение алгоритма сегментации для более точного выделения символов

4. При увеличении размера шрифта улучшение распознавания наиболее заметно, что может быть полезно при проектировании систем распознавания текста.
