import tensorflow as tf
import fasttext
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, \
    decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
import pickle
import time


def get_vector(word, ft_model):
    return ft_model.get_word_vector(word)


TOP_N_TAGS = 5


def get_tuple(img_path, model, ft_model, trans):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array,
                                verbose=0)
    decoded_preds = decode_predictions(predictions, top=TOP_N_TAGS)[0]

    vectors_with_scores = []  # (вектор, вероятность)
    log_data = []

    for _, label, score in decoded_preds:
        score = float(score)
        if score > 0.2:
            rus_translation = trans.get(label, label)

            log_data.append({"tag": rus_translation, "probability": score})

            for word in rus_translation.split(' '):
                vec = get_vector(word, ft_model)
                vectors_with_scores.append((vec, score))

    return img_path, vectors_with_scores, log_data


def get_list_of_pair(t: tuple):
    res = []
    img_path, vectors_with_scores, _ = t
    for vec, score in vectors_with_scores:
        res.append((vec, img_path, score))
    return res


# Настройки путей
img_dir = './images/val2017'
translation_file = './translation.txt'

# Загрузка словаря
translate = {}
with open(translation_file, 'r', encoding='utf-8') as f:
    for line in f:
        if ' - ' in line:
            eng, rus = line.strip().split(' - ')
            eng = eng.replace(' ', '_')
            translate[eng] = rus

# Загрузка моделей
print("Загрузка моделей...")
model = ResNet50(weights='imagenet')
ft_model = fasttext.load_model('./model.bin')

all_features = []
tags_log = {}  # JSON лог

image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Начинаю обработку {len(image_files)} изображений...")
start_time = time.time()
cnt = 0

for img_p in image_files:
    try:
        t = get_tuple(img_p, model, ft_model, translate)

        current_img_path, vectors_with_scores, current_log_data = t

        all_features.extend(get_list_of_pair(t))

        file_name = os.path.basename(img_p)
        tags_log[file_name] = current_log_data

        cnt += 1
        if cnt % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"Обработано {cnt}/{len(image_files)}. Прошло {elapsed:.1f} сек.")

    except Exception as e:
        print(f"\nОшибка при обработке {img_p}: {e}")

end_time = time.time()
print(f"\nГотово! Обработка завершена за {end_time - start_time:.2f} сек.")

# Сохраняем индекс (векторы)
with open('image_index.pkl', 'wb') as f:
    pickle.dump(all_features, f)

# Сохраняем JSON лог с русскими тегами
with open('image_tags_log.json', 'w', encoding='utf-8') as f:
    json.dump(tags_log, f, ensure_ascii=False, indent=4)

print("Индекс сохранен в image_index.pkl")
print("Лог тегов сохранен в image_tags_log.json")