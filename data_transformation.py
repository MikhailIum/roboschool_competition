import os
import random
import shutil
from sklearn.model_selection import train_test_split

PATH = '/home/mikhail/Downloads/Cubes_classes.yolo26'

# Пути к вашим исходным данным
source_img = os.path.join(PATH, 'train/images')
source_lbl = os.path.join(PATH, 'train/labels')

# Куда сохранять
base_dir = os.path.join(PATH, 'dataset_split')
split_names = ['train', 'val', 'test']

# Создаем структуру папок
for split in split_names:
    os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

# Получаем список всех файлов (только имена без расширений)
files = [os.path.splitext(f)[0] for f in os.listdir(source_img) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Сначала отделяем 10% на тест
train_val, test = train_test_split(files, test_size=0.10, random_state=42)

# Из оставшихся берем 22.2% на валидацию (чтобы от общего объема это было 20%)
# 0.20 / (1 - 0.10) = 0.222
train, val = train_test_split(train_val, test_size=0.222, random_state=42)

def move_files(file_list, split_type):
    for f in file_list:
        # Переносим картинку (проверяем расширение)
        img_ext = '.png' if os.path.exists(os.path.join(source_img, f + '.png')) else '.jpg'
        shutil.copy(os.path.join(source_img, f + img_ext), os.path.join(base_dir, split_type, 'images'))
        # Переносим метку
        shutil.copy(os.path.join(source_lbl, f + '.txt'), os.path.join(base_dir, split_type, 'labels'))

# Раскладываем по полочкам
move_files(train, 'train')
move_files(val, 'val')
move_files(test, 'test')

print(f"Готово! Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")