from ultralytics import YOLO

# 1. Загружаем ВАШУ обученную модель
# Путь может отличаться, проверьте папку runs/detect/train/weights
model = YOLO('runs/detect/train5/weights/best.pt')

# 2. Запускаем режим валидации на тестовых данных
metrics = model.val(
    data='/home/mikhail/Downloads/Cubes_classes.yolo26/data.yaml', 
    split='test'  # Указываем, что хотим использовать именно папку test
)

print(f"Точность (mAP50): {metrics.results_dict['metrics/mAP50(B)']:.3f}")

# Путь к папке с картинками, которые мы отложили для теста
test_images_path = '/home/mikhail/Downloads/Cubes_classes.yolo26/dataset_split/test/images'

# Запуск предсказания
results = model.predict(
    source=test_images_path,
    classes=[0, 1, 2, 4, 5],
    conf=0.25,      # Порог уверенности (от 0 до 1)
    save=True,      # Сохранить картинки с рамками
    project='runs/test_results', # Папка для сохранения
    name='cubes_check'
)

print(f"Результаты сохранены в: {results[0].save_dir}")