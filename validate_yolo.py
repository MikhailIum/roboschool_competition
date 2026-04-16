from ultralytics import YOLO
import os

# 1. Загружаем вашу последнюю обученную модель
# Проверьте путь, обычно это последний запуск в runs/detect/
model_path = 'runs/detect/train2/weights/best.pt' 
model = YOLO(model_path)

# 2. Путь к папке с новыми картинками
source_path = '/home/mikhail/Downloads/testing'

# 3. Запуск предсказания
results = model.predict(
    source=source_path,
    conf=0.3,       # Порог уверенности (отображать всё, что выше 30%)
    classes=[0, 1, 2, 4, 5],
    imgsz=800,      # Используйте тот же размер, что был при обучении
    save=True,      # Сохранить картинки с рамками на диск
    save_txt=True,  # Сохранить координаты в .txt (если нужно для робота)
    project='runs/testing_results', # Корневая папка для тестов
    name='my_test_run'              # Имя папки для этого конкретного запуска
)

print(f"Готово! Результаты сохранены в: runs/testing_results/my_test_run")