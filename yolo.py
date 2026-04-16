from ultralytics import YOLO

# Загружаем модель
model = YOLO('yolo26s.pt') 

# Запускаем обучение с явными параметрами
results = model.train(
    data='/home/mikhail/Downloads/Cubes_classes.yolo26/data.yaml',
    epochs=100,
    imgsz=800,       # Увеличили с 640 до 800 (3050 Ti потянет при batch=8)
    batch=4,
    nbs=64,          # Номинальный размер батча для стабилизации градиента
    mosaic=1.0,      # Обязательно для мелких объектов
    mixup=0.1,       # Поможет, если кубики перекрывают друг друга
    device=0,
    lr0=0.01         # Начальная скорость обучения
)