from ultralytics import YOLO

# Загружаем вашу обученную модель (.pt)
model = YOLO("runs/detect/train5/weights/best.pt")


# Экспортируем в формат ONNX
# dynamic=True позволяет менять размер входного изображения на лету
model.export(format="onnx", dynamic=True)