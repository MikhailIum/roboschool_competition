import os
import cv2
import numpy as np
import onnxruntime as ort

# ================= CONFIGURATION =================
model_path = 'weights/detector_small.onnx'
source_path = '/home/mikhail/Downloads/testing'
output_path = 'runs/onnx_test_results'
img_size = 800  # Как в вашем примере
conf_threshold = 0.3
target_classes = [0, 1, 2, 4, 5] # Классы, которые хотим видеть
# =================================================

def main():
    # 1. Создаем сессию и папки
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Если есть GPU, смените на ['CUDAExecutionProvider']
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Получаем имена файлов
    images = [f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Найдено изображений: {len(images)}")

    for img_name in images:
        full_path = os.path.join(source_path, img_name)
        orig_image = cv2.imread(full_path)
        if orig_image is None: continue
        
        h_orig, w_orig = orig_image.shape[:2]

        # 2. Препроцессинг
        # YOLO ожидает RGB и определенный размер
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (img_size, img_size))
        
        input_data = image_resized.transpose(2, 0, 1)  # HWC -> CHW
        input_data = input_data.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0) # Add batch dim

        # 3. Инференс
        outputs = session.run(None, {input_name: input_data})
        output = outputs[0] # Обычно это [1, 84, 8400] или [1, 300, 6]

        # 4. Постпроцессинг (универсальный для YOLOv8-v11 и далее)
        # Если форма [1, 84, 8400], нужно транспонировать
        if output.shape[1] > output.shape[2]:
            predictions = output[0]
        else:
            predictions = output[0].T 

        boxes, scores, class_ids = [], [], []

        for pred in predictions:
            # Формат YOLO обычно: [x, y, w, h, class0_score, class1_score, ...]
            # Или [x1, y1, x2, y2, confidence, class_id]
            
            if len(pred) > 6: # Формат с множеством вероятностей классов
                cls_scores = pred[4:]
                score = np.max(cls_scores)
                cls_id = np.argmax(cls_scores)
                # Координаты cx, cy, w, h
                cx, cy, w, h = pred[:4]
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
            else: # Формат End-to-End [x1, y1, x2, y2, conf, cls]
                x1, y1, x2, y2, score, cls_id = pred
            
            if score > conf_threshold and int(cls_id) in target_classes:
                # Масштабируем обратно к оригиналу
                x1 = int(x1 * w_orig / img_size)
                y1 = int(y1 * h_orig / img_size)
                x2 = int(x2 * w_orig / img_size)
                y2 = int(y2 * h_orig / img_size)
                
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(score))
                class_ids.append(int(cls_id))

        # 5. NMS (Удаление дубликатов рамок)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)

        # 6. Отрисовка
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"ID:{class_ids[i]} {scores[i]:.2f}"
                cv2.putText(orig_image, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 7. Сохранение
        save_path = os.path.join(output_path, img_name)
        cv2.imwrite(save_path, orig_image)
        print(f"Сохранено: {save_path}")

if __name__ == "__main__":
    main()