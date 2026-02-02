from ultralytics import YOLO

def main():
    # 1. Загружаем предобученную модель (transfer learning)
    # Она уже умеет выделять фичи, мы просто доучим её под ракурсы парковки
    model = YOLO('yolov8n.pt') 

    # 2. Запускаем обучение
    # data='data.yaml' -> путь к твоему конфигу
    # epochs=20 -> для практики хватит за глаза (можно 50, если есть GPU)
    # imgsz=640 -> размер входного изображения
    model.train(data='dataset/data.yaml', epochs=20, imgsz=640, device=0) 
    # Примечание: Если у тебя Mac M1/M2/M3, используй device='mps'. 
    # Если Nvidia GPU + CUDA, используй device=0. Если ничего нет — 'cpu'.

    # 3. Валидация (проверка качества)
    metrics = model.val()
    print(f"Maps: {metrics.box.map}") # Mean Average Precision

    # 4. Сохраняем модель (хотя YOLO делает это сам в папку runs/)
    # Лучшая модель будет лежать по пути: runs/detect/train/weights/best.pt
    success = model.export(format='onnx') # Опционально, для переносимости

if __name__ == '__main__':
    main()
