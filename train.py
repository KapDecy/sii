from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt') 

    model.train(data='dataset/data.yaml', epochs=20, imgsz=640, device=0) 
    metrics = model.val()
    print(f"Maps: {metrics.box.map}") # Mean Average Precision

    success = model.export(format='onnx') # Опционально, для переносимости

if __name__ == '__main__':
    main()
