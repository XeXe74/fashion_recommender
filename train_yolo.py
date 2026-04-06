from ultralytics import YOLO
import torch

if __name__ == '__main__':

    # Check for CUDA availability for training
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load the pre-trained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model on our fashion dataset
    results = model.train(
        data="deepfashion2/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,        # bajado de 16 a 8
        workers=2,      # bajado de 8 a 2 (menos RAM)
        name="fashion_detector",
        project="runs"
    )

    print("Training completed.")
    print("Best model saved at: runs/detect/fashion_detector/weights/best.pt")
