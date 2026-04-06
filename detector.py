from ultralytics import YOLO
from PIL import Image
import os

# Load the trained YOLOv8 model for clothing detection
model = YOLO("runs/detect/runs/fashion_detector/weights/best.pt")

def detect_and_crop(image_path, output_folder="data/output/crops"):
    """
    Receives an image path, detects clothing items using the YOLO model, and saves cropped images of each detected item.
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert to RGB to avoid issues with YOLO
    try:
        safe_path = os.path.splitext(image_path)[0] + "_safe.jpg"
        Image.open(image_path).convert("RGB").save(safe_path)
        image_path = safe_path
    except Exception as e:
        print(f"Error loading image: {e}")
        return []

    # Load the image and run detection
    image = Image.open(image_path)
    results = model(image_path)

    # Iterate over detected boxes and crop the clothing items
    crops = []
    for i, box in enumerate(results[0].boxes):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop the detected clothing item
        crop = image.crop((x1, y1, x2, y2))

        # Get label and confidence score
        label = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])

        # Only keep high confidence detections
        if confidence < 0.4:
            continue

        # Save the cropped image with a unique name
        crop_path = os.path.join(output_folder, f"crop_{i}_{label}.jpg")
        crop.save(crop_path)

        # Append crop info to the list
        crops.append({
            "path": crop_path,
            "label": label,
            "confidence": round(confidence, 2)
        })

        print(f"Detected: {label} ({confidence:.0%}) -> saved at {crop_path}")

    if os.path.exists(safe_path):
        os.remove(safe_path)

    print(f"\nTotal clothing items detected: {len(crops)}")
    return crops


# TEST
if __name__ == "__main__":
    crops = detect_and_crop("data/input_outfits/outfit_2.jpg")
    for crop in crops:
        print(crop)
