from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load YOLOv8 model
model = YOLO("C:/Users/harsh/OneDrive/Desktop/project/New folder/yolo_model/content/runs/detect/train5/weights/best.torchscript")

def detect_with_yolo(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Predict with YOLO model
    results = model(img_array)

    # Prepare for drawing
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=55)
    except:
        font = ImageFont.load_default()

    boxes = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            box = box.tolist()
            cls_id = int(cls.item())
            label = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)

            # Draw bounding box
            draw.rectangle(box, outline="red", width=3)

            # Compute text size properly
            if hasattr(draw, "textbbox"):
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            else:
                text_width, text_height = font.getsize(label)

            # Draw background for text
            draw.rectangle([box[0], box[1] - text_height, box[0] + text_width, box[1]], fill="red")
            # Draw text label
            draw.text((box[0], box[1] - text_height), label, fill="white", font=font)

            boxes.append(box)

    return image, boxes
