import cv2
import torch
import numpy as np
import ultralytics  # Install using: pip install ultralytics
from ultralytics import YOLO

# Load YOLOv8 model (Pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' version is the fastest, replace with 's' or 'm' for better accuracy

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 1 if using external webcam

# Set frame size for smoother performance
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Draw detections on frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = model.names[cls]  # Get class name

            # Draw bounding box
            color = (0, 255, 0)  # Green color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show frame in full screen
    cv2.imshow("Real-Time Object Detection (YOLOv8)", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()