import cv2
import torch
import os
import shutil
from datetime import datetime
from ultralytics import YOLO

# Suppress macOS warnings
os.environ["QT_LOGGING_RULES"] = "*.debug=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

# Configuration
FEEDBACK_DIR = "feedback_data"
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.5
RETRAIN_AFTER = 5

# Initialize feedback system
def setup_feedback():
    os.makedirs(f"{FEEDBACK_DIR}/images", exist_ok=True)
    os.makedirs(f"{FEEDBACK_DIR}/labels", exist_ok=True)
    print(f"Feedback system ready. Data will be saved in: {os.path.abspath(FEEDBACK_DIR)}")

# Initialize model
model = YOLO(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
setup_feedback()

# State management
current_detections = []
selected_box = -1
feedback_count = 0

# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global selected_box
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (x1, y1, x2, y2, *_) in enumerate(current_detections):
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_box = i
                break

def retrain_model():
    global model
    try:
        print("\n=== Starting retraining ===")
        model.train(
            data=f"{FEEDBACK_DIR}/dataset.yaml",
            epochs=10,
            imgsz=640,
            device="0" if torch.cuda.is_available() else "cpu",
            resume=True
        )
        model = YOLO("runs/detect/train/weights/best.pt")
        print("=== Model update successful! ===")
        return True
    except Exception as e:
        print(f"Retraining failed: {str(e)}")
        return False

# Initialize camera with AVFoundation backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS specific backend

for i in range(0,2):  # Try indices 0
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
else:
    print("Error: No camera detected at indices 0-3!")
    exit(1)

cv2.namedWindow("AI Annotation System")
cv2.setMouseCallback("AI Annotation System", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame! Trying to reconnect...")
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Reinitialize
        if not cap.isOpened():
            print("Permanent camera error!")
            break
        continue

    # Inference
    results = model(frame, conf=CONF_THRESH)
    current_detections = []
    annotated_frame = frame.copy()

    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            
            current_detections.append((x1, y1, x2, y2, cls, label, conf))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Handle feedback
    if selected_box != -1:
        if selected_box >= len(current_detections):
            print("Invalid box selection!")
            selected_box = -1
        else:
            x1, y1, x2, y2, cls, old_label, conf = current_detections[selected_box]
            print(f"\nCorrection requested for: {old_label} (confidence: {conf:.2f})")
            
            new_label = input("Enter correct label: ").strip()
            if new_label in model.names.values():
                # Save feedback data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                img_path = f"{FEEDBACK_DIR}/images/{timestamp}.jpg"
                label_path = f"{FEEDBACK_DIR}/labels/{timestamp}.txt"
                
                # Save image crop
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    print("Error: Empty image crop!")
                else:
                    cv2.imwrite(img_path, crop)
                    
                    # Create label file
                    new_cls = list(model.names.values()).index(new_label)
                    x_center = (x1 + x2) / 2 / frame.shape[1]
                    y_center = (y1 + y2) / 2 / frame.shape[0]
                    width = (x2 - x1) / frame.shape[1]
                    height = (y2 - y1) / frame.shape[0]
                    
                    with open(label_path, "w") as f:
                        f.write(f"{new_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                    feedback_count += 1
                    print(f"Feedback saved: {feedback_count}/{RETRAIN_AFTER}")
                    
                    # Trigger retraining
                    if feedback_count >= RETRAIN_AFTER:
                        if retrain_model():
                            feedback_count = 0
                            shutil.rmtree("runs/detect/train", ignore_errors=True)
            else:
                print(f"Invalid label! Available options: {list(model.names.values())}")
            
            selected_box = -1

    # Display
    cv2.imshow("AI Annotation System", annotated_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System shutdown.")