A good README.md file should clearly explain the purpose of your project, how to install and use it, and any additional details that would help users understand and contribute to it. Below is a well-structured README file for your real-time object detection project using YOLOv8.

⸻

README.md for Real-Time Object Detection using YOLOv8

# Real-Time Object Detection using YOLOv8  

![YOLOv8 Object Detection](https://user-images.githubusercontent.com/your-image.png)  <!-- Replace with an actual image or GIF -->

## 📌 **Overview**
This project implements real-time object detection using the **YOLOv8 model** and **OpenCV**. The program captures live video from the webcam, processes each frame, and detects objects in real-time with bounding boxes and labels.

## 🛠 **Features**
- 🔹 **Live object detection** from webcam.
- 🔹 Uses **YOLOv8** (You Only Look Once) model for fast and accurate detection.
- 🔹 Automatically detects **GPU** (CUDA) support for better performance.
- 🔹 Supports **custom YOLO models** (Replace `yolov8n.pt` with your own model).
- 🔹 Press **'q'** to exit the program.

---

## 📥 **Installation**
1️⃣ Clone the Repository**


2️⃣ Install Dependencies

Ensure you have Python installed, then run:

pip install ultralytics opencv-python torch numpy

3️⃣ Run the Script

python object_detection.py



⸻

🎯 Usage
	•	The script will open your webcam and start detecting objects in real-time.
	•	Bounding boxes will be drawn around detected objects.
	•	Confidence scores and labels will be displayed.
	•	Press ‘q’ to close the program.

⸻

⚡ Customization

1️⃣ Use a Different YOLOv8 Model

Replace:

model = YOLO("yolov8n.pt")  # 'n' version is fast but less accurate

With:
	•	"yolov8s.pt" - Small version (more accurate)
	•	"yolov8m.pt" - Medium version (higher accuracy)

2️⃣ Change Webcam Source

By default, the script uses 0 (built-in webcam).
To use an external camera, change:

cap = cv2.VideoCapture(1)  # Use 1 or 2 for external webcams

3️⃣ Modify Frame Size for Performance

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

Lowering resolution (e.g., 320x240) improves speed.

⸻

🔥 Common Issues & Fixes

❌ 1. “ModuleNotFoundError: No module named ‘ultralytics’”

✅ Solution:

pip install ultralytics

❌ 2. “CUDA out of memory” (for GPU users)

✅ Solution:
Try running on CPU mode by modifying:

device = torch.device("cpu")

❌ 3. Webcam Not Opening

✅ Solution:
Try changing the camera index:

cap = cv2.VideoCapture(1)  # Try 1 or 2 if 0 doesn't work



⸻

📜 License

This project is licensed under the MIT License.
Feel free to use and modify it.

⸻

🤝 Contributing

Want to improve this project? Contributions are welcome!
	•	Fork the repository.
	•	Create a new branch (git checkout -b feature-name).
	•	Commit your changes (git commit -m "Added new feature").
	•	Push to your branch (git push origin feature-name).
	•	Open a Pull Request.

⸻

📬 Contact

For any queries, reach out via:
	•	📧 Email: adnaangouri58@gmail.com
	•	💬 GitHub Issues: Open an issue

⸻

🚀 Star this repo ⭐ if you found it useful!
Happy Coding! 🎉

---
