# 🎥 Live Object Detection & Tracking using YOLOv8 + Streamlit

A real-time AI web application that performs **object detection, tracking, counting, and alerting** using **YOLOv8**, Streamlit, OpenCV, and WebRTC webcam streaming.

---

## 🚀 Project Overview

This project is a **real-time computer vision system** that uses a webcam to detect and track objects live in the browser. It applies a YOLOv8 deep learning model to identify objects frame-by-frame and displays results with bounding boxes, labels, FPS, and object counts.

---

## ✨ Features

- 🎯 Real-time object detection using YOLOv8
- 📹 Live webcam streaming in browser (Streamlit WebRTC)
- 📦 Object counting per class (Person, Bottle, Cell Phone, etc.)
- ⚡ FPS (performance) monitoring
- 🚨 Smart alerts for specific objects
- 🧠 Bounding boxes with confidence scores
- 🎨 Clean UI with custom Streamlit design
- 📊 Live detection dashboard

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🌐
- streamlit-webrtc 🎥
- Ultralytics YOLOv8 🤖
- OpenCV 👁️
- PyAV (av)
- Collections (Counter)

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/live-object-detection.git
cd live-object-detection
2. Install dependencies
pip install streamlit
pip install streamlit-webrtc
pip install ultralytics
pip install opencv-python
pip install av
3. Run the application
streamlit run app.py
📁 Project Structure
live-object-detection/
│
├── app.py              # Main Streamlit application
├── README.md           # Project documentation
└── requirements.txt    # Dependencies (optional)
⚙️ How It Works
The webcam captures live video frames
Each frame is sent to the YOLOv8 model
YOLO detects objects in real-time
Bounding boxes + labels are drawn on the frame
Objects are counted per category
FPS and alerts are displayed
Output is streamed live in the browser
📊 Output Example
Person: 2
Bottle: 1
Cell Phone: 1
FPS: 18
ALERT: PERSON
🚨 Alerts System

The system triggers alerts when detecting:

👤 Person
📱 Cell Phone
🧴 Bottle
🎯 Key Functionalities

✔ Real-time detection
✔ Object tracking
✔ Class-wise counting
✔ FPS monitoring
✔ Alert system
✔ Web-based UI

🧠 Model Used
YOLOv8 (Ultralytics)
Lightweight model: yolov8s.pt
⚠️ Notes
Allow webcam access in browser
Good lighting improves detection accuracy
First run may download YOLO model automatically
Stable internet required for WebRTC connection
📸 Demo Screenshot (Optional)

Add your screenshots here:

/screenshots/demo1.png
/screenshots/demo2.png
👨‍💻 Author

Developer: Princess Ely Nicole O. Espares 
Developed for educational purposes in Computer Vision & AI learning

📜 License

This project is open-source and free for educational use.


---

If you want next upgrade, I can also generate:
- 📦 `requirements.txt`
- 🚀 GitHub-ready folder structure
- 🌐 Streamlit Cloud deployment guide
- 🎥 project demo script (for presentation grading)

Just tell me 👍