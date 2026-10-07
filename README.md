# 🚦 AI-Powered Traffic Monitoring & Violation Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv3](https://img.shields.io/badge/YOLO-v3%20Darknet-FF6F00?style=for-the-badge&logo=target&logoColor=white)](https://pjreddie.com/darknet/yolo/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An intelligent, end-to-end computer vision and web-based traffic surveillance system. It performs real-time vehicle classification, multi-object tracking, traffic violation detection (red light running and wrong-way driving), automated flow analytics, and adaptive smart traffic signal simulation.

---

## 📌 Key Features

- **🚗 Multi-Class Vehicle Detection:** Detects and classifies cars, buses, trucks, and SUVs using the **YOLOv3 (Darknet)** deep neural network via OpenCV's DNN module.
- **🎯 Unique Object Tracking (Centroid Tracker):** Employs Euclidean distance matrices and frame disappearance buffers to track objects across video frames and prevent duplicate counts.
- **🚨 Automated Violation Detection:**
  - **Red Light Violations:** Identifies stationary vehicles encroaching on designated intersection/junction zones.
  - **Wrong-Way Driving:** Tracks vehicle trajectory vectors against expected lane flow directions.
- **🚥 Smart Traffic Signal Control:** Simulates dynamic traffic signal state switching (Green $\leftrightarrow$ Red) driven by real-time violation frequency thresholds.
- **📊 Automated Analytics & Graph Generation:** Generates time-series line charts for cumulative vehicle counts and violation trends over time using **Matplotlib**.
- **🌐 Interactive Web Dashboard:** Built with **Flask** and **Jinja2**, featuring asynchronous video upload processing, animated loading indicators, and visual result summaries.
- **🌤️ Integrated Weather Utility:** Includes a quick weather tracker querying real-time atmospheric data.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    A[📹 Video Upload / Input Stream] --> B[Frame Extraction & Preprocessing]
    B --> C[YOLOv3 Deep Neural Network Inference]
    C --> D[Non-Maximum Suppression (NMS)]
    D --> E[Centroid Tracker & ID Assignment]
    E --> F{Rule-Based Violation Engine}
    F -->|Direction Check| G[Wrong-Way Detection]
    F -->|Zone Check| H[Red-Light Violation]
    E --> I[Unique Vehicle Counter]
    G & H --> J[Adaptive Signal Controller]
    I & G & H --> K[Matplotlib Analytics Engine]
    K --> L[📊 Web Dashboard & Annotated Video Output]
```

---

## 📂 Project Structure

```plaintext
traffic_monitoring_web/
├── app.py                 # Core Flask application and video analytics pipeline
├── wt.py                  # Standalone weather tracking module
├── coco.names             # COCO dataset class labels
├── yolov3.cfg             # YOLOv3 network configuration file
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore configuration
├── templates/
│   ├── index.html         # Main dashboard template for upload & analytics
│   └── weather.html       # Weather tracker template
└── static/                # Static assets, uploads, and output graphs
    └── .gitkeep
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/shasankk/traffic_monitoring_web.git
cd traffic_monitoring_web
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download YOLOv3 Weights
The YOLOv3 pre-trained weights file (`yolov3.weights`, ~237 MB) is required for deep learning inference. Download it and place it in the root project directory:

- **Direct Download Link:** [pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

Or download via terminal:
```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://pjreddie.com/media/files/yolov3.weights" -OutFile "yolov3.weights"

# Linux / macOS (curl / wget)
curl -O https://pjreddie.com/media/files/yolov3.weights
# or
wget https://pjreddie.com/media/files/yolov3.weights
```

---

## 🚀 Running the Application

1. **Start the Flask Web Server:**
   ```bash
   python app.py
   ```
2. **Access the Web Dashboard:**
   Open your browser and navigate to `http://localhost:5000` (or `http://127.0.0.1:5000`).

3. **Analyze Video:**
   - Upload any traffic surveillance video (`.mp4`, `.avi`, `.mov`).
   - Click **Analyze Video**.
   - Review live frame tracking, vehicle totals, violation counts, and generated flow graphs.

---

## 🔬 Core Algorithms & Implementation Details

| Component | Technique / Algorithm | Purpose |
| :--- | :--- | :--- |
| **Object Detection** | YOLOv3 (Darknet) + NMS | Identifies 80 COCO classes, filtered for vehicles (`car`, `bus`, `truck`, `suv`). |
| **Object Tracking** | Centroid Tracking (Euclidean Distance) | Associates centroids between consecutive frames; assigns persistent IDs. |
| **Motion Vectoring** | Displacement vector $\Delta = (x_t - x_{t-1}, y_t - y_{t-1})$ | Flags movement vectors opposite to standard lane heading (Wrong-way). |
| **Spatial Incursion** | Bounding Box in ROI $+ \|\vec{v}\| \approx 0$ | Detects stopped vehicles obstructing intersections during red phase. |
| **Signal Adaptation** | Sliding Window Trigger | Changes signal phase if violation count exceeds threshold within rolling window. |

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, Flask
- **Computer Vision:** OpenCV (`cv2.dnn`), YOLOv3 Darknet
- **Data Science & Visualization:** NumPy, Matplotlib
- **Frontend:** HTML5, CSS3 (Modern Responsive UI), JavaScript
- **APIs:** REST API (`wttr.in`)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.