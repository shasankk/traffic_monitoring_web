import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
import os
import requests
from ultralytics import YOLO

app = Flask(__name__)

# 1. Load YOLOv8s (Small) model for better bus/truck accuracy
model = YOLO("yolov8s.pt") 

# 2. Load OpenCV Haar Cascade for License Plate Recognition (ANPR)
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Global state for dashboard metrics
CURRENT_STATS = {
    "car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "plates": 0
}

def get_weather(city):
    try:
        url = f"https://wttr.in/{city}?format=%C+%t"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip()
    except:
        pass
    return "Weather data unavailable"

def generate_frames(video_path):
    global CURRENT_STATS
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    # Reset stats on new video start
    CURRENT_STATS = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "plates": 0}
    tracked_ids = set()
    plate_detections = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 Tracking
        # Setting conf=0.3 to improve detection of larger vehicles like buses
        results = model.track(frame, persist=True, classes=list(VEHICLE_CLASSES.keys()), conf=0.3, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = VEHICLE_CLASSES[cls_id]

                # Update counts uniquely per track ID
                if track_id not in tracked_ids:
                    tracked_ids.add(track_id)
                    CURRENT_STATS[label] += 1

                # Draw vehicle box
                color = (0, 255, 0)
                if label == 'bus': color = (255, 0, 255) # Magenta for bus
                elif label == 'truck': color = (0, 165, 255) # Orange for truck
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # ANPR (License Plate Detection) inside the vehicle bounding box
                # Extract vehicle ROI to look for plates
                roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                plates = plate_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 10))
                
                for (px, py, pw, ph) in plates:
                    # Draw plate box relative to original frame
                    px_frame, py_frame = x1 + px, y1 + py
                    cv2.rectangle(frame, (px_frame, py_frame), (px_frame + pw, py_frame + ph), (0, 255, 255), 2)
                    cv2.putText(frame, "PLATE", (px_frame, py_frame - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # Prevent over-counting plates (simplified: if plate detected on a tracked vehicle, count it once)
                    if track_id not in plate_detections:
                        plate_detections.add(track_id)
                        CURRENT_STATS["plates"] += 1

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return render_template("index.html", error="No video file uploaded.")
        
        video = request.files["video"]
        city = request.form.get("city", "Unknown").strip()

        if video.filename == "":
            return render_template("index.html", error="No video selected.")
        
        os.makedirs("static", exist_ok=True)
        video_path = os.path.join("static", "input_video.mp4")
        video.save(video_path)
        
        # Redirect to dashboard with city parameter
        return redirect(url_for('dashboard', city=city))
    
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    city = request.args.get("city", "Unknown")
    weather = get_weather(city)
    return render_template("dashboard.html", city=city, weather=weather)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(os.path.join("static", "input_video.mp4")),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(CURRENT_STATS)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)