import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import matplotlib.pyplot as plt
from collections import defaultdict

app = Flask(__name__)

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Consider using YOLOv4 for better accuracy
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detect objects in a frame
def detect_objects(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, height, width

# Get bounding boxes, confidences, and class IDs
def get_box_dimensions(outs, height, width, classes):
    class_ids = []
    confidences = []
    boxes = []
    centroids = []
    vehicle_classes = {"car", "bus", "truck", "suv"}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            label = classes[class_id]
            if confidence > 0.6 and label in vehicle_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centroids.append((center_x, center_y))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, indices, class_ids, confidences, centroids

# Simple centroid tracker class
class CentroidTracker:
    def __init__(self, max_disappeared=5, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store object IDs and their centroids
        self.disappeared = {}  # Track how many frames an object has been missing
        self.max_disappeared = max_disappeared  # Max frames before deregistering
        self.max_distance = max_distance  # Max distance to consider same object

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance between each pair of object centroids and new centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(centroids), axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(object_centroids))) - used_rows
            unused_cols = set(range(len(centroids))) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(centroids[col])

        return self.objects

# Process frame and count vehicles only once, with violation detection
def process_frame(boxes, indices, class_ids, classes, confidences, centroids, frame, tracker, counts, prev_positions, red_light_zone=(200, 400, 100, 300), expected_direction="right"):
    current_centroids = []
    violations = {}  # Store violations per object ID

    for i in indices:
        i = i  # Fix for OpenCV 4.5.5+
        box = boxes[i]
        x, y, w, h = box
        label = classes[class_ids[i]]
        confidence = confidences[i]
        centroid = centroids[i]
        current_centroids.append(centroid)

    # Update tracker with current centroids
    objects = tracker.update(current_centroids)

    # Check for violations
    for object_id, centroid in objects.items():
        if object_id in prev_positions:
            prev_centroid = prev_positions[object_id]
            movement = np.array(centroid) - np.array(prev_centroid)
            movement_dist = np.linalg.norm(movement)

            # Red light violation: Check if vehicle is stationary in red light zone
            rx, rx_end, ry, ry_end = red_light_zone
            if rx <= centroid[0] <= rx_end and ry <= centroid[1] <= ry_end and movement_dist < 5:  # Stationary
                violations[object_id] = "Red Light Violation"

            # Wrong-way driving: Check if movement direction is opposite to expected
            if expected_direction == "right" and movement[0] < -10:  # Moving left in a right-moving lane
                violations[object_id] = "Wrong-Way Driving"
            elif expected_direction == "left" and movement[0] > 10:  # Moving right in a left-moving lane
                violations[object_id] = "Wrong-Way Driving"

        prev_positions[object_id] = centroid  # Update previous position

        # Count only newly registered vehicles
        if object_id not in counts["tracked_ids"]:
            for i, c in enumerate(centroids):
                if np.linalg.norm(np.array(c) - np.array(centroid)) < 5:  # Match centroid to class
                    label = classes[class_ids[i]]
                    if label == "car":
                        counts["cars"] += 1
                    elif label == "bus":
                        counts["buses"] += 1
                    elif label == "truck":
                        counts["trucks"] += 1
                    elif label == "suv":
                        counts["suvs"] += 1
                    counts["tracked_ids"].add(object_id)
                    print(f"Counted: {label}, Total: {counts}")
                    break

    # Draw bounding boxes and labels
    for i in indices:
        i = i
        box = boxes[i]
        x, y, w, h = box
        label = classes[class_ids[i]]
        confidence = confidences[i]
        centroid = centroids[i]
        # Find the object ID for this centroid
        object_id = next((oid for oid, c in objects.items() if np.linalg.norm(np.array(c) - np.array(centroid)) < 5), None)
        
        if object_id in violations:
            color = (0, 0, 255)  # Red for violations
            violation_text = violations[object_id]
            cv2.putText(frame, f"{label} {confidence:.2f} - {violation_text}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (0, 255, 0)  # Green for no violations
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

# Generate graphs for analysis
def generate_graphs(counts_over_time):
    timestamps = list(range(len(counts_over_time["cars"])))
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, counts_over_time["cars"], label="Cars", marker="o")
    plt.plot(timestamps, counts_over_time["buses"], label="Buses", marker="o")
    plt.plot(timestamps, counts_over_time["trucks"], label="Trucks", marker="o")
    plt.plot(timestamps, counts_over_time["suvs"], label="SUVs", marker="o")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Vehicle Count")
    plt.title("Vehicle Count Over Time")
    plt.legend()
    plt.grid()
    graph_path = os.path.join("static", "vehicle_count_graph.png")
    plt.savefig(graph_path)
    plt.close()
    return graph_path

# Integration with Smart Traffic Signals
def smart_traffic_signal_control(violations_over_time, frame):
    # Simulate traffic signal control based on recent violations
    # In a real system, this would interface with actual traffic signal hardware
    # Here, we overlay the traffic light status on the video frame
    if len(violations_over_time) >= 5 and sum(violations_over_time[-5:]) > 5:  # More than 5 violations in last 5 frames
        # Change to red light
        cv2.putText(frame, "Traffic Light: RED", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red text
        return "RED"
    else:
        # Stay green or change to green
        cv2.putText(frame, "Traffic Light: GREEN", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green text
        return "GREEN"

# Process video and return counts
def process_video(video_path):
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return False, "Error: Could not open video."

    counts = {"cars": 0, "buses": 0, "trucks": 0, "suvs": 0, "tracked_ids": set()}
    counts_over_time = {"cars": [], "buses": [], "trucks": [], "suvs": []}
    violations_over_time = []  # Store violations for reporting
    traffic_light_states = []  # Store traffic light states over time
    tracker = CentroidTracker(max_disappeared=5, max_distance=50)
    prev_positions = {}  # Track previous positions for movement detection
    output_path = os.path.join("static", "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    frame_count = 0
    skip_frames = 2  # Process every 3rd frame to reduce load

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        outs, height, width = detect_objects(frame, net, output_layers)
        boxes, indices, class_ids, confidences, centroids = get_box_dimensions(outs, height, width, classes)
        frame = process_frame(boxes, indices, class_ids, classes, confidences, centroids, frame, tracker, counts, prev_positions)

        # Draw red light zone for visualization (optional)
        cv2.rectangle(frame, (200, 100), (400, 300), (255, 0, 0), 1)  # Blue rectangle for red light zone

        # Update counts over time
        counts_over_time["cars"].append(counts["cars"])
        counts_over_time["buses"].append(counts["buses"])
        counts_over_time["trucks"].append(counts["trucks"])
        counts_over_time["suvs"].append(counts["suvs"])

        # Track violations
        current_violations = sum(1 for oid in tracker.objects.keys() if oid in prev_positions and (
            (200 <= tracker.objects[oid][0] <= 400 and 100 <= tracker.objects[oid][1] <= 300 and 
             np.linalg.norm(np.array(tracker.objects[oid]) - np.array(prev_positions[oid])) < 5) or
            (np.array(tracker.objects[oid])[0] - np.array(prev_positions[oid])[0] < -10)
        ))
        violations_over_time.append(current_violations)

        # Integrate smart traffic signal control
        traffic_light_state = smart_traffic_signal_control(violations_over_time, frame)
        traffic_light_states.append(traffic_light_state)

        cv2.imshow("Vehicle Detection & Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Final counts: {counts}")
    print(f"Total violations detected: {sum(violations_over_time)}")
    print(f"Traffic light states recorded: {traffic_light_states[-5:]} (last 5 frames)")

    # Generate graphs
    graph_path = generate_graphs(counts_over_time)

    # Generate violation graph
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(violations_over_time))), violations_over_time, label="Violations", marker="o", color="red")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Violation Count")
    plt.title("Violations Over Time")
    plt.legend()
    plt.grid()
    violation_graph_path = os.path.join("static", "violation_graph.png")
    plt.savefig(violation_graph_path)
    plt.close()

    return True, counts, graph_path, violation_graph_path, sum(violations_over_time)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return render_template("index.html", error="No video file uploaded.")
        
        video = request.files["video"]
        if video.filename == "":
            return render_template("index.html", error="No video selected.")
        
        video_path = os.path.join("static", video.filename)
        video.save(video_path)
        
        success, result, graph_path, violation_graph_path, total_violations = process_video(video_path)
        if not success:
            return render_template("index.html", error=result)
        
        graph_filename = os.path.basename(graph_path)
        violation_graph_filename = os.path.basename(violation_graph_path)
        
        return render_template("index.html", counts=result, 
                              graph_filename=graph_filename, violation_graph_filename=violation_graph_filename,
                              total_violations=total_violations)
    
    return render_template("index.html")

@app.route("/static/<filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True, host="0.0.0.0", port=5000)