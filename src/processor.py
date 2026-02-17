import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from shapely.geometry import Point, Polygon, LineString

class TrafficAnalyzer:
    def __init__(self, model_path='yolov8n.pt', use_color_detection=False):
        self.model_path = model_path
        self.use_color_detection = use_color_detection
        self.model = None # Lazy load or pass in
        if not self.use_color_detection:
             # Load model if not passed? 
             # Better: assume model is loaded outside or load once
             self.model = YOLO(self.model_path)
             
        self.reset()

    def reset(self):
        self.track_history = defaultdict(lambda: [])
        self.queue_roi = None 
        self.stop_line = None 
        self.signal_state = "GREEN" 
        self.frame_count = 0
        self.violations = 0
        self.violated_ids = set()
        self.next_track_id = 0
        self.prev_centers = {} 

    def set_roi(self, points):
        """Points: list of (x, y) tuples"""
        if points:
            self.queue_roi = Polygon(points)

    def set_stop_line(self, line):
        """Line: list of two (x, y) tuples"""
        if line:
            self.stop_line = LineString(line)
            
    # ... (rest of methods) ...

    def update_signal(self):
        # Simulate signal: Green (0-150), Red (150-300), etc. (5s interval at 30fps)
        period = 300
        if (self.frame_count % period) < (period // 2):
            self.signal_state = "GREEN"
        else:
            self.signal_state = "RED"

    def detect_colored_rectangles(self, frame):
        """Fallback detection for synthetic videos using color segmentation"""
        detections = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect saturated colors (the synthetic cars are bright, colored rectangles)
        # Saturation > 100, Value > 100 filters out gray road
        mask = cv2.inRange(hsv, (0, 100, 100), (180, 255, 255))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                detections.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area
                })
        
        return detections

    def simple_track(self, detections):
        """Simple tracking by matching closest centers"""
        tracked = []
        used_prev = set()
        
        for det in detections:
            cx, cy = det['center']
            best_id = None
            best_dist = 100  # Max matching distance
            
            for prev_id, (px, py) in self.prev_centers.items():
                if prev_id in used_prev:
                    continue
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = prev_id
            
            if best_id is not None:
                used_prev.add(best_id)
                tracked.append((det, best_id))
            else:
                tracked.append((det, self.next_track_id))
                self.next_track_id += 1
        
        # Update prev_centers
        self.prev_centers = {tid: det['center'] for det, tid in tracked}
        
        return tracked

    def process_frame(self, frame):
        self.frame_count += 1
        self.update_signal()
        
        current_queue_length = 0
        annotated_frame = frame.copy()
        
        if self.use_color_detection:
            # Use color-based detection for synthetic videos
            detections = self.detect_colored_rectangles(frame)
            tracked = self.simple_track(detections)
            
            for det, track_id in tracked:
                x, y, w, h = det['bbox']
                cx, cy = det['center']
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID:{track_id}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                center = Point(cx, cy)
                
                # Update history
                track = self.track_history[track_id]
                track.append((float(cx), float(cy)))
                if len(track) > 30:
                    track.pop(0)
                
                # Queue Detection
                if self.queue_roi and self.queue_roi.contains(center):
                    current_queue_length += 1
                
                # Violation Detection
                if self.stop_line and self.signal_state == "RED":
                    if track_id not in self.violated_ids and len(track) >= 2:
                        path = LineString(track[-2:])
                        if path.intersects(self.stop_line):
                            self.violations += 1
                            self.violated_ids.add(track_id)
                            cv2.circle(annotated_frame, (cx, cy), 20, (0, 0, 255), -1)
        else:
            # Use YOLO detection for real videos
            results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            annotated_frame = results[0].plot()
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                    if class_id in [2, 3, 5, 7]:
                        x, y, w, h = box
                        center = Point(x, y)
                        
                        track = self.track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)

                        if self.queue_roi and self.queue_roi.contains(center):
                            current_queue_length += 1

                        if self.stop_line and self.signal_state == "RED":
                            if track_id not in self.violated_ids and len(track) >= 2:
                                path = LineString(track[-2:])
                                if path.intersects(self.stop_line):
                                    self.violations += 1
                                    self.violated_ids.add(track_id)
                                    cv2.circle(annotated_frame, (int(x), int(y)), 20, (0, 0, 255), -1)

        # Draw ROI
        if self.queue_roi:
            pts = np.array(self.queue_roi.exterior.coords, np.int32)
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.polylines(annotated_frame, [pts], True, (0, 255, 255), 2)
            
        # Draw Stop Line
        if self.stop_line:
            x1, y1 = int(self.stop_line.coords[0][0]), int(self.stop_line.coords[0][1])
            x2, y2 = int(self.stop_line.coords[1][0]), int(self.stop_line.coords[1][1])
            color = (0, 0, 255) if self.signal_state == "RED" else (0, 255, 0)
            cv2.line(annotated_frame, (x1, y1), (x2, y2), color, 4)

        # Overlay Info
        # Overlay Info - Top Left Summary
        cv2.putText(annotated_frame, f"Signal: {self.signal_state}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if self.signal_state == "RED" else (0, 255, 0), 2)
        
        # Draw Queue Count inside the ROI if it exists
        if self.queue_roi:
            # simple centroid approximation for text placement
            min_x, min_y, max_x, max_y = self.queue_roi.bounds
            center_x = int((min_x + max_x) // 2)
            center_y = int((min_y + max_y) // 2)
            cv2.putText(annotated_frame, f"Queue: {current_queue_length}", (center_x - 50, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # Draw Violation Count near the stop line if it exists
        if self.stop_line:
            # Place text near the start of the line
            x1, y1 = int(self.stop_line.coords[0][0]), int(self.stop_line.coords[0][1])
            cv2.putText(annotated_frame, f"Violations: {self.violations}", (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate Density
        density = 0
        if self.queue_roi:
            area = self.queue_roi.area
            if area > 0:
                density = (current_queue_length / area) * 10000 
        
        stats = {
            "Queue Length": current_queue_length,
            "Queue Density": round(density, 4),
            "Signal": self.signal_state,
            "Violations": self.violations
        }
        
        return annotated_frame, stats



    def process_video(self, video_path, skip_frames=2):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame Skipping
            if self.frame_count % (skip_frames + 1) != 0:
                self.frame_count += 1
                continue

            processed_frame, stats = self.process_frame(frame)
            yield processed_frame, stats
            
        cap.release()

    def analyze_traffic_pattern(self, video_path, sample_intervals=20):
        """
        Analyze video to determine active traffic area.
        sample_intervals: Number of frames to check distributed across the video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        all_boxes = []
        
        # Check frames distributed evenly
        stride = max(1, total_frames // sample_intervals)
        
        for frame_idx in range(0, total_frames, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection
            results = self.model(frame, verbose=False)
            
            if results[0].boxes:
                boxes = results[0].boxes.xywh.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for box, cls in zip(boxes, classes):
                    if int(cls) in [2, 3, 5, 7]: # Car, motorcycle, bus, truck
                        x, y, w, h = box
                        # Store corners
                        x1, y1 = x - w/2, y - h/2
                        x2, y2 = x + w/2, y + h/2
                        all_boxes.append((x1, y1, x2, y2))
                        
        cap.release()
        
        if not all_boxes:
            return None, None
            
        all_boxes = np.array(all_boxes)
        
        # Find global min/max with some margin
        min_x = max(0, int(np.min(all_boxes[:, 0])))
        min_y = max(0, int(np.min(all_boxes[:, 1])))
        max_x = min(width, int(np.max(all_boxes[:, 2])))
        max_y = min(height, int(np.max(all_boxes[:, 3])))
        
        # Define Queue ROI (The active traffic area)
        roi_points = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        
        # Define Stop Line (Placed at 70% of the ROI height)
        roi_h = max_y - min_y
        stop_y = int(min_y + roi_h * 0.7)
        
        stop_line_points = [(min_x, stop_y), (max_x, stop_y)]
        
        return roi_points, stop_line_points
