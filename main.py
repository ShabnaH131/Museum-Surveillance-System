import cv2
from ultralytics import YOLO
from nlp_alert import generate_alert
import numpy as np
import time

# 1. Load the model
# Ensure this path is correct and the model was trained on more than just one class
model = YOLO("runs/detect/object_removal_model3/weights/best.pt")
cap = cv2.VideoCapture(0)

# --- ULTRA-STABLE CONFIGURATION ---
STABLE_TIME_REQUIRED = 5.0    
MISSING_TIME_REQUIRED = 5.0   
DISTANCE_THRESHOLD = 80       
# Increased confidence to prevent "everything" being a notebook
CONFIDENCE_THRESHOLD = 0.6    
FLICKER_GRACE_FRAMES = 200    

# State Management
tracked_objects = {}
next_id = 0
alert_timer = 0
alert_message = ""
alert_queue = []

print(f"Model loaded. Detecting classes: {model.names}")

while True:
    current_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference with higher confidence to filter out false positives
    results = model(frame, conf=CONFIDENCE_THRESHOLD) 
    detected_this_frame = []

    for r in results:
        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                # STRENGTHENED FILTER: Only track if it's specifically a 'Notebook'
                if label.lower() == "notebook":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    detected_this_frame.append({
                        'label': label, 
                        'bbox': (x1, y1, x2, y2), 
                        'center': (cx, cy),
                        'conf': conf
                    })

    updated_ids = set()
    for det in detected_this_frame:
        matched_id = None
        for obj_id, data in tracked_objects.items():
            dist = np.hypot(det['center'][0] - data['center'][0], det['center'][1] - data['center'][1])
            if det['label'] == data['label'] and dist < DISTANCE_THRESHOLD:
                matched_id = obj_id
                break
        
        if matched_id is not None:
            tracked_objects[matched_id].update({
                'center': det['center'],
                'bbox': det['bbox'],
                'last_seen_at': current_time,
                'frames_missing': 0 
            })
            updated_ids.add(matched_id)
        else:
            tracked_objects[next_id] = {
                'label': det['label'],
                'center': det['center'],
                'bbox': det['bbox'],
                'first_seen_at': current_time,
                'last_seen_at': current_time,
                'frames_missing': 0,
                'is_monitored': False
            }
            updated_ids.add(next_id)
            next_id += 1

    # --- LOGIC: Stabilization and Removal ---
    for obj_id in list(tracked_objects.keys()):
        obj = tracked_objects[obj_id]

        if not obj['is_monitored']:
            if (current_time - obj['first_seen_at']) >= STABLE_TIME_REQUIRED:
                tracked_objects[obj_id]['is_monitored'] = True

        if obj_id not in updated_ids:
            tracked_objects[obj_id]['frames_missing'] += 1
            
            if tracked_objects[obj_id]['frames_missing'] > FLICKER_GRACE_FRAMES:
                time_really_missing = current_time - obj['last_seen_at']
                
                if obj['is_monitored']:
                    display_timer = round(MISSING_TIME_REQUIRED - (time_really_missing - (FLICKER_GRACE_FRAMES/20)), 1)
                    if display_timer > 0:
                        cv2.putText(frame, f"VERIFYING REMOVAL: {display_timer}s", (20, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                if time_really_missing >= (MISSING_TIME_REQUIRED + (FLICKER_GRACE_FRAMES/20)):
                    if obj['is_monitored']:
                        alert_queue.append(generate_alert(obj['label']))
                    del tracked_objects[obj_id]

    # --- DRAWING ---
    for obj_id, data in tracked_objects.items():
        if obj_id in updated_ids: 
            x1, y1, x2, y2 = data['bbox']
            color = (0, 255, 0) if not data['is_monitored'] else (255, 100, 0)
            status = "STABILIZING..." if not data['is_monitored'] else "SECURED"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{data['label']} {status}", (x1, y1-10), 0, 0.5, color, 2)

    # Manage Displayed Alert
    if alert_queue and alert_timer == 0:
        alert_message = alert_queue.pop(0)
        alert_timer = 180

    if alert_timer > 0:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 255), -1)
        cv2.putText(frame, f"ALERT: {alert_message}", (20, 50), 0, 0.8, (255, 255, 255), 2)
        alert_timer -= 1

    cv2.imshow("Security Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
