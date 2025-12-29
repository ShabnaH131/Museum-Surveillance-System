import streamlit as st
import cv2
from ultralytics import YOLO
from datetime import datetime

st.set_page_config(page_title="Smart Object Removal Alert", layout="wide")
st.title("🚨 Smart Object Removal Detection System")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train6/weights/best.pt")

model = load_model()

conf = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
MISSING_THRESHOLD = st.slider("Frames before removal alert", 10, 100, 40)

# Tracking memory
tracked_objects = {}
alert_messages = []

start = st.checkbox("Start Camera")
frame_window = st.image([])

if start:
    cap = cv2.VideoCapture(0)

    while start:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        results = model.track(
            frame,
            persist=True,
            conf=conf,
            tracker="bytetrack.yaml"
        )

        current_ids = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, classes):
                obj_id = int(obj_id)
                current_ids.add(obj_id)

                object_name = model.names[int(cls)]

                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = {
                        "missing": 0,
                        "class": object_name,
                        "alerted": False
                    }
                else:
                    tracked_objects[obj_id]["missing"] = 0

        # Check for removed objects
        for obj_id in list(tracked_objects.keys()):
            if obj_id not in current_ids:
                tracked_objects[obj_id]["missing"] += 1

                if (
                    tracked_objects[obj_id]["missing"] >= MISSING_THRESHOLD
                    and not tracked_objects[obj_id]["alerted"]
                ):
                    object_name = tracked_objects[obj_id]["class"]
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    alert = f"⚠️ ALERT: {object_name} removed at {time_now}"
                    alert_messages.append(alert)

                    tracked_objects[obj_id]["alerted"] = True

        annotated_frame = results[0].plot()
        frame_window.image(
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    cap.release()

# Alert section
st.subheader("🚨 Removal Alerts")
if alert_messages:
    for msg in reversed(alert_messages):
        st.error(msg)
else:
    st.info("No removal alerts yet")
