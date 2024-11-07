import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import tempfile 

# --- Streamlit UI ---
st.title("Live Deepfake Detection App")

# Sidebar options
source = st.sidebar.radio("Source", ["Webcam", "Upload Video"])
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)

# Load YOLO model
model = YOLO("best.pt")  # Change model if needed
classNames = model.names

# --- Functions ---
def process_frame(img):
    new_frame_time = time.time()
    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if cls < len(classNames) and conf > confidence:
                # Class-specific coloring
                color = (0, 255, 0) if classNames[cls] == 'mobile phone' else (255, 0, 0)

                # Draw bounding box with corners
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)

                # Put text
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                    (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color, colorB=color)

    return img

# --- Main App Logic ---
if source == "Webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access the camera.")
        st.stop()

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = process_frame(frame)
            stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

elif source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = process_frame(frame)
                stframe.image(frame, channels="BGR", use_column_width=True)
            else:
                break

        cap.release()