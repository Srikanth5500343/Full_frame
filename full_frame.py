import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# ------------------ LOGIC ------------------
def calculate_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def score_frame(frame):
    small = cv2.resize(frame, (256, 192))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = small.shape[:2]
    best_score = 0
    is_full = False

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw < 20 or ch < 50:
            continue

        if ch / (cw + 1e-5) > 2:
            height = ch / h
            center = 1 - abs((x + cw/2) - w/2) / (w/2)

            crop = gray[y:y+ch, x:x+cw]
            sharpness = calculate_sharpness(crop) / 1000

            score = 0.6*height + 0.3*center + 0.1*sharpness

            if height > 0.7:
                is_full = True

            if score > best_score:
                best_score = score

    return best_score, is_full

# ------------------ UI ------------------
st.title("📦 Full Frame Rack Detector")

mode = st.radio("Select Mode", ["Upload Image", "Upload Video", "Live Camera"])

# ------------------ IMAGE ------------------
if mode == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if file:
        image = Image.open(file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        score, full = score_frame(frame)

        st.image(image, caption=f"Score: {score:.2f}")

        if full:
            st.success("✅ Full Frame Detected")
        else:
            st.error("❌ Not Full Frame")

# ------------------ VIDEO ------------------
elif mode == "Upload Video":
    file = st.file_uploader("Upload Video", type=["mp4"])

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)

        best_frame = None
        best_score = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            score, full = score_frame(frame)

            if full and score > best_score:
                best_score = score
                best_frame = frame.copy()

        cap.release()

        if best_frame is not None:
            st.image(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
            st.success("✅ Best Full Frame Extracted")

# ------------------ LIVE CAMERA ------------------
elif mode == "Live Camera":
    img = st.camera_input("Capture Rack")

    if img:
        image = Image.open(img)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        score, full = score_frame(frame)

        st.image(image, caption=f"Score: {score:.2f}")

        if full:
            st.success("✅ Full Frame Detected")
        else:
            st.error("❌ Not Full Frame")