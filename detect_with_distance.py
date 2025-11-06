# detect_with_distance.py
import cv2
import numpy as np
import pyttsx3
import time
import threading
import os

# ---- tts helper (non-blocking) ----
def speak(text):
    # run pyttsx3 in this thread (pyttsx3 is blocking) but we call speak in a separate thread
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def speak_async(text):
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()

# ---- model + labels ----
MODEL_PROTO = "models/MobileNetSSD_deploy.prototxt"
MODEL_WEIGHTS = "models/MobileNetSSD_deploy.caffemodel"

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

CONF_THRESHOLD = 0.5
COOLDOWN = 2.5  # seconds per label

# ---- load net ----
if not os.path.exists(MODEL_PROTO) or not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError("Model files not found in models/. Put prototxt and caffemodel in models/")

net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)

# ---- prepare captures folder ----
CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ---- helper: simple distance approximation ----
def distance_category(box_height, frame_height):
    # box_height and frame_height are in pixels
    if frame_height == 0:
        return "unknown"
    ratio = box_height / frame_height  # relative height (0..1)
    # thresholds tuned for laptop webcam typical framing; adjust if needed
    if ratio > 0.28:
        return "very near"
    if ratio > 0.12:
        return "near"
    return "far"

# ---- main detection loop ----
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: cannot open webcam. Try changing camera index.")
        return

    last_spoken = {}
    frame_count = 0

    print("Controls: 'q' = quit, 'c' = capture screenshot, 'p' = pause display")
    speak_async("System ready. Starting detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        frame_count += 1
        (h, w) = frame.shape[:2]
        # optional performance: process every frame or skip frames
        # create blob and forward
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # draw detections
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < CONF_THRESHOLD:
                continue

            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx] if idx < len(CLASSES) else f"class_{idx}"

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            box_h = endY - startY

            # draw
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"{label} {confidence:.2f}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # distance category
            dist_cat = distance_category(box_h, h)
            status_text = f"{label} ({dist_cat})"
            cv2.putText(frame, status_text, (startX, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            # speak with cooldown per label+category
            now = time.time()
            key = f"{label}_{dist_cat}"
            last = last_spoken.get(key, 0)
            if now - last > COOLDOWN:
                announcement = f"{label} {dist_cat}"
                print("Announce:", announcement)
                speak_async(announcement)
                last_spoken[key] = now

        # show frame
        cv2.imshow("Blind Navigation Assistant - Distance Mode", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(0)
        if key == ord('c'):
            # save screenshot with timestamp
            ts = int(time.time())
            fname = os.path.join(CAPTURE_DIR, f"capture_{ts}.jpg")
            cv2.imwrite(fname, frame)
            print("Saved screenshot:", fname)
            speak_async("Captured image for report")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
