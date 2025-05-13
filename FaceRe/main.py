import os
import cv2
import time
import threading
import requests
import glob
import numpy as np
from deepface import DeepFace
from imutils.video import VideoStream
import tempfile
import argparse

# --------------------------------------------
# Configuration & Argument Parsing
# --------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Cross-platform Face Recognition with Telegram Alert")
    parser.add_argument("--db", default="database", help="Path to face images database")
    parser.add_argument("--token", required=True, help="Telegram Bot Token (export or pass here)")
    parser.add_argument("--chat_id", required=True, help="Telegram Chat ID to send alerts")
    parser.add_argument("--source", default=0, help="Camera source index or path (default=0)")
    parser.add_argument("--detector", default="retinaface", choices=["opencv", "ssd", "mtcnn", "retinaface"], help="Face detector backend")
    parser.add_argument("--model", default="ArcFace", choices=["VGG-Face","Facenet","ArcFace","DeepFace"], help="Recognition model")
    parser.add_argument("--threshold", type=float, default=0.4, help="Distance threshold for recognition")
    parser.add_argument("--interval", type=int, default=5, help="Frame skip interval between recognition calls")
    return parser.parse_args()

args = parse_args()
DB_PATH = args.db
TELEGRAM_TOKEN = args.token
TELEGRAM_CHAT_ID = args.chat_id
DETECTOR = args.detector
MODEL = args.model
THRESHOLD = args.threshold
INTERVAL = args.interval

# --------------------------------------------
# Prepare Database Embeddings
# --------------------------------------------
print("🔄 Loading database embeddings...")
embeddings = []
names = []
for img_path in glob.glob(os.path.join(DB_PATH, "*.jpg")):
    try:
        rep = DeepFace.represent(img_path=img_path, model_name=MODEL, detector_backend=DETECTOR, enforce_detection=False)
        vec = np.array(rep[0]['embedding']) if isinstance(rep, list) else np.array(rep)
        name = os.path.splitext(os.path.basename(img_path))[0].split("_")[0]
        embeddings.append(vec)
        names.append(name)
    except Exception as e:
        print(f"⚠️ Failed to process {img_path}: {e}")
print(f"✅ {len(embeddings)} embeddings loaded.")

# --------------------------------------------
# Telegram Alert Function
# --------------------------------------------
def send_telegram_alert(message, image_path=None):
    try:
        base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        requests.post(f"{base}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        if image_path:
            with open(image_path, 'rb') as img:
                requests.post(f"{base}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": img})
    except Exception as e:
        print(f"⚠️ Telegram alert failed: {e}")

# --------------------------------------------
# Recognition Logic
# --------------------------------------------
name_lock = threading.Lock()
current_name = None

def recognize_worker(frame):
    global current_name
    try:
        rep = DeepFace.represent(img_path=frame, model_name=MODEL, detector_backend=DETECTOR, enforce_detection=False)
        emb = np.array(rep[0]['embedding']) if isinstance(rep, list) else np.array(rep)
    except Exception as e:
        print(f"⚠️ Embed error: {e}")
        return

    dists = [np.linalg.norm(emb - db_emb) for db_emb in embeddings]
    min_dist = min(dists) if dists else float('inf')
    if min_dist < THRESHOLD:
        idx = int(np.argmin(dists))
        name = names[idx]
    else:
        name = "Not authorized"

    with name_lock:
        prev = current_name
        current_name = name
    if name == "Not authorized" and prev != name:
        print("🔒 Unknown detected!")
        tmp = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(tmp, frame)
        send_telegram_alert("🚨 ALERT: Stranger detected!", tmp)
    elif name != prev:
        print(f"✅ Identified: {name}")

# --------------------------------------------
# Main Stream Loop
# --------------------------------------------
print("▶️ Starting video stream...")
# Use VideoStream for smoother capture
vs = VideoStream(src=args.source).start()
time.sleep(1.0)
frame_count = 0

while True:
    frame = vs.read()
    if frame is None:
        break
    # Resize for speed
    h, w = frame.shape[:2]
    scale = 640.0 / max(w, h)
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = DeepFace.extract_faces(img_path = rgb, detector_backend=DETECTOR, enforce_detection=False)
    for (x, y, face_img, region) in faces:
        # region: (x, y, w, h) in resized coords
        fx, fy, fw, fh = region
        # Scale back to original frame coords
        cx, cy, cw, ch = int(fx/scale), int(fy/scale), int(fw/scale), int(fh/scale)
        face_crop = frame[cy:cy+ch, cx:cx+cw]
        if frame_count % INTERVAL == 0:
            # Launch thread for recognition
            threading.Thread(target=recognize_worker, args=(face_crop,)).start()
        # Draw box
        with name_lock:
            label = current_name or "Scanning..."
        color = (0,255,0) if label != "Not authorized" else (0,0,255)
        cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), color, 2)
        cv2.putText(frame, label, (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_count += 1

# Cleanup
vs.stop()
cv2.destroyAllWindows()
