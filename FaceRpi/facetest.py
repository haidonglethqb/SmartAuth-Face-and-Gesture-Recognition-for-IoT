import os
import cv2
import face_recognition
import threading
import requests
import time
import tempfile
import pickle
from collections import deque
import paho.mqtt.client as mqtt
import ssl

# ---------- CẤU HÌNH (Điền thông tin của bạn vào đây) ----------
# Đảm bảo dataset nằm kế bên file facetest.py
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(SCRIPT_DIR, "dataset")
TELEGRAM_TOKEN   = "YOUR_TELEGRAM_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
MIN_CONFIDENCE   = 0.45
VOTE_HISTORY     = 5

# MQTT broker (hostname hoặc IP)
mqtt_broker = "28766de7c6c947dd865b4f3ab34e8883.s1.eu.hivemq.cloud"
mqtt_port   = 8883
mqtt_user   = "haichu"
mqtt_pass   = "H@ichu321"
mqtt_topic  = "home/control"

# Tạo thư mục dataset nếu chưa có
os.makedirs(DATASET_DIR, exist_ok=True)

# Globals
known_face_encodings = []
known_face_names     = []
frame_to_check       = None
name                 = "Scanning..."
result_lock          = threading.Lock()
vote_buffer          = deque(maxlen=VOTE_HISTORY)


def send_telegram_alert(message, image_path=None):
    """Gửi alert qua Telegram"""
    try:
        base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        requests.post(f"{base}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                      timeout=5)
        if image_path:
            with open(image_path, 'rb') as photo:
                requests.post(f"{base}/sendPhoto",
                              data={"chat_id": TELEGRAM_CHAT_ID},
                              files={"photo": photo},
                              timeout=5)
    except Exception as e:
        print("⚠️ Không thể gửi Telegram:", e)


def send_mqtt_message(client, message):
    """Gửi message qua MQTT"""
    try:
        client.publish(mqtt_topic, message)
        print(f"[MQTT] Đã gửi: {message} -> {mqtt_topic}")
    except Exception as e:
        print("⚠️ Không thể gửi MQTT:", e)


def apply_clahe(image):
    """Cải thiện tương phản bằng CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def load_known_faces():
    """Load cache hoặc encode mới bộ dataset"""
    global known_face_encodings, known_face_names
    cache_path = os.path.join(DATASET_DIR, "encodings.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"✅ Đã tải {len(known_face_names)} người từ cache.")
        return

    known_face_encodings = []
    known_face_names     = []
    for file in os.listdir(DATASET_DIR):
        if file.lower().endswith((".jpg", ".png")):
            path = os.path.join(DATASET_DIR, file)
            image = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(image)
            if encs:
                known_face_encodings.append(encs[0])
                person = os.path.splitext(file)[0].split("_")[0]
                known_face_names.append(person)

    with open(cache_path, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"✅ Đã encode và lưu cache {len(known_face_names)} người.")


def add_face_from_webcam():
    """Thêm người mới qua webcam USB với preview chất lượng cao"""
    person_name = input("Nhập tên người mới: ").strip()
    cam_id = int(input("Nhập ID camera (mặc định 0): ") or 0)
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    count = 0
    max_images = 5
    print("📸 Nhấn 's' để chụp, 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        disp = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Thêm khuôn mặt", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and count < max_images:
            locs = face_recognition.face_locations(frame, model='hog')
            print("Detected boxes:", locs)
            if locs:
                top, right, bottom, left = locs[0]
                face_img = frame[top:bottom, left:right]
                filename = os.path.join(DATASET_DIR, f"{person_name}_{count+1}.jpg")
                cv2.imwrite(filename, face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                print(f"✅ Đã lưu: {filename}")
                count += 1
            else:
                print("⚠️ Không tìm thấy khuôn mặt, thử lại.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cache_path = os.path.join(DATASET_DIR, "encodings.pkl")
    if os.path.exists(cache_path):
        os.remove(cache_path)
    load_known_faces()


def recognize_background(client):
    """Thread nhận diện nền, gửi alert khi có thay đổi"""
    global name, frame_to_check, vote_buffer
    previous_name = ""

    while True:
        if frame_to_check is None:
            time.sleep(0.01)
            continue

        with result_lock:
            frame = frame_to_check.copy()
            frame_to_check = None

        small = apply_clahe(frame)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb, model='hog')
        encs  = face_recognition.face_encodings(rgb, locs)

        recognized = "Not authorized"
        for e in encs:
            dists = face_recognition.face_distance(known_face_encodings, e)
            idx   = dists.argmin() if dists.size else None
            if idx is not None and dists[idx] < MIN_CONFIDENCE:
                recognized = known_face_names[idx]

        vote_buffer.append(recognized)
        voted = max(set(vote_buffer), key=vote_buffer.count)

        with result_lock:
            name = voted

        if voted != previous_name:
            if voted == "Not authorized":
                print("🔒 Phát hiện người lạ!")
                imgfile = tempfile.mktemp(suffix=".jpg")
                cv2.imwrite(imgfile, frame)
                send_telegram_alert("🚨 CẢNH BÁO: Người lạ trước camera!", imgfile)
                send_mqtt_message(client, "Người lạ trước camera!")
            else:
                print(f"✅ Nhận diện: {voted}")
                send_mqtt_message(client, f"Nhận diện: {voted}")
            previous_name = voted


def start_recognition(client):
    """Chạy nhận diện liên tục, hiển thị khung"""
    global frame_to_check, name
    threading.Thread(target=recognize_background, args=(client,), daemon=True).start()

    cam_id = int(input("Nhập ID camera (mặc định 0): ") or 0)
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    print("🔍 Đang chạy nhận diện... Nhấn 'q' để thoát.")
    frame_count = 0
    interval    = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with result_lock:
            display_name = name

        disp = frame.copy()
        gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if w < 80 or h < 80:
                continue
            color = (0,255,0) if display_name != "Not authorized" else (0,0,255)
            cv2.rectangle(disp, (x,y), (x+w,y+h), color, 2)
            cv2.putText(disp, display_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if frame_count % interval == 0 and frame_to_check is None:
                with result_lock:
                    face_img = frame[y:y+h, x:x+w]
                    frame_to_check = cv2.resize(face_img, (150,150))

        cv2.imshow("Face Recognition", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()


def create_mqtt_client():
    """Khởi tạo MQTT client, dùng MQTTv311 để tránh cảnh báo"""
    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.username_pw_set(mqtt_user, mqtt_pass)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.connect(mqtt_broker, mqtt_port, keepalive=60)
    return client


def main_menu():
    client = create_mqtt_client()
    load_known_faces()
    while True:
        print("\n===== MENU =====")
        print("1. Nhận diện khuôn mặt")
        print("2. Thêm người mới")
        print("0. Thoát")
        choice = input("Chọn: ").strip()
        if choice == "1":
            start_recognition(client)
        elif choice == "2":
            add_face_from_webcam()
        elif choice == "0":
            client.disconnect()
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    main_menu()
