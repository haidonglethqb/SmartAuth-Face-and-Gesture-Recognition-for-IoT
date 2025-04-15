import os
import cv2
import face_recognition
import threading
import requests
import time
import tempfile
import pickle
from datetime import datetime
from collections import deque
import paho.mqtt.client as mqtt
import ssl  # Thêm import ssl để sử dụng cho kết nối MQTT bảo mật

# === Cấu hình ===
DATASET_DIR = "dataset"
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
MIN_CONFIDENCE = 0.45
VOTE_HISTORY = 5  # Số frame vote nhận diện gần nhất

# === Cấu hình MQTT HiveMQ Cloud ===
mqtt_broker = "0030fdd1cc9b4f8dae5a436d2347d84a.s1.eu.hivemq.cloud"
mqtt_port = 8883
mqtt_user = "iot_home"
mqtt_pass = "Tngan1724"
mqtt_topic = "home/control"

# === Tạo thư mục nếu chưa tồn tại ===
os.makedirs(DATASET_DIR, exist_ok=True)

known_face_encodings = []
known_face_names = []
frame_to_check = None
name = "Scanning..."
result_lock = threading.Lock()
vote_buffer = deque(maxlen=VOTE_HISTORY)

# === Gửi cảnh báo Telegram ===
def send_telegram_alert(message, image_path=None):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        if image_path:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            with open(image_path, 'rb') as photo:
                requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": photo})
    except Exception as e:
        print("⚠️ Không thể gửi Telegram:", e)

# === Gửi thông báo MQTT ===
def send_mqtt_message(client, message):
    try:
        client.publish(mqtt_topic, message)
        print(f"[MQTT] Đã gửi: {message} -> {mqtt_topic}")
    except Exception as e:
        print(f"⚠️ Không thể gửi MQTT: {e}")

# === CLAHE để tăng cường độ sáng khuôn mặt ===
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# === Load khuôn mặt đã biết ===
def load_known_faces():
    global known_face_encodings, known_face_names
    if os.path.exists("encodings.pkl"):
        with open("encodings.pkl", "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"✅ Đã tải {len(known_face_names)} người từ cache.")
        return

    known_face_encodings = []
    known_face_names = []

    for file in os.listdir(DATASET_DIR):
        if file.lower().endswith((".jpg", ".png")):
            path = os.path.join(DATASET_DIR, file)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(file)[0].split("_")[0]
                known_face_names.append(name)

    with open("encodings.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"✅ Đã encode và lưu cache {len(known_face_names)} người.")

# === Thêm người mới từ webcam ===
def add_face_from_webcam():
    name = input("Nhập tên người mới: ").strip()
    cam_id = int(input("Nhập ID camera (0 nếu không chắc): ") or 0)
    cap = cv2.VideoCapture(cam_id)
    cap.set(3, 640)
    cap.set(4, 480)

    count = 0
    max_images = 5
    print("📸 Nhấn 's' để chụp, 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_clahe(frame)
        cv2.imshow("Thêm khuôn mặt", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and count < max_images:
            face_locations = face_recognition.face_locations(frame, model='cnn')
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_img = frame[top:bottom, left:right]
                filename = f"{DATASET_DIR}/{name}_{count+1}.jpg"
                cv2.imwrite(filename, face_img)
                print(f"✅ Lưu: {filename}")
                count += 1
            else:
                print("⚠️ Không tìm thấy khuôn mặt.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists("encodings.pkl"):
        os.remove("encodings.pkl")
    load_known_faces()

# === Nhận diện nền (background thread) ===
def recognize_background(client):
    global name, frame_to_check, vote_buffer
    previous_name = ""

    while True:
        if frame_to_check is not None:
            with result_lock:
                small_frame = frame_to_check.copy()
                frame_to_check = None

            small_frame = apply_clahe(small_frame)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            recognized_name = "Not authorized"

            for encoding in face_encodings:
                distances = face_recognition.face_distance(known_face_encodings, encoding)
                best_match_index = distances.argmin()
                if distances[best_match_index] < MIN_CONFIDENCE:
                    recognized_name = known_face_names[best_match_index]

            vote_buffer.append(recognized_name)
            voted_name = max(set(vote_buffer), key=vote_buffer.count)

            with result_lock:
                name = voted_name

            if voted_name == "Not authorized" and previous_name != "Not authorized":
                print("🔒 Phát hiện người lạ!")
                temp_img = tempfile.mktemp(suffix=".jpg")
                cv2.imwrite(temp_img, small_frame)
                send_telegram_alert("🚨 CẢNH BÁO: Người lạ trước camera!", temp_img)
                send_mqtt_message(client, "Người lạ trước camera!")
            elif voted_name != previous_name:
                print(f"✅ Nhận diện: {voted_name}")
                send_mqtt_message(client, f"Nhận diện: {voted_name}")

            previous_name = voted_name

# === Nhận diện trực tiếp từ webcam ===
def start_recognition(client):
    global frame_to_check, name

    threading.Thread(target=recognize_background, args=(client,), daemon=True).start()

    cam_id = int(input("Nhập ID camera (0 nếu không chắc): ") or 0)
    cap = cv2.VideoCapture(cam_id)
    cap.set(3, 640)
    cap.set(4, 480)

    print("🔍 Đang chạy nhận diện... Nhấn 'q' để thoát.")
    frame_count = 0
    interval = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_name = name
        frame_disp = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if w < 80 or h < 80:
                continue
            color = (0, 255, 0) if display_name != "Not authorized" else (0, 0, 255)
            cv2.rectangle(frame_disp, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame_disp, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if frame_count % interval == 0 and frame_to_check is None:
                with result_lock:
                    face_img = frame[y:y+h, x:x+w]
                    frame_to_check = cv2.resize(face_img, (150, 150))

        cv2.imshow("Face Recognition", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

# === Tạo client MQTT ===
def create_mqtt_client():
    client = mqtt.Client()
    client.username_pw_set(mqtt_user, mqtt_pass)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.connect(mqtt_broker, mqtt_port, 60)
    return client

# === Menu chính ===
def main_menu():
    client = create_mqtt_client()
    load_known_faces()
    while True:
        print("\n===== MENU =====")
        print("1. Nhận diện khuôn mặt")
        print("2. Thêm người mới")
        print("0. Thoát")
        choice = input("Chọn: ")

        if choice == "1":
            start_recognition(client)
        elif choice == "2":
            add_face_from_webcam()
        elif choice == "0":
            print("👋 Tạm biệt!")
            client.disconnect()
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")

# === Chạy chính ===
if __name__ == "__main__":
    main_menu()
