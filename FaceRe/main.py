import os
import cv2
import time
import threading
import requests
from deepface import DeepFace
import tempfile

# Ẩn cảnh báo TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DB_PATH = "database"

# Thông tin Telegram Bot (🔁 Thay bằng giá trị thật)
TELEGRAM_TOKEN = "7695555624:AAHJoIjeriV_AvsUY6KW2rOawkKzqTc71UU"
TELEGRAM_CHAT_ID = "5788605495"  # Chat ID của bạn

# Tạo thư mục nếu chưa có
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Biến toàn cục dùng cho luồng nhận diện
name = "Scanning......"
frame_to_check = None
result_lock = threading.Lock()

# Hàm gửi thông báo kèm ảnh qua Telegram
def send_telegram_alert(message, image_path=None):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        }

        # Gửi tin nhắn văn bản
        response = requests.post(url, data=payload)

        # Nếu có ảnh, gửi ảnh kèm theo
        if image_path:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                response = requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID}, files=files)

    except Exception as e:
        print(f"⚠️ Không thể gửi Telegram: {e}")

# Chọn camera sử dụng
def select_camera():
    print("\n🖥️ Danh sách ID camera có thể là:")
    print("0: Camera mặc định (thường là tích hợp trong laptop)")
    print("1: Camera USB rời (nếu có)")
    print("👉 Bạn có thể thử nhập 0 hoặc 1 nếu không chắc.")
    try:
        cam_id = int(input("Nhập ID camera muốn dùng: "))
        return cam_id
    except ValueError:
        print("⚠️ ID không hợp lệ. Dùng mặc định: 0")
        return 0

# Thêm khuôn mặt mới vào cơ sở dữ liệu
def add_face_from_webcam():
    name_input = input("Nhập tên người dùng: ").strip()
    filename = f"{DB_PATH}/{name_input}_{{}}.jpg"

    cam_id = select_camera()
    cap = cv2.VideoCapture(cam_id)
    print("📸 Nhấn 's' để chụp ảnh, hoặc 'q' để thoát")

    count = 0
    max_images = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Chụp khuôn mặt mới", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if count < max_images:
                img_filename = filename.format(count + 1)
                cv2.imwrite(img_filename, frame)
                print(f"✅ Đã lưu ảnh tại {img_filename}")
                count += 1
            else:
                print(f"⚠️ Đã chụp đủ {max_images} ảnh. Không thể chụp thêm.")
                break
        elif key == ord('q'):
            print("❌ Huỷ bỏ thao tác thêm khuôn mặt.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Nhận diện khuôn mặt từ ảnh và cơ sở dữ liệu
def recognize_face(frame, db_path=DB_PATH):
    try:
        result = DeepFace.find(img_path=frame, db_path=db_path, enforce_detection=False)
        if len(result[0]) > 0:
            identity = result[0].iloc[0]['identity']
            name_found = os.path.basename(identity).split(".")[0]

            if "_" in name_found:
                name_found = name_found.split("_")[0]

            return name_found
        else:
            return "Not authorized"
    except Exception as e:
        print("Lỗi:", e)
        return "Error"

# Nhận diện nền (lặp liên tục)
def recognize_background():
    global name, frame_to_check
    previous_name = ""
    while True:
        if frame_to_check is not None:
            with result_lock:
                frame = frame_to_check.copy()
                frame_to_check = None

            new_name = recognize_face(frame)
            with result_lock:
                name = new_name

            if new_name == "Not authorized":
                if previous_name != "Not authorized":
                    print("🔒 Người lạ xuất hiện!")
                    # Lưu ảnh người lạ tạm thời
                    temp_image_path = tempfile.mktemp(suffix='.jpg')
                    cv2.imwrite(temp_image_path, frame)
                    # Gửi thông báo kèm ảnh
                    send_telegram_alert("🚨 CẢNH BÁO: Có người lạ xuất hiện trước camera!", temp_image_path)
            elif new_name != previous_name:
                print(f"✅ Nhận diện: {new_name}")
            previous_name = new_name

# Bắt đầu nhận diện khuôn mặt
def start_recognition():
    global frame_to_check, name

    recognition_thread = threading.Thread(target=recognize_background, daemon=True)
    recognition_thread.start()

    cam_id = select_camera()
    cap = cv2.VideoCapture(cam_id)
    print("🔍 Đang chạy nhận diện khuôn mặt... Nhấn 'q' để thoát.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_count = 0
    recognition_interval = 10  # Nhận diện mỗi N frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        with result_lock:
            display_name = name

        for (x, y, w, h) in faces:
            # Tô khung xanh nếu nhận diện được, đỏ nếu là người lạ
            color = (0, 255, 0) if display_name != "Not authorized" else (0, 0, 255)

            # Vẽ khung và tên
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

            # Lấy ảnh nhỏ để nhận diện mỗi N frame
            if frame_count % recognition_interval == 0 and frame_to_check is None:
                face_crop = frame[y:y + h, x:x + w]
                face_crop = cv2.resize(face_crop, (160, 160))

                with result_lock:
                    frame_to_check = face_crop.copy()

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        time.sleep(0.0001)

    cap.release()
    cv2.destroyAllWindows()

# Menu chính
def main_menu():
    while True:
        print("\n=== MENU ===")
        print("1. Nhận diện khuôn mặt")
        print("2. Thêm người dùng mới từ webcam")
        print("0. Thoát")
        choice = input("Chọn chức năng: ")

        if choice == "1":
            start_recognition()
        elif choice == "2":
            add_face_from_webcam()
        elif choice == "0":
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main_menu()
