import os
import cv2
import time
from deepface import DeepFace

# Ẩn cảnh báo TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DB_PATH = "database"

# Tạo thư mục nếu chưa có
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

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

def add_face_from_webcam():
    name = input("Nhập tên người dùng: ").strip()
    filename = f"{DB_PATH}/{name}_{{}}.jpg"  # Tên ảnh sẽ có số thứ tự

    cam_id = select_camera()
    cap = cv2.VideoCapture(cam_id)
    print("📸 Nhấn 's' để chụp ảnh, hoặc 'q' để thoát")

    count = 0  # Biến đếm số ảnh
    max_images = 10  # Số ảnh tối đa

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Chụp khuôn mặt mới", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if count < max_images:  # Kiểm tra số lượng ảnh
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

def recognize_face(frame, db_path=DB_PATH):
    try:
        result = DeepFace.find(img_path=frame, db_path=db_path, enforce_detection=False)
        if len(result[0]) > 0:
            identity = result[0].iloc[0]['identity']
            name = os.path.basename(identity).split(".")[0]
            return name
        else:
            return "Unknown"
    except Exception as e:
        print("Lỗi:", e)
        return "Error"

def start_recognition():
    cam_id = select_camera()
    cap = cv2.VideoCapture(cam_id)
    print("🔍 Đang chạy nhận diện khuôn mặt... Nhấn 'q' để thoát.")

    name = "Đang nhận diện..."
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Hiển thị trước, xử lý sau
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)

        # Nhận diện mỗi 30 frame để tránh lag
        if frame_count % 45 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            name = recognize_face(small_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.0001)

    cap.release()
    cv2.destroyAllWindows()

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
