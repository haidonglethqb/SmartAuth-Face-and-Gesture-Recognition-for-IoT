import cv2
import mediapipe as mp
import math

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Hàm tính khoảng cách giữa 2 điểm
def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

# Mở camera
cap = cv2.VideoCapture(1)  # Thay đổi 0 hoặc 1 tùy camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh từ BGR sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện cử chỉ tay
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ các điểm landmark bàn tay
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy các điểm landmark quan trọng
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

            # Tính khoảng cách giữa các ngón tay
            thumb_index_distance = calculate_distance(thumb_tip, index_tip)
            index_middle_distance = calculate_distance(index_tip, middle_tip)
            middle_ring_distance = calculate_distance(middle_tip, ring_tip)
            ring_pinky_distance = calculate_distance(ring_tip, pinky_tip)

            # === Ưu tiên kiểm tra MỞ/ĐÓNG CỬA trước ===
            if thumb_index_distance > 0.09 and index_middle_distance > 0.09 and middle_ring_distance > 0.09 and ring_pinky_distance > 0.09:
                cv2.putText(frame, "Mo cua - Xoe ban tay", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Code mở cửa ở đây

            elif thumb_index_distance < 0.15 and index_middle_distance < 0.15 and middle_ring_distance < 0.15 and ring_pinky_distance < 0.15:
                cv2.putText(frame, "Nam tay - Dong cua", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Code đóng cửa ở đây

            # === Nếu không phải mở/đóng cửa, kiểm tra MỞ/TẮT điều hòa ===
            elif thumb_tip.y < thumb_ip.y and thumb_ip.y < thumb_mcp.y:
                cv2.putText(frame, "Mo dieu hoa - Like", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # Code mở điều hòa ở đây

            elif thumb_tip.y > thumb_ip.y and thumb_ip.y > thumb_mcp.y:
                cv2.putText(frame, "Tat dieu hoa - Dislike", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # Code tắt điều hòa ở đây

    # Hiển thị kết quả
    cv2.imshow("Hand Gesture Recognition", frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
