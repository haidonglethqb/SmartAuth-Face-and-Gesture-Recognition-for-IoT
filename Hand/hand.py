import cv2
import mediapipe as mp
import math
import os
import platform
import time
import sys

# === Cấu hình môi trường TensorFlow Lite ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DISABLE_TFLITE_XNNPACK'] = '1'

# Chọn camera
is_windows = platform.system() == "Windows"
camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

# Mở camera và warm-up
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(20):
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        break
    time.sleep(0.05)
if not cap.isOpened():
    print(f" Không thể mở camera index {camera_index}")
    sys.exit()

# === Biến trạng thái ra lệnh và timers ===
command_mode = False
fist_start_time = None
command_executed = False

command_mode_start_time = None
MODE_DISPLAY_DURATION = 2.0    # hiển thị "COMMAND MODE" trong 2s

exit_mode_start_time = None
EXIT_DISPLAY_DURATION = 2.0    # hiển thị "EXIT COMMAND MODE" trong 2s

gesture_name = None
gesture_start_time = None
HOLD_TIME = 3.0                # giữ gesture 3s để thực thi

# Ngưỡng nắm tay
THR_CLS = 0.15  # nếu tất cả khoảng cách < THR_CLS thì xem là nắm

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Hiển thị "COMMAND MODE" trong 2s đầu
        if command_mode and command_mode_start_time:
            if now - command_mode_start_time < MODE_DISPLAY_DURATION:
                cv2.putText(frame, "COMMAND MODE", (200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # Hiển thị "EXIT COMMAND MODE" trong 2s sau khi thoát
        if exit_mode_start_time:
            if now - exit_mode_start_time < EXIT_DISPLAY_DURATION:
                cv2.putText(frame, " EXIT COMMAND MODE", (200, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                exit_mode_start_time = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # Lấy các điểm tip và pip
            tips = {
                'thumb': lm.landmark[mp_hands.HandLandmark.THUMB_TIP],
                'index': lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                'middle': lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                'ring': lm.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                'pinky': lm.landmark[mp_hands.HandLandmark.PINKY_TIP],
            }
            pips = {
                'thumb': lm.landmark[mp_hands.HandLandmark.THUMB_IP],
                'index': lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                'middle': lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                'ring': lm.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
                'pinky': lm.landmark[mp_hands.HandLandmark.PINKY_PIP],
            }

            # Tính khoảng cách liên tiếp để nhận fist
            d_t_i = calculate_distance(tips['thumb'], tips['index'])
            d_i_m = calculate_distance(tips['index'], tips['middle'])
            d_m_r = calculate_distance(tips['middle'], tips['ring'])
            d_r_p = calculate_distance(tips['ring'], tips['pinky'])
            is_fist = (d_t_i < THR_CLS and d_i_m < THR_CLS and
                       d_m_r < THR_CLS and d_r_p < THR_CLS)

            # -- Chưa command mode: chờ nắm 3s để vào --
            if not command_mode:
                if is_fist:
                    if fist_start_time is None:
                        fist_start_time = now
                    elif now - fist_start_time >= 3.0:
                        command_mode = True
                        command_executed = False
                        command_mode_start_time = now
                        exit_mode_start_time = None
                        gesture_name = None
                        gesture_start_time = None
                else:
                    fist_start_time = None

            # -- Đang ở command mode --
            else:
                # Chờ hết 2s hiển thị mode thì mới xử lý gesture
                if not (command_mode_start_time and now - command_mode_start_time < MODE_DISPLAY_DURATION):
                    # Đếm số ngón giơ lên
                    count = 0
                    # Thumb (tay phải) xét x
                    if tips['thumb'].x < pips['thumb'].x:
                        count += 1
                    # Các ngón khác xét y
                    for f in ['index','middle','ring','pinky']:
                        if tips[f].y < pips[f].y:
                            count += 1

                    # Hiển thị số ngón
                    cv2.putText(frame, f"Fingers: {count}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                    # Nếu giơ 5 ngón: thoát ngay Command Mode
                    if count == 5:
                        command_mode = False
                        fist_start_time = None
                        command_mode_start_time = None
                        exit_mode_start_time = now
                        continue

                    if not command_executed:
                        # Ánh xạ số ngón sang lệnh
                        mapping = {
                            1: "Open door",
                            2: "Close door",
                            3: "Turn on AC",
                            4: "Turn off AC"
                        }
                        current = mapping.get(count, None)

                        # Nếu thay đổi cử chỉ, reset timer giữ
                        if current != gesture_name:
                            gesture_name = current
                            gesture_start_time = now

                        if gesture_name:
                            elapsed = now - gesture_start_time
                            remaining = max(0, HOLD_TIME - elapsed)
                            cv2.putText(frame,
                                        f"Exec '{gesture_name}' in {remaining:.1f}s",
                                        (200, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0,255,0) if elapsed>=HOLD_TIME else (0,255,255),
                                        2)
                            if elapsed >= HOLD_TIME:
                                cv2.putText(frame, f"✅ {gesture_name}!", (200, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                                command_executed = True
                    else:
                        # Sau khi thực thi, chờ nắm tay để reset
                        if is_fist:
                            command_mode = False
                            fist_start_time = now
                            command_mode_start_time = None

        else:
            # Không thấy tay: reset timer nắm
            fist_start_time = None

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
