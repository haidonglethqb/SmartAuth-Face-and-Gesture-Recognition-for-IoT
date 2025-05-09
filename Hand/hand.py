import cv2
import mediapipe as mp
import math
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)


cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

       
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

                
                thumb_index_distance = calculate_distance(thumb_tip, index_tip)
                index_middle_distance = calculate_distance(index_tip, middle_tip)
                middle_ring_distance = calculate_distance(middle_tip, ring_tip)
                ring_pinky_distance = calculate_distance(ring_tip, pinky_tip)

                
                if thumb_index_distance > 0.09 and index_middle_distance > 0.09 and middle_ring_distance > 0.09 and ring_pinky_distance > 0.09:
                    cv2.putText(frame, "Mo cua - Xoe ban tay", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    

                elif thumb_index_distance < 0.15 and index_middle_distance < 0.15 and middle_ring_distance < 0.15 and ring_pinky_distance < 0.15:
                    cv2.putText(frame, "Nam tay - Dong cua", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    

                
                elif thumb_tip.y < thumb_ip.y and thumb_ip.y < thumb_mcp.y:
                    cv2.putText(frame, "Mo dieu hoa - Like", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    

                elif thumb_tip.y > thumb_ip.y and thumb_ip.y > thumb_mcp.y:
                    cv2.putText(frame, "Tat dieu hoa - Dislike", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    

        
        cv2.imshow("Hand Gesture Recognition", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    
    cap.release()
    cv2.destroyAllWindows()