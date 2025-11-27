""" import cv2
import mediapipe as mp
import socket
import math
import time

# UDP setup
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Function to send commands
def send_command(cmd):
    sock.sendto(cmd.encode(), (UDP_IP, UDP_PORT))
    print("[Sent]", cmd)

# Detect gestures
def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = math.dist(thumb_tip, index_tip)
    # pinch
    if dist < 0.05:
        return "PINCH"
    # spread
    elif dist > 0.2:
        return "SPREAD"
    return None

# Main loop
cap = cv2.VideoCapture(0)
prev_gesture = None
gesture_start = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            lm = [(p.x, p.y) for p in handLms.landmark]

            gesture = detect_gesture(lm)
            if gesture:
                if gesture != prev_gesture:
                    gesture_start = time.time()
                    prev_gesture = gesture
                elif time.time() - gesture_start > 0.5:
                    if gesture == "PINCH":
                        send_command("LIGHT_OFF")
                    elif gesture == "SPREAD":
                        send_command("LIGHT_ON")
                    prev_gesture = None

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows() """

# import cv2
# import mediapipe as mp
# import socket
# import math
# import time

# # =========================================
# # UDP setup - Gửi tín hiệu sang Blender
# # =========================================
# UDP_IP = "127.0.0.1"
# UDP_PORT = 9999
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# def send_command(cmd):
#     """Gửi tín hiệu sang Blender"""
#     sock.sendto(cmd.encode(), (UDP_IP, UDP_PORT))
#     print("[Sent]", cmd)

# # =========================================
# # MediaPipe setup - Nhận diện bàn tay
# # =========================================
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# # =========================================
# # Tính khoảng cách giữa 2 điểm
# # =========================================
# def distance(p1, p2):
#     return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# # =========================================
# # Nhận dạng cử chỉ
# # =========================================
# def detect_gesture(landmarks):
#     thumb_tip = landmarks[4]
#     index_tip = landmarks[8]
#     dist = distance(thumb_tip, index_tip)

#     # pinch
#     if dist < 0.05:
#         return "PINCH"
#     # spread
#     elif dist > 0.20:
#         return "SPREAD"
#     return None

# # =========================================
# # Chạy webcam & nhận diện
# # =========================================
# cap = cv2.VideoCapture(0)
# prev_gesture = None
# gesture_start = 0

# while True:
#     success, frame = cap.read()
#     if not success:
#         break

#     # Lật ảnh gương cho tự nhiên
#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     if result.multi_hand_landmarks:
#         for handLms in result.multi_hand_landmarks:
#             h, w, _ = frame.shape
#             lm = [(p.x, p.y) for p in handLms.landmark]

#             # Nhận diện gesture
#             gesture = detect_gesture(lm)
#             if gesture:
#                 if gesture != prev_gesture:
#                     gesture_start = time.time()
#                     prev_gesture = gesture
#                 elif time.time() - gesture_start > 0.5:
#                     if gesture == "PINCH":
#                         send_command("LIGHT_OFF")
#                     elif gesture == "SPREAD":
#                         send_command("LIGHT_ON")
#                     prev_gesture = None

#             # Vẽ khung bàn tay
#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

#     cv2.imshow("🖐️ Gesture Smart Home", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import math
import time
import numpy as np
import socket

# ---------- UDP setup ----------
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send_command(cmd):
    sock.sendto(cmd.encode(), (UDP_IP, UDP_PORT))
    print("[Sent]", cmd)

# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ---------- Helper ----------
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# ---------- Gesture tracking ----------
prev_pos = None
last_tap_time = 0
press_start = 0
gesture_state = None
last_gesture = None
tap_count = 0

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lm = [(p.x, p.y) for p in hand.landmark]
            thumb, index, middle = lm[4], lm[8], lm[12]

            # ---------- Detect Pinch / Spread ----------
            d_thumb_index = distance(thumb, index)
            if d_thumb_index < 0.05:
                gesture_state = "PINCH"
            elif d_thumb_index > 0.20:
                gesture_state = "SPREAD"

            # ---------- Detect Drag / Flick ----------
            cur_pos = np.array([index[0], index[1]])
            if prev_pos is not None:
                move_vec = cur_pos - prev_pos
                speed = np.linalg.norm(move_vec)

                if speed > 0.02:
                    gesture_state = "FLICK"
                elif 0.002 < speed < 0.02:
                    gesture_state = "DRAG"
            prev_pos = cur_pos

            # ---------- Detect Tap / Double Tap ----------
            current_time = time.time()
            if d_thumb_index < 0.05:
                if current_time - last_tap_time < 0.4:
                    gesture_state = "DOUBLE_TAP"
                else:
                    gesture_state = "TAP"
                last_tap_time = current_time

            # ---------- Detect Press ----------
            if d_thumb_index < 0.06:
                if press_start == 0:
                    press_start = time.time()
                elif time.time() - press_start > 1.0:
                    gesture_state = "PRESS"
            else:
                press_start = 0

            # ---------- Detect Press + Tap ----------
            d_index_middle = distance(index, middle)
            if d_thumb_index < 0.06 and d_index_middle < 0.05:
                gesture_state = "PRESS_TAP"

            # ---------- Gửi lệnh demo ----------
            if gesture_state and gesture_state != last_gesture:
                if gesture_state == "PINCH":
                    send_command("LIGHT_OFF")
                elif gesture_state == "SPREAD":
                    send_command("LIGHT_ON")
                elif gesture_state == "TAP":
                    send_command("TAP_ACTION")
                elif gesture_state == "DOUBLE_TAP":
                    send_command("DOUBLE_TAP_ACTION")
                elif gesture_state == "PRESS":
                    send_command("PRESS_ACTION")
                elif gesture_state == "PRESS_TAP":
                    send_command("PRESS_TAP_ACTION")
                elif gesture_state == "DRAG":
                    send_command("MOVE_OBJECT")
                elif gesture_state == "FLICK":
                    send_command("SWIPE_ACTION")

                last_gesture = gesture_state
                print("Detected:", gesture_state)

    cv2.imshow("Gesture Control Pro", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

