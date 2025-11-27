import cv2  #xử lý hình ảnh, mở webcam, vẽ khung, hiển thị.
import mediapipe as mp #nhận dạng bàn tay và các keypoints (21 điểm trên tay)
import time #dùng để đo thời gian giữa các hành động (ví dụ double tap)
import math #dùng các hàm toán (ví dụ khoảng cách giữa hai điểm).
import numpy as np  #để tính toán vector, tốc độ, vận tốc.
import socket #gửi dữ liệu UDP ra ngoài (nếu bạn muốn app khác nhận kết quả cử chỉ).
 
# --------------------------
# CẤU HÌNH (tinh chỉnh nếu cần)
# --------------------------
CAM_ID = 0 # chọn webcam mặc định
MIN_DET_CONF = 0.6 # độ tin cậy khi detect tay
MIN_TRACK_CONF = 0.5

# Thresholds (tuy chỉnh theo camera / khoảng cách)
PINCH_THRESH = 0.06          # normalized distance thumb-index -> pinch
SPREAD_THRESH = 0.10         # > this -> spread
TAP_MAX_MOVEMENT = 0.03      # tap: vị trí ngón trỏ không dịch quá nhiều
TAP_MAX_DURATION = 0.25      # giây
DOUBLE_TAP_INTERVAL = 0.4    # giây: thời gian giữa 2 tap để coi là double tap
PRESS_TIME = 0.6             # giây để coi là press (giữ)
DRAG_SPEED_MAX = 0.6         # vận tốc nhỏ (đơn vị normalized/sec) -> drag khi giữ và di chuyển chậm
FLICK_SPEED_MIN = 1.2        # vận tốc lớn -> flick (normalized/sec)
HISTORY_LEN = 8              # số frame để tính vận tốc/acc

# UDP (tuỳ chọn) - nếu không cần thì không dùng
UDP_ENABLED = True
UDP_IP = "127.0.0.1"
UDP_PORT = 9999 #→ Gửi kết quả gesture qua localhost:9999, có thể dùng app khác nghe để điều khiển (VD: game, Blender, Arduino…).

# --------------------------
# Setup
# --------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_module = mp_hands.Hands(
#MediaPipe tạo ra Hands detector – nó cho 21 điểm (landmarks) trên mỗi bàn tay.
#Mỗi frame:

#hands_module.process() nhận ảnh RGB.

#Trả về multi_hand_landmarks (list bàn tay phát hiện).


    max_num_hands=2,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
)

sock = None
if UDP_ENABLED:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(CAM_ID)

# --------------------------
# Helpers
# --------------------------


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def norm_point(lm):
    # return (x,y) normalized
    return np.array([lm.x, lm.y], dtype=np.float32)

# Dataclass like structure per hand
class HandState:
    def __init__(self, hand_id):
        self.id = hand_id
        self.last_seen = time.time()
        self.history = []  # list of (pos, t) wrist pos
        self.index_history = [] # index tip history
        self.thumb_history = []
        self.press_start = None
        self.last_tap_time = 0
        self.tap_in_progress = False
        self.is_pinched = False
        self.is_spread = False
        self.last_gesture = None

#Nó lưu:

#Lịch sử vị trí các ngón tay → tính vận tốc / hướng.

#Thời điểm bắt đầu nhấn, tap.

#Trạng thái hiện tại (đang pinch/spread/press...).

#Hàm hỗ trợ:

#update_hist() → lưu lịch sử điểm.

#velocity_index() → tính vector vận tốc ngón trỏ.

#speed_index() → lấy độ lớn vận tốc (norm).

#avg_wrist_speed() → tốc độ cổ tay (để phân biệt khi tay di chuyển cả bàn).




    def update_hist(self, wrist_pt, index_pt, thumb_pt, t):
        self.last_seen = t
        self.history.append((wrist_pt, t))
        self.index_history.append((index_pt, t))
        self.thumb_history.append((thumb_pt, t))
        if len(self.history) > HISTORY_LEN:
            self.history.pop(0)
        if len(self.index_history) > HISTORY_LEN:
            self.index_history.pop(0)
        if len(self.thumb_history) > HISTORY_LEN:
            self.thumb_history.pop(0)





    def velocity_index(self):
        if len(self.index_history) < 2:
            return np.array([0.0, 0.0])
        p0, t0 = self.index_history[0]
        p1, t1 = self.index_history[-1]
        dt = t1 - t0
        if dt <= 0:
            return np.array([0.0, 0.0])
        return (p1 - p0) / dt





    def speed_index(self):
        v = self.velocity_index()
        return np.linalg.norm(v)






    def avg_wrist_speed(self):
        if len(self.history) < 2:
            return 0.0
        p0, t0 = self.history[0]
        p1, t1 = self.history[-1]
        dt = t1 - t0
        if dt <= 0: return 0.0
        return np.linalg.norm(p1 - p0) / dt

# store by hand label (0 or 1)
hand_states = {}

# --------------------------
# Gesture detection
# --------------------------
def detect_gestures_for_hand(hand_id, landmarks, image_w, image_h, hand_state, other_hands_states):#Hàm trung tâm, xử lý logic gesture cho 1 bàn tay.
    """
    returns gesture_name (str) or None
    """
    t = time.time()
    # Lấy các điểm quan trọng:
    wrist = norm_point(landmarks[0])
    index_tip = norm_point(landmarks[8])
    thumb_tip = norm_point(landmarks[4])
    middle_tip = norm_point(landmarks[12])

    hand_state.update_hist(wrist, index_tip, thumb_tip, t)

    # compute normalized distances (use landmark coords directly)
    p_thumb = thumb_tip
    p_index = index_tip


    #Tính khoảng cách và tốc độ
    d_thumb_index = np.linalg.norm(p_thumb - p_index)

    speed_index = hand_state.speed_index()  # normalized units per sec
    wrist_speed = hand_state.avg_wrist_speed()

    # ---- Pinch / Spread
    if d_thumb_index < PINCH_THRESH:
        if not hand_state.is_pinched:
            hand_state.is_pinched = True
            hand_state.is_spread = False
            hand_state.last_gesture = "Pinch"
            send_udp(f"HAND{hand_id}:PINCH")
            return "Pinch"
#Nếu ngón cái & trỏ gần nhau < PINCH_THRESH → Pinch.

#Nếu rất xa nhau > SPREAD_THRESH → Spread.

#Gửi qua UDP: "HAND0:PINCH" hoặc "HAND0:SPREAD".





    elif d_thumb_index > SPREAD_THRESH:
        if not hand_state.is_spread:
            hand_state.is_spread = True
            hand_state.is_pinched = False
            hand_state.last_gesture = "Spread"
            send_udp(f"HAND{hand_id}:SPREAD")
            return "Spread"
    else:
        # between thresholds -> neutral
        hand_state.is_pinched = False
        hand_state.is_spread = False

    # ---- Press (hold nearly stationary)
    # Condition: index tip not moving much and near camera plane (we rely on normalized coords)
    recent_idx = hand_state.index_history[-1][0]
    # compute movement over last few frames
    movement = 0.0
    if len(hand_state.index_history) >= 2:
        p0, t0 = hand_state.index_history[0] 
        #Nếu ngón trỏ di chuyển rất ít (movement < TAP_MAX_MOVEMENT) trong thời gian dài hơn PRESS_TIME, thì coi là nhấn giữ.
        p1, t1 = hand_state.index_history[-1]
        movement = np.linalg.norm(p1 - p0)
    # if movement small -> candidate press
    if movement < TAP_MAX_MOVEMENT and wrist_speed < DRAG_SPEED_MAX:
        if hand_state.press_start is None:
            hand_state.press_start = t
        else:
            if t - hand_state.press_start >= PRESS_TIME:
                # press recognized
                if hand_state.last_gesture != "Press":
                    hand_state.last_gesture = "Press"
                    send_udp(f"HAND{hand_id}:PRESS")
                    return "Press"
    else:
        hand_state.press_start = None

    # ---- Tap & Double Tap
    # Tap: quick contact -> approximated by brief low-movement epoch
    # We'll detect when index movement goes from near-zero to out-of-range (simulate touch down/up)
    # Simpler heuristic: if movement small AND short duration < TAP_MAX_DURATION, consider tap.
    # Implement as: detect "touch" start when index near thumb (or near wrist plane?) - approximate by distance from previous frame
    # We'll track transient low-movement window
    if movement < TAP_MAX_MOVEMENT:
        # possible tap start
        if not hand_state.tap_in_progress:
            hand_state.tap_in_progress = True
            hand_state.tap_started_at = t
    else:
        if hand_state.tap_in_progress:
            duration = t - hand_state.tap_started_at
#Nếu ngón trỏ dừng lại ngắn (<0.25s) rồi di chuyển ra xa → Tap.
#Nếu hai lần Tap trong 0.4s → Double Tap.



            hand_state.tap_in_progress = False
            if duration <= TAP_MAX_DURATION:
                # a tap occurred
                # check double tap
                if t - hand_state.last_tap_time <= DOUBLE_TAP_INTERVAL:
                    hand_state.last_tap_time = 0
                    hand_state.last_gesture = "Double Tap"
                    send_udp(f"HAND{hand_id}:DOUBLE_TAP")
                    return "Double Tap"
                else:
                    hand_state.last_tap_time = t
                    hand_state.last_gesture = "Tap"
                    send_udp(f"HAND{hand_id}:TAP")
                    return "Tap"

    # ---- Flick vs Drag
    # Flick: index speed above threshold
    if speed_index >= FLICK_SPEED_MIN:
        # direction


#Nếu tốc độ ngón trỏ > 1.2 (chuẩn hoá theo giây), xác định hướng trái/phải → Flick-Left / Flick-Right.
        v = hand_state.velocity_index()
        dir_x = "Right" if v[0] > 0 else "Left"
        hand_state.last_gesture = f"Flick-{dir_x}"
        send_udp(f"HAND{hand_id}:FLICK:{dir_x}")
        return f"Flick-{dir_x}"

    # Drag: moderate speed while index is "engaged" (we treat as when index is somewhat stable but moving)
    if speed_index > 0.05 and speed_index < DRAG_SPEED_MAX:
        hand_state.last_gesture = "Drag"
#Nếu tốc độ trung bình (0.05 < v < 0.6) → Drag (kéo).
        send_udp(f"HAND{hand_id}:DRAG")
        return "Drag"
    



    # ---- Press + Tap (complex): If this hand detects a tap while any other hand (or another finger) is in Press state
    for other in other_hands_states:
#Nếu tay này đang “Press” và tay kia thực hiện “Tap” trong khoảng thời gian gần → nhận diện “Press+Tap”.
        if other is None:
            continue
        if other.press_start is not None and time.time() - other.press_start >= PRESS_TIME:
            # other hand is pressing; if this hand registers tap event just now -> Press+Tap
            # We approximate: if we had a recent tap within DOUBLE_TAP_INTERVAL -> treat as press+tap
            if hand_state.last_gesture == "Tap" or (time.time() - hand_state.last_tap_time <= DOUBLE_TAP_INTERVAL):
                hand_state.last_gesture = "Press+Tap"
                send_udp(f"HAND{hand_id}:PRESS+TAP")
                return "Press+Tap"
    # Also check same-hand press + another finger quick touch: if press_start exists and index does tap -> press+tap
    if hand_state.press_start is not None and (t - hand_state.press_start >= PRESS_TIME):
        # if a quick flick of middle finger or quick movement of index qualifies as tap -> handled earlier
        pass

    return None

def send_udp(message):
    if UDP_ENABLED and sock is not None:
        try:
#Gửi chuỗi gesture cho app khác xử lý.
#Ví dụ trong Blender, Unity, hoặc script khác có thể đọc và thực hiện hành động tương ứng.
            sock.sendto(message.encode("utf-8"), (UDP_IP, UDP_PORT))
        except Exception as e:
            # ignore network errors
            pass

# --------------------------
# Main loop
# --------------------------
print("Starting Gesture Detector. Press ESC to quit.")
cv2.namedWindow("Gesture Detector", cv2.WINDOW_NORMAL)  
cv2.setWindowProperty("Gesture Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_module.process(img_rgb)
    now = time.time()
#Đọc khung hình từ webcam.

#Lật ảnh gương cho tự nhiên.

#Nhận diện bàn tay.

#Xoá trạng thái cũ nếu >1 giây không thấy tay (stale_keys).






    # remove stale hand states (not seen for >1s)
    stale_keys = [k for k,st in hand_states.items() if now - st.last_seen > 1.0]
    for k in stale_keys:
        del hand_states[k]

    gesture_texts = []

    if res.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
            # use handedness label as id if available
            # fallback use index i
            hand_label = i
            if hand_label not in hand_states:
                hand_states[hand_label] = HandState(hand_label)
            hs = hand_states[hand_label]

            # draw skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # detect gesture
            gesture = detect_gestures_for_hand(hand_label, hand_landmarks.landmark, w, h, hs,
                                               [st for k,st in hand_states.items() if k != hand_label])
            if gesture:
                gesture_texts.append((hand_label, gesture))
                # print gesture near hand (index tip)
                idx = hand_landmarks.landmark[8]
                cx, cy = int(idx.x * w), int(idx.y * h)
                cv2.putText(frame, gesture, (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # overlay active gestures summary
    y0 = 30
    for (hid, gt) in gesture_texts:
        cv2.putText(frame, f"H{hid}: {gt}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        y0 += 30

    # show instructions/tuning tips
    cv2.putText(frame, "Gestures: Tap,DoubleTap,Drag,Flick,Pinch,Spread,Press,Press+Tap", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
#Hiển thị danh sách cử chỉ đang hoạt động trên góc màn hình.

#ESC để thoát.




    cv2.imshow("Gesture Detector", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()


def action_from_gesture(gesture):
    # Map cử chỉ → hành động hệ thống
    if gesture == "PINCH":
        return "LIGHT_ON"          # bật đèn
    if gesture == "SPREAD":
        return "LIGHT_OFF"         # tắt đèn
    if gesture == "TAP":
        return "NEXT_SONG"         # chuyển bài nhạc
    if gesture == "DOUBLE_TAP":
        return "PREV_SONG"         # bài trước
    if gesture == "FLICK_Right":
        return "SELECT_DEVICE"     # chọn thiết bị bên phải
    if gesture == "FLICK_Left":
        return "SELECT_DEVICE_LEFT"
    
    return None
