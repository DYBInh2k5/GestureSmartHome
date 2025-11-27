import cv2  # EN: OpenCV for image processing and camera. / VN: OpenCV để xử lý ảnh và mở webcam.
import mediapipe as mp  # EN: MediaPipe Hands for 21 hand landmarks. / VN: MediaPipe Hands để nhận diện 21 điểm bàn tay.
import time  # EN: For timing (tap durations, intervals). / VN: Dùng để đo thời gian giữa các hành động.
import math  # EN: Math helpers (e.g. hypot). / VN: Hàm toán học.
import numpy as np  # EN: Vector math (velocities, norms). / VN: Tính toán vector, vận tốc.
import socket  # EN: Optional UDP to send gesture strings to other apps. / VN: Gửi dữ liệu UDP ra ngoài (Unity, Blender, Arduino,...)

# --------------------------
# CONFIGURATION / CẤU HÌNH
# --------------------------
CAM_ID = 0  # EN: default webcam index. / VN: id webcam mặc định
MIN_DET_CONF = 0.6  # EN: MediaPipe minimum detection confidence. / VN: ngưỡng phát hiện bàn tay
MIN_TRACK_CONF = 0.5  # EN: tracking confidence. / VN: ngưỡng theo dõi

# Thresholds (tweak for your camera / distance)
PINCH_THRESH = 0.06          # EN: normalized distance thumb-index -> pinch. / VN: khoảng cách chuẩn hóa ngón cái - trỏ để xem là pinch
SPREAD_THRESH = 0.10         # EN: > this -> spread. / VN: > này -> spread (xòe)
TAP_MAX_MOVEMENT = 0.03      # EN: max movement for a tap (normalized coords). / VN: ngưỡng di chuyển tối đa cho tap
TAP_MAX_DURATION = 0.25      # EN: max duration (s) for a tap. / VN: thời gian tối đa (s) để tính là tap
DOUBLE_TAP_INTERVAL = 0.4    # EN: max interval between taps to count as double-tap. / VN: khoảng cách thời gian 2 tap để thành double tap
PRESS_TIME = 0.6             # EN: hold time (s) to consider as press. / VN: thời gian giữ để coi là press
DRAG_SPEED_MAX = 0.6         # EN: normalized/sec -> max wrist speed for drag scenario. / VN: vận tốc tối đa (chuẩn hóa) để xem là drag
FLICK_SPEED_MIN = 1.2        # EN: min speed to consider as flick. / VN: vận tốc tối thiểu để coi là flick
HISTORY_LEN = 8              # EN: number of frames to keep in history for velocity calculation. / VN: độ dài lịch sử (frame)

# UDP (optional) - set UDP_ENABLED False if you don't want network
UDP_ENABLED = True
UDP_IP = "127.0.0.1"
UDP_PORT = 9999  # EN: port to send gestures to. / VN: cổng UDP để gửi cử chỉ

# --------------------------
# SETUP MediaPipe, UDP socket, Camera
# --------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands_module = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
)

sock = None
if UDP_ENABLED:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # EN: create UDP socket / VN: tạo socket UDP

cap = cv2.VideoCapture(CAM_ID)  # EN: open camera. / VN: mở webcam

# --------------------------
# HELPERS / HỖ TRỢ
# --------------------------

def dist(a, b):
    """
    EN: Euclidean distance between two simple objects with x,y attributes.
    VN: Khoảng cách Euclid giữa hai điểm có thuộc tính x,y.
    """
    return math.hypot(a.x - b.x, a.y - b.y)

def norm_point(lm):
    """
    EN: Convert MediaPipe landmark (has x,y normalized) to numpy array [x,y].
    VN: Chuyển landmark của MediaPipe (x,y normalized) sang numpy array [x,y].
    """
    return np.array([lm.x, lm.y], dtype=np.float32)


# ...existing code...

def detect_black_object(frame, lower_thresh=(0, 0, 0), upper_thresh=(180, 255, 50)):
    """
    EN: Detect black object in the frame using color segmentation.
    VN: Phát hiện vật màu đen trong khung hình bằng phân đoạn màu.
    - lower_thresh: Lower HSV threshold for black.
    - upper_thresh: Upper HSV threshold for black.
    Returns:
        - mask: Binary mask of detected black regions.
        - contours: Contours of detected black regions.
    """
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Create a binary mask for black color
    mask = cv2.inRange(hsv, np.array(lower_thresh), np.array(upper_thresh))
    # Find contours of the black regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours










# Dataclass-like per-hand state
class HandState:
    """
    EN: Keeps history and simple state for each detected hand.
    VN: Lưu lịch sử và trạng thái đơn giản cho từng bàn tay.
    """
    def __init__(self, hand_id):
        self.id = hand_id
        self.last_seen = time.time()
        self.history = []         # list of (wrist_pos, t)
        self.index_history = []   # list of (index_tip_pos, t)
        self.thumb_history = []   # list of (thumb_tip_pos, t)
        self.press_start = None   # timestamp when candidate press started
        self.last_tap_time = 0    # timestamp last tap
        self.tap_in_progress = False
        self.tap_started_at = 0
        self.is_pinched = False
        self.is_spread = False
        self.last_gesture = None

    def update_hist(self, wrist_pt, index_pt, thumb_pt, t):
        """
        EN: Append new positions and trim history to HISTORY_LEN.
        VN: Thêm vị trí mới và cắt lịch sử theo HISTORY_LEN.
        """
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
        """
        EN: Compute velocity vector of index fingertip (normalized units/sec).
        VN: Tính vector vận tốc của đầu ngón trỏ (đơn vị chuẩn hóa / giây).
        """
        if len(self.index_history) < 2:
            return np.array([0.0, 0.0])
        p0, t0 = self.index_history[0]
        p1, t1 = self.index_history[-1]
        dt = t1 - t0
        if dt <= 0:
            return np.array([0.0, 0.0])
        return (p1 - p0) / dt

    def speed_index(self):
        """
        EN: Speed magnitude of index fingertip.
        VN: Độ lớn vận tốc ngón trỏ.
        """
        v = self.velocity_index()
        return np.linalg.norm(v)

    def avg_wrist_speed(self):
        """
        EN: Average wrist speed over history (norm of displacement / time).
        VN: Tốc độ trung bình cổ tay (norm khoảng cách / thời gian).
        """
        if len(self.history) < 2:
            return 0.0
        p0, t0 = self.history[0]
        p1, t1 = self.history[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return np.linalg.norm(p1 - p0) / dt

# store HandState by hand label/index
hand_states = {}

# --------------------------
# UDP SENDER / GỬI UDP
# --------------------------
def send_udp(message):
    """
    EN: Send message over UDP if enabled. Errors are ignored.
    VN: Gửi chuỗi qua UDP nếu bật. Bỏ qua lỗi mạng.
    """
    if UDP_ENABLED and sock is not None:
        try:
            sock.sendto(message.encode("utf-8"), (UDP_IP, UDP_PORT))
        except Exception:
            pass

# --------------------------
# GESTURE DETECTION FOR A SINGLE HAND
# --------------------------
def detect_gestures_for_hand(hand_id, landmarks, image_w, image_h, hand_state, other_hands_states):
    """
    EN: Main per-hand gesture logic. Returns gesture name string or None.
    VN: Hàm chính xử lý cử chỉ cho 1 bàn tay. Trả về tên cử chỉ (str) hoặc None.
    Chức năng:
    - Nhận diện các cử chỉ: Pinch, Spread, Tap, Double Tap, Drag, Flick, Press, Press+Tap.
    - Quản lý trạng thái từng bàn tay, lịch sử vị trí, vận tốc.
    - Gửi kết quả cử chỉ qua UDP.
    - Hỗ trợ nhận diện đa bàn tay (Press+Tap).
    """
    t = time.time()

    # --- extract important landmarks (normalized coords from MediaPipe)
    wrist = norm_point(landmarks[0])
    index_tip = norm_point(landmarks[8])
    thumb_tip = norm_point(landmarks[4])
    middle_tip = norm_point(landmarks[12])

    # update per-hand history (positions + timestamps)
    hand_state.update_hist(wrist, index_tip, thumb_tip, t)

    # compute normalized thumb-index distance and index speed
    d_thumb_index = np.linalg.norm(thumb_tip - index_tip)
    speed_index = hand_state.speed_index()  # normalized units/sec
    wrist_speed = hand_state.avg_wrist_speed()

    # ---- Pinch / Spread detection
    if d_thumb_index < PINCH_THRESH:
        # EN: thumb and index very close -> Pinch
        # VN: ngón cái và trỏ gần nhau => Pinch
        if not hand_state.is_pinched:
            hand_state.is_pinched = True
            hand_state.is_spread = False
            hand_state.last_gesture = "Pinch"
            send_udp(f"HAND{hand_id}:PINCH")
            return "Pinch"
    elif d_thumb_index > SPREAD_THRESH:
        # EN: fingers far apart -> Spread
        # VN: ngón tách ra -> Spread (xòe)
        if not hand_state.is_spread:
            hand_state.is_spread = True
            hand_state.is_pinched = False
            hand_state.last_gesture = "Spread"
            send_udp(f"HAND{hand_id}:SPREAD")
            return "Spread"
    else:
        # EN: between thresholds -> neutral
        # VN: ở giữa -> none
        hand_state.is_pinched = False
        hand_state.is_spread = False

    # ---- Press (hold nearly stationary)
    movement = 0.0
    if len(hand_state.index_history) >= 2:
        p0, t0 = hand_state.index_history[0]
        p1, t1 = hand_state.index_history[-1]
        movement = np.linalg.norm(p1 - p0)

    # EN: If index hardly moves and wrist is not moving much over time -> Press candidate.
    # VN: Nếu ngón trỏ gần như đứng yên và cổ tay không dịch chuyển -> coi là press candidate.
    if movement < TAP_MAX_MOVEMENT and wrist_speed < DRAG_SPEED_MAX:
        if hand_state.press_start is None:
            hand_state.press_start = t
        else:
            if t - hand_state.press_start >= PRESS_TIME:
                # recognized Press
                if hand_state.last_gesture != "Press":
                    hand_state.last_gesture = "Press"
                    send_udp(f"HAND{hand_id}:PRESS")
                    return "Press"
    else:
        hand_state.press_start = None

    # ---- Tap & Double Tap detection (heuristic)
    # EN: We'll treat a short near-zero movement epoch as a tap.
    # VN: Ta coi 1 epoch ngắn di chuyển gần 0 là tap.
    if movement < TAP_MAX_MOVEMENT:
        if not hand_state.tap_in_progress:
            hand_state.tap_in_progress = True
            hand_state.tap_started_at = t
    else:
        if hand_state.tap_in_progress:
            duration = t - hand_state.tap_started_at
            hand_state.tap_in_progress = False
            if duration <= TAP_MAX_DURATION:
                # a tap occurred
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
    if speed_index >= FLICK_SPEED_MIN:
        # EN: fast movement -> Flick. Determine left/right by x component.
        # VN: di chuyển nhanh -> Flick, xác định trái/phải bằng thành phần x.
        v = hand_state.velocity_index()
        dir_x = "Right" if v[0] > 0 else "Left"
        hand_state.last_gesture = f"Flick-{dir_x}"
        send_udp(f"HAND{hand_id}:FLICK:{dir_x}")
        return f"Flick-{dir_x}"

    # Drag: moderate speed
    if 0.05 < speed_index < DRAG_SPEED_MAX and movement > TAP_MAX_MOVEMENT:
        # EN: moderate index speed -> Drag
        # VN: tốc độ trung bình -> Drag
        hand_state.last_gesture = "Drag"
        send_udp(f"HAND{hand_id}:DRAG")
        return "Drag"

    # ---- Press + Tap (two-hand interaction): if another hand is holding (press) and this hand taps -> Press+Tap
    for other in other_hands_states:
        if other is None:
            continue
        if other.press_start is not None and time.time() - other.press_start >= PRESS_TIME:
            # EN: other hand is pressing; if this hand had a recent tap -> Press+Tap
            # VN: tay kia đang press; nếu tay này vừa tap -> Press+Tap
            # We approximate using last_gesture or last_tap_time
            if hand_state.last_gesture == "Tap" or (time.time() - hand_state.last_tap_time <= DOUBLE_TAP_INTERVAL):
                hand_state.last_gesture = "Press+Tap"
                send_udp(f"HAND{hand_id}:PRESS+TAP")
                return "Press+Tap"

    # If nothing detected return None
    return None

# --------------------------
# MAIN LOOP / VÒNG LẶP CHÍNH
# --------------------------
print("Starting Gesture Detector. Press ESC to quit.")
cv2.namedWindow("Gesture Detector", cv2.WINDOW_NORMAL)
# EN: optional fullscreen property (some systems may ignore). / VN: đặt fullscreen (tùy hệ thống).
try:
    cv2.setWindowProperty("Gesture Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
except Exception:
    pass




# ...existing code...

# --- Mouse click event for Tap & Double Tap ---
mouse_last_click_time = 0
mouse_click_pos = None
mouse_gesture = None

def mouse_callback(event, x, y, flags, param):
    global mouse_last_click_time, mouse_click_pos, mouse_gesture
    if event == cv2.EVENT_LBUTTONDOWN:
        now = time.time()
        if now - mouse_last_click_time < DOUBLE_TAP_INTERVAL:
            mouse_gesture = "Double Tap"
            mouse_last_click_time = 0
        else:
            mouse_gesture = "Tap"
            mouse_last_click_time = now
        mouse_click_pos = (x, y)

cv2.namedWindow("Gesture Detector", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Gesture Detector", mouse_callback)

# ...existing code...














while True:
    ret, frame = cap.read()
    if not ret:
        break
    # EN: mirror the frame for natural interaction (like a mirror).
    # VN: lật ngang cho cảm giác tương tác tự nhiên.
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_module.process(img_rgb)
    now = time.time()

    # remove stale hand states (if not seen > 1s)
    stale_keys = [k for k, st in hand_states.items() if now - st.last_seen > 1.0]
    for k in stale_keys:
        del hand_states[k]

    gesture_texts = []

    if res.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
            # EN: use detected index as hand label (0..n-1). / VN: dùng i làm label đơn giản.
            hand_label = i
            if hand_label not in hand_states:
                hand_states[hand_label] = HandState(hand_label)
            hs = hand_states[hand_label]

            # draw skeleton on frame for debugging / visualization
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # detect gestures for this hand (pass other hand states for multi-hand logic)
            gesture = detect_gestures_for_hand(
                hand_label,
                hand_landmarks.landmark,
                w, h,
                hs,
                [st for k, st in hand_states.items() if k != hand_label]
            )
            if gesture:
                gesture_texts.append((hand_label, gesture))
                # EN: print gesture near index fingertip on screen
                # VN: hiển thị text gần đầu ngón trỏ
                idx = hand_landmarks.landmark[8]
                cx, cy = int(idx.x * w), int(idx.y * h)
                cv2.putText(frame, gesture, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # overlay summary of active gestures on top-left
    y0 = 30
    for (hid, gt) in gesture_texts:
        cv2.putText(frame, f"H{hid}: {gt}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y0 += 30




# ...existing code...

    # Phát hiện vật màu đen
    mask, contours = detect_black_object(frame)

    # Vẽ các vùng màu đen được phát hiện
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # EN: Ignore small areas / VN: Bỏ qua vùng quá nhỏ
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ khung quanh vật màu đen

            # Kiểm tra Tap hoặc Double Tap
            now = time.time()
            if now - mouse_last_click_time < DOUBLE_TAP_INTERVAL:
                mouse_gesture = "Double Tap"
                mouse_last_click_time = 0
            else:
                mouse_gesture = "Tap"
                mouse_last_click_time = now
            mouse_click_pos = (x + w // 2, y + h // 2)  # Vị trí trung tâm của vật màu đen

    # Hiển thị Tap/Double Tap khi phát hiện
    if mouse_gesture and mouse_click_pos:
        cv2.putText(frame, mouse_gesture, mouse_click_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # Reset sau khi hiển thị
        mouse_gesture = None
        mouse_click_pos = None

















# ...existing code...

    # Hiển thị Tap/Double Tap khi click chuột
    if mouse_gesture and mouse_click_pos:
        cv2.putText(frame, mouse_gesture, mouse_click_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # Reset sau khi hiển thị
        mouse_gesture = None
        mouse_click_pos = None

# ...existing code...



















    # instructions / tuning tips text at bottom
    cv2.putText(frame,
                "Gestures: Tap, Double Tap, Drag, Flick, Pinch, Spread, Press, Press+Tap  |  ESC to quit",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Gesture Detector", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# --------------------------
# MAP GESTURE -> ACTION
# --------------------------
def action_from_gesture(gesture):
    """
    EN: Map a gesture string to a high-level action (e.g., LIGHT_ON).
    VN: Map cử chỉ sang hành động (ví dụ LIGHT_ON).
    Note: gesture param may be "Pinch" or "PINCH" etc; normalize to upper.
    Chức năng:
    - Chuyển đổi tên cử chỉ sang hành động thực tế (bật/tắt đèn, chuyển bài hát, chọn thiết bị,...)
    - Có thể mở rộng để tích hợp với hệ thống nhà thông minh hoặc các ứng dụng khác.
    """
    if gesture is None:
        return None
    g = gesture.upper().replace(" ", "_")  # EN: normalize e.g. "Double Tap" -> "DOUBLE_TAP"
    # VN: chuẩn hóa chuỗi cử chỉ

    if g == "PINCH":
        return "LIGHT_ON"          # EN: pinch -> turn on light. / VN: pinch -> bật đèn
    if g == "SPREAD":
        return "LIGHT_OFF"         # EN: spread -> turn off light. / VN: xòe -> tắt đèn
    if g == "TAP":
        return "NEXT_SONG"         # EN: tap -> next song. / VN: tap -> chuyển bài
    if g == "DOUBLE_TAP" or g == "DOUBLE-TAP":
        return "PREV_SONG"         # EN: double tap -> previous song. / VN: double tap -> bài trước
    if g.startswith("FLICK"):
        # EN: could be FLICK-LEFT or FLICK-Right -> choose devices accordingly
        if "RIGHT" in g:
            return "SELECT_DEVICE_RIGHT"
        else:
            return "SELECT_DEVICE_LEFT"
    if g == "DRAG":
        return "DRAG_MOVE"
    if g == "PRESS":
        return "PRESS_HOLD"
    if g == "PRESS_TAP" or g == "PRESS+TAP":
        return "SPECIAL_PRESS_TAP"

    return None

# Example usage:
# EN: you can call action_from_gesture("Pinch") to get "LIGHT_ON".
# VN: gọi action_from_gesture("Pinch") sẽ trả "LIGHT_ON".

# --------------------------
# TÓM TẮT CHỨC NĂNG FILE
# --------------------------
# 1. Khởi tạo camera, MediaPipe, UDP socket.
# 2. Nhận diện bàn tay, lấy landmark, lưu lịch sử vị trí/vận tốc.
# 3. Nhận diện các cử chỉ: Pinch, Spread, Tap, Double Tap, Drag, Flick, Press, Press+Tap.
# 4. Hiển thị cử chỉ lên màn hình, gửi kết quả qua UDP.
# 5. Ánh xạ cử chỉ sang hành động điều khiển thiết bị thông minh.