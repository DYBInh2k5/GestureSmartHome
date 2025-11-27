import cv2
import mediapipe as mp
import time
import math
import numpy as np

# --------------------------
# CONFIGURATION / CẤU HÌNH
# --------------------------
CAM_ID = 0  # Webcam ID
MIN_DET_CONF = 0.6  # MediaPipe detection confidence
MIN_TRACK_CONF = 0.5  # MediaPipe tracking confidence

# Gesture thresholds
PINCH_THRESH = 0.06
SPREAD_THRESH = 0.10
TAP_MAX_MOVEMENT = 0.03
TAP_MAX_DURATION = 0.25
DOUBLE_TAP_INTERVAL = 0.4
PRESS_TIME = 0.6
DRAG_SPEED_MAX = 0.6
FLICK_SPEED_MIN = 1.2
HISTORY_LEN = 8

# --------------------------
# SETUP MediaPipe, Camera
# --------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands_module = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
)

cap = cv2.VideoCapture(CAM_ID)

# --------------------------
# HELPERS / HỖ TRỢ
# --------------------------

def dist(a, b):
    """Calculate Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def norm_point(lm):
    """Convert MediaPipe landmark to numpy array."""
    return np.array([lm.x, lm.y], dtype=np.float32)

class HandState:
    """Track state and history for each hand."""
    def __init__(self, hand_id):
        self.id = hand_id
        self.last_seen = time.time()
        self.history = []
        self.index_history = []
        self.thumb_history = []
        self.press_start = None
        self.last_tap_time = 0
        self.tap_in_progress = False
        self.tap_started_at = 0
        self.is_pinched = False
        self.is_spread = False
        self.last_gesture = None

    def update_hist(self, wrist_pt, index_pt, thumb_pt, t):
        """Update history for hand landmarks."""
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
        """Calculate velocity vector of index fingertip."""
        if len(self.index_history) < 2:
            return np.array([0.0, 0.0])
        p0, t0 = self.index_history[0]
        p1, t1 = self.index_history[-1]
        dt = t1 - t0
        if dt <= 0:
            return np.array([0.0, 0.0])
        return (p1 - p0) / dt

    def speed_index(self):
        """Calculate speed magnitude of index fingertip."""
        v = self.velocity_index()
        return np.linalg.norm(v)

    def avg_wrist_speed(self):
        """Calculate average wrist speed."""
        if len(self.history) < 2:
            return 0.0
        p0, t0 = self.history[0]
        p1, t1 = self.history[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return np.linalg.norm(p1 - p0) / dt

hand_states = {}

# --------------------------
# ACTIONS / HÀNH ĐỘNG
# --------------------------

def perform_action(gesture):
    """Perform actions based on detected gestures."""
    if gesture == "Tap":
        print("Action: Turn ON the light")
    elif gesture == "Double Tap":
        print("Action: Turn OFF the light")
    elif gesture == "Drag":
        print("Action: Change song")
    elif gesture.startswith("Flick"):
        print("Action: Increase brightness")
    elif gesture == "Pinch":
        print("Action: Zoom out")
    elif gesture == "Spread":
        print("Action: Zoom in")
    elif gesture == "Press":
        print("Action: Warning!")
    elif gesture == "Press+Tap":
        print("Action: Thank you!")

# --------------------------
# GESTURE DETECTION
# --------------------------

def detect_gestures_for_hand(hand_id, landmarks, image_w, image_h, hand_state, other_hands_states):
    """Detect gestures for a single hand."""
    t = time.time()

    wrist = norm_point(landmarks[0])
    index_tip = norm_point(landmarks[8])
    thumb_tip = norm_point(landmarks[4])
    middle_tip = norm_point(landmarks[12])

    hand_state.update_hist(wrist, index_tip, thumb_tip, t)

    d_thumb_index = np.linalg.norm(thumb_tip - index_tip)
    speed_index = hand_state.speed_index()
    wrist_speed = hand_state.avg_wrist_speed()

    # Pinch / Spread detection
    if d_thumb_index < PINCH_THRESH:
        if not hand_state.is_pinched:
            hand_state.is_pinched = True
            hand_state.is_spread = False
            hand_state.last_gesture = "Pinch"
            return "Pinch"
    elif d_thumb_index > SPREAD_THRESH:
        if not hand_state.is_spread:
            hand_state.is_spread = True
            hand_state.is_pinched = False
            hand_state.last_gesture = "Spread"
            return "Spread"
    else:
        hand_state.is_pinched = False
        hand_state.is_spread = False

    # Press detection
    movement = 0.0
    if len(hand_state.index_history) >= 2:
        p0, t0 = hand_state.index_history[0]
        p1, t1 = hand_state.index_history[-1]
        movement = np.linalg.norm(p1 - p0)

    if movement < TAP_MAX_MOVEMENT and wrist_speed < DRAG_SPEED_MAX:
        if hand_state.press_start is None:
            hand_state.press_start = t
        else:
            if t - hand_state.press_start >= PRESS_TIME:
                if hand_state.last_gesture != "Press":
                    hand_state.last_gesture = "Press"
                    return "Press"
    else:
        hand_state.press_start = None

    # Tap & Double Tap detection
    if movement < TAP_MAX_MOVEMENT:
        if not hand_state.tap_in_progress:
            hand_state.tap_in_progress = True
            hand_state.tap_started_at = t
    else:
        if hand_state.tap_in_progress:
            duration = t - hand_state.tap_started_at
            hand_state.tap_in_progress = False
            if duration <= TAP_MAX_DURATION:
                if t - hand_state.last_tap_time <= DOUBLE_TAP_INTERVAL:
                    hand_state.last_tap_time = 0
                    hand_state.last_gesture = "Double Tap"
                    return "Double Tap"
                else:
                    hand_state.last_tap_time = t
                    hand_state.last_gesture = "Tap"
                    return "Tap"

    # Flick detection
    if speed_index >= FLICK_SPEED_MIN:
        v = hand_state.velocity_index()
        dir_x = "Right" if v[0] > 0 else "Left"
        hand_state.last_gesture = f"Flick-{dir_x}"
        return f"Flick-{dir_x}"

    # Drag detection
    if 0.05 < speed_index < DRAG_SPEED_MAX:
        hand_state.last_gesture = "Drag"
        return "Drag"

    # Press + Tap detection
    for other in other_hands_states:
        if other is None:
            continue
        if other.press_start is not None and time.time() - other.press_start >= PRESS_TIME:
            if hand_state.last_gesture == "Tap" or (time.time() - hand_state.last_tap_time <= DOUBLE_TAP_INTERVAL):
                hand_state.last_gesture = "Press+Tap"
                return "Press+Tap"

    return None

# --------------------------
# MAIN LOOP
# --------------------------

print("Starting Gesture Detector. Press ESC to quit.")
cv2.namedWindow("Gesture Detector", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_module.process(img_rgb)
    now = time.time()

    stale_keys = [k for k, st in hand_states.items() if now - st.last_seen > 1.0]
    for k in stale_keys:
        del hand_states[k]

    gesture_texts = []

    if res.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
            hand_label = i
            if hand_label not in hand_states:
                hand_states[hand_label] = HandState(hand_label)
            hs = hand_states[hand_label]

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gestures_for_hand(
                hand_label,
                hand_landmarks.landmark,
                w, h,
                hs,
                [st for k, st in hand_states.items() if k != hand_label]
            )
            if gesture:
                gesture_texts.append((hand_label, gesture))
                idx = hand_landmarks.landmark[8]
                cx, cy = int(idx.x * w), int(idx.y * h)
                cv2.putText(frame, gesture, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                perform_action(gesture)  # Thực hiện hành động

    y0 = 30
    for (hid, gt) in gesture_texts:
        cv2.putText(frame, f"H{hid}: {gt}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y0 += 30

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