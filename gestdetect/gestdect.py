import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import math
import urllib.request   
import os
import socket
import json
import argparse
import numpy as np
import threading 
import winsound
from collections import deque
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  QNX alert target (UDP) — QNX IS THE BACKBONE
#
#  All alerts AND all cancels flow ONLY through QNX.
#  If QNX is off, the dashboard receives nothing.
#  There is no direct-to-dashboard bypass.
# ─────────────────────────────────────────────
QNX_HOST        = "10.0.0.1"    # ← Set to your RPi4 QNX IP
QNX_PORT        = 5005          # alert UDP port on alert_daemon
DASHBOARD_HOST  = "127.0.0.1"   # fallback relay for local webdashboard
DASHBOARD_PORT  = 5006
ALERT_COOLDOWN  = 5.0           # seconds between repeat alerts per person

_udp_sock       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_udp_sock.setblocking(False)
_last_alert_sent: dict = {}     # {pid: {gkey: timestamp}}
_qnx_rtt_pending: dict = {}
_qnx_rtt_ms = 0.0
_qnx_rtt_ok = False
_udp_send_ms = 0.0
_udp_seq = 0
_event_log = deque(maxlen=6)
FIST_SCORE_THRESHOLD = 0.50   # tunable via --fist-score

GESTURE_MODEL_PATH = "gesture_recognizer.task"
GESTURE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/latest/"
    "gesture_recognizer.task"
)

HAND_CONNECTIONS = (
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
)


def is_fist_from_gesture(gesture_label: str, score: float) -> bool:
    """Classifier-based closed-fist detection from GestureRecognizer output."""
    return gesture_label == "Closed_Fist" and score > FIST_SCORE_THRESHOLD


def _draw_hand(frame, hand_lm, fw: int, fh: int, fist: bool) -> None:
    pts_h    = [(int(lm.x * fw), int(lm.y * fh)) for lm in hand_lm]
    ln_col   = (0, 60, 255)   if fist else (0, 210, 255)
    dot_col  = (0, 40, 255)   if fist else (0, 230, 255)
    tip_col  = (0, 20, 220)   if fist else (0, 255, 200)
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(frame, pts_h[p1], pts_h[p2], ln_col, 2, cv2.LINE_AA)
    for i, p in enumerate(pts_h):
        is_tip = i in (4, 8, 12, 16, 20)
        r = 7 if is_tip else 4
        cv2.circle(frame, p, r, tip_col if is_tip else dot_col, -1)
        cv2.circle(frame, p, r + 1, (255, 255, 255), 1)
    label   = "FIST" if fist else "open"
    lbl_col = (0, 60, 255) if fist else (160, 160, 160)
    cv2.putText(frame, label, (pts_h[0][0] + 10, pts_h[0][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, lbl_col, 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergency gesture detection")
    parser.add_argument("--model",  choices=["lite", "full", "heavy"],
                        default=os.getenv("GESTURE_MODEL", "heavy"),
                        help="Deprecated (MediaPipe pose model selector); ignored in YOLO mode")
    parser.add_argument("--pose-model",
                        default=os.getenv("GESTURE_POSE_MODEL", "yolov8s-pose.pt"),
                        help="YOLO pose model path/name (e.g. yolov8s-pose.pt)")
    parser.add_argument("--pose-conf", type=float,
                        default=float(os.getenv("GESTURE_POSE_CONF", "0.60")),
                        help="YOLO pose confidence threshold")
    parser.add_argument("--imgsz", type=int,
                        default=int(os.getenv("GESTURE_POSE_IMGSZ", "416")),
                        help="YOLO inference image size (lower = faster)")
    parser.add_argument("--pose-scale", type=float,
                        default=float(os.getenv("GESTURE_POSE_SCALE", "0.75")),
                        help="Pre-scale frame before YOLO inference (0.5-1.0)")
    parser.add_argument("--hand-skip", type=int,
                        default=int(os.getenv("GESTURE_HAND_SKIP", "1")),
                        help="Run hand inference every N+1 frames")
    parser.add_argument("--hand-min-conf", type=float,
                        default=float(os.getenv("GESTURE_HAND_MIN_CONF", "0.35")),
                        help="Hand detector/presence/tracking minimum confidence")
    parser.add_argument("--fist-score", type=float,
                        default=float(os.getenv("GESTURE_FIST_SCORE", "0.50")),
                        help="Closed_Fist classifier score threshold")
    parser.add_argument("--hand-infer-w", type=int,
                        default=int(os.getenv("GESTURE_HAND_INFER_W", "0")),
                        help="Override hand inference width (0 keeps preset)")
    parser.add_argument("--hand-infer-h", type=int,
                        default=int(os.getenv("GESTURE_HAND_INFER_H", "0")),
                        help="Override hand inference height (0 keeps preset)")
    parser.add_argument("--draw-hands", type=int, choices=[0, 1],
                        default=int(os.getenv("GESTURE_DRAW_HANDS", "1")),
                        help="Draw hand landmarks (0 off, 1 on)")
    parser.add_argument("--preset", choices=["pi", "balanced", "high_end"],
                        default=os.getenv("GESTURE_PRESET", "high_end"))
    parser.add_argument("--camera", type=int,
                        default=int(os.getenv("GESTURE_CAMERA", "1")),
                        help="Camera index (0=built-in, 1=USB webcam)")
    parser.add_argument("--prefer-gpu", action="store_true",
                        default=os.getenv("GESTURE_PREFER_GPU", "1") == "1")
    return parser.parse_args()


def resolve_pose_model_path(model_name: str) -> str:
    if model_name.lower().endswith(".pt"):
        engine_path = f"{os.path.splitext(model_name)[0]}.engine"
        if os.path.exists(engine_path):
            print(f"[YOLO] Found TensorRT engine, using {engine_path}")
            return engine_path
    return model_name


def create_pose_model(model_name: str):
    model_name = resolve_pose_model_path(model_name)
    try:
        model = YOLO(model_name)
        if model_name.lower().endswith(".engine"):
            print(f"[YOLO] Loaded TensorRT engine: {model_name}")
            return model
        try:
            model.to("cuda")
            model.fuse()
            device_msg = next(model.model.parameters()).device
            print(f"[YOLO] Model device: {device_msg}  (GPU enabled)")
        except (RuntimeError, AssertionError) as cuda_err:
            print(f"[YOLO] CUDA not available, using CPU. ({cuda_err})")
            print("[YOLO] For GPU: pip install torch torchvision from PyTorch CUDA index")
            device_msg = next(model.model.parameters()).device
            print(f"[YOLO] Model device: {device_msg}")
        return model
    except Exception as ex:
        raise RuntimeError(
            f"Could not load YOLO pose model '{model_name}'. "
            f"Install ultralytics and ensure model is reachable. Error: {ex}"
        ) from ex

_hand_lock            = threading.Lock()
_hand_latest_frame: list = [None]
_hand_result: list        = [None]

_pose_lock            = threading.Lock()
_pose_latest_frame: list = [None]
_pose_result: list       = [[]]

_stats_lock = threading.Lock()
_pose_fps_display = 0.0
_hand_fps_display = 0.0
_pose_ms_display = 0.0
_hand_ms_display = 0.0
_pose_done_count = 0
_hand_done_count = 0
_pose_last_stat_t = time.time()
_hand_last_stat_t = time.time()

def _hand_result_callback(result, _image, _ts):
    with _hand_lock:
        _hand_result[0] = result


def create_gesture_recognizer(model_path, prefer_gpu, num_hands=2, min_conf=0.40):
    delegates = (
        [mp_python.BaseOptions.Delegate.GPU, mp_python.BaseOptions.Delegate.CPU]
        if prefer_gpu else [mp_python.BaseOptions.Delegate.CPU]
    )
    last_err = None
    for delegate in delegates:
        try:
            base_opts = mp_python.BaseOptions(model_asset_path=model_path,
                                              delegate=delegate)
            opts = mp_vision.GestureRecognizerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_hands=num_hands,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
                min_tracking_confidence=min_conf,
                result_callback=_hand_result_callback,
            )
            return mp_vision.GestureRecognizer.create_from_options(opts), delegate.name
        except Exception as ex:
            last_err = ex
            print(f"[WARN] gesture delegate {delegate.name} failed: {ex}")
    raise RuntimeError(f"Could not create gesture recognizer: {last_err}")


def _infer_worker_hands():
    local_ts = 0
    while True:
        with _hand_lock:
            frame_bgr = _hand_latest_frame[0]
        if frame_bgr is None:
            time.sleep(0.001); continue
        t0 = time.time()
        small    = cv2.resize(frame_bgr, (HAND_INFER_W, HAND_INFER_H))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ts_ms = int(time.time() * 1000)
        if ts_ms <= local_ts: ts_ms = local_ts + 1
        local_ts = ts_ms
        hand_landmarker.recognize_async(mp_image, ts_ms)
        infer_ms = (time.time() - t0) * 1000.0
        with _stats_lock:
            global _hand_done_count, _hand_last_stat_t, _hand_fps_display, _hand_ms_display
            _hand_done_count += 1
            _hand_ms_display = _hand_ms_display * 0.85 + infer_ms * 0.15
            now_s = time.time()
            dt = now_s - _hand_last_stat_t
            if dt >= 1.0:
                _hand_fps_display = _hand_done_count / dt
                _hand_done_count = 0
                _hand_last_stat_t = now_s
        with _hand_lock:
            if _hand_latest_frame[0] is frame_bgr:
                _hand_latest_frame[0] = None


def _infer_worker_pose():
    while True:
        with _pose_lock:
            frame_bgr = _pose_latest_frame[0]
        if frame_bgr is None:
            time.sleep(0.001)
            continue

        t0 = time.time()
        poses = _run_pose_inference(frame_bgr)
        infer_ms = (time.time() - t0) * 1000.0
        with _stats_lock:
            global _pose_done_count, _pose_last_stat_t, _pose_fps_display, _pose_ms_display
            _pose_done_count += 1
            _pose_ms_display = _pose_ms_display * 0.85 + infer_ms * 0.15
            now_s = time.time()
            dt = now_s - _pose_last_stat_t
            if dt >= 1.0:
                _pose_fps_display = _pose_done_count / dt
                _pose_done_count = 0
                _pose_last_stat_t = now_s
        with _pose_lock:
            _pose_result[0] = poses
            if _pose_latest_frame[0] is frame_bgr:
                _pose_latest_frame[0] = None


ARGS = parse_args()
FIST_SCORE_THRESHOLD = max(0.05, min(0.95, ARGS.fist_score))
WEBCAM_TUNING = ARGS.camera != 0

if WEBCAM_TUNING:
    if ARGS.pose_conf > 0.55:
        ARGS.pose_conf = 0.55
    if ARGS.hand_min_conf > 0.25:
        ARGS.hand_min_conf = 0.25
    if FIST_SCORE_THRESHOLD > 0.35:
        FIST_SCORE_THRESHOLD = 0.35

# ── Presets — hand inference size is CRITICAL for fist detection ──
# The old working values (480×270 on high_end) proved reliable.
# Do not raise these without testing; MediaPipe hand detector
# works best at its trained input range, and larger isn't better.
if ARGS.preset == "pi":
    CAM_W, CAM_H, CAM_FPS = 640, 480, 30
    INFER_W, INFER_H, SKIP_FRAMES = 320, 240, 2
    HAND_INFER_W, HAND_INFER_H = 480, 360
elif ARGS.preset == "balanced":
    CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
    INFER_W, INFER_H, SKIP_FRAMES = 480, 270, 1
    HAND_INFER_W, HAND_INFER_H = 640, 360
else:
    CAM_W, CAM_H, CAM_FPS = 1280, 720, 60
    INFER_W, INFER_H, SKIP_FRAMES = 960, 540, 1
    HAND_INFER_W, HAND_INFER_H = 480, 270   # proven working size

if ARGS.hand_infer_w > 0 and ARGS.hand_infer_h > 0:
    HAND_INFER_W, HAND_INFER_H = ARGS.hand_infer_w, ARGS.hand_infer_h

POSE_MODEL_PATH = resolve_pose_model_path(ARGS.pose_model)
pose_model = create_pose_model(POSE_MODEL_PATH)
if POSE_MODEL_PATH.lower().endswith(".engine"):
    POSE_DEVICE = "0"
else:
    POSE_DEVICE = "0" if next(pose_model.model.parameters()).is_cuda else "cpu"
if not os.path.exists(GESTURE_MODEL_PATH):
    print("Downloading gesture recognizer model (~8 MB)...")
    urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)
    print("Download complete.")
hand_landmarker, hand_delegate = create_gesture_recognizer(GESTURE_MODEL_PATH,
                                                            ARGS.prefer_gpu,
                                                            min_conf=ARGS.hand_min_conf)
print(f"[CONFIG] preset={ARGS.preset}  pose_model={POSE_MODEL_PATH}  "
    f"pose_conf={ARGS.pose_conf:.2f}  imgsz={ARGS.imgsz}  pose_scale={ARGS.pose_scale:.2f}  "
    f"cam={CAM_W}x{CAM_H}@{CAM_FPS}  hand_skip={ARGS.hand_skip}  draw_hands={ARGS.draw_hands}  "
    f"hand_infer={HAND_INFER_W}x{HAND_INFER_H}  hand_min_conf={ARGS.hand_min_conf:.2f}  "
    f"fist_score={FIST_SCORE_THRESHOLD:.2f}  hand_delegate={hand_delegate}")
print(f"[UDP]    QNX target → {QNX_HOST}:{QNX_PORT}  (QNX is mandatory backbone)")

PERSON_COLORS = [
    (0, 255, 255),
    (255, 100, 255),
    (100, 255, 100),
    (255, 200, 0),
    (200, 100, 255),
    (0, 200, 255),
]


def pid_color(pid: int):
    return PERSON_COLORS[(pid - 1) % len(PERSON_COLORS)]

def _configure_camera(cam_obj):
    cam_obj.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
    cam_obj.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cam_obj.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cam_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cam_obj.set(cv2.CAP_PROP_FPS,          CAM_FPS)


def _open_camera_with_fallback(primary_idx):
    candidates = [primary_idx] + [i for i in (0, 1) if i != primary_idx]
    for idx in candidates:
        cap_try = cv2.VideoCapture(idx)
        if not cap_try.isOpened():
            cap_try.release()
            continue
        _configure_camera(cap_try)
        got_frame = False
        for _ in range(20):
            ok, _ = cap_try.read()
            if ok:
                got_frame = True
                break
            time.sleep(0.01)
        if got_frame:
            return cap_try, idx
        cap_try.release()
    return None, None


cap, active_camera_index = _open_camera_with_fallback(ARGS.camera)
if cap is None:
    raise RuntimeError(
        "Could not open any camera (tried indexes 0 and 1). "
        "Close apps that use the camera and retry with --camera 0 or --camera 1."
    )

print(f"[CAM] Using camera index {active_camera_index} — change with --camera 0 or --camera 1")

# ── Tunable constants ─────────────────────────────────────────────────────────
ALERT_TIME     = 3.0
MIN_VISIBILITY = 0.20
TRACK_MAX_DIST = 150
CANCEL_DEBOUNCE_AFTER_GESTURE = 1.5
CANCEL_REQUIRED_FISTS = 1
TARGET_RENDER_FPS = 30.0

if WEBCAM_TUNING:
    CANCEL_DEBOUNCE_AFTER_GESTURE = 0.55
    print(f"[CAM] Webcam cancel tuning: debounce={CANCEL_DEBOUNCE_AFTER_GESTURE:.2f}s")

IDX = {
    "NOSE": 0,
    "L_EAR": 3, "R_EAR": 4,
    "LS": 5, "RS": 6,
    "LE": 7, "RE": 8,
    "LW": 9, "RW": 10,
    "L_HIP": 11, "R_HIP": 12,
}

COCO_CONNECTIONS = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
]

def _detect_ambulance(pts, sw):
    lw, rw, ls, rs = pts["lw"], pts["rw"], pts["ls"], pts["rs"]
    mid_x = (ls[0] + rs[0]) / 2
    shoulder_y = (ls[1] + rs[1]) / 2
    if ls[0] > rs[0]:
        lw_crossed = lw[0] < mid_x
        rw_crossed = rw[0] > mid_x
    else:
        lw_crossed = lw[0] > mid_x
        rw_crossed = rw[0] < mid_x
    wrist_y = (lw[1] + rw[1]) / 2
    at_chest = wrist_y > shoulder_y - 0.30 * sw and wrist_y < shoulder_y + 1.20 * sw
    wrists_close = dist(lw, rw) / sw < 1.40
    return lw_crossed and rw_crossed and at_chest and wrists_close

def _detect_police(pts, sw):
    lw, rw       = pts["lw"], pts["rw"]
    l_ear, r_ear = pts["lear"], pts["rear"]
    ear_y  = min(l_ear[1], r_ear[1])
    return lw[1] < ear_y and rw[1] < ear_y and dist(lw, rw) / sw > 0.80

def _detect_fire(pts, sw):
    lw, rw, ls, rs = pts["lw"], pts["rw"], pts["ls"], pts["rs"]
    shoulder_y = (ls[1] + rs[1]) / 2
    horiz_tol = 0.55 * sw
    left_horiz = abs(lw[1] - shoulder_y) < horiz_tol
    right_horiz = abs(rw[1] - shoulder_y) < horiz_tol
    wrist_span_x = abs(lw[0] - rw[0])
    wide = wrist_span_x / sw > 1.80
    shoulder_xmin = min(ls[0], rs[0])
    shoulder_xmax = max(ls[0], rs[0])
    outside = (min(lw[0], rw[0]) < shoulder_xmin + 0.10 * sw and
               max(lw[0], rw[0]) > shoulder_xmax - 0.10 * sw)
    return left_horiz and right_horiz and wide and outside

def _detect_distress(pts, sw):
    lw, rw         = pts["lw"], pts["rw"]
    le, re         = pts["le"], pts["re"]
    l_ear, r_ear   = pts["lear"], pts["rear"]
    ear_avg_y      = (l_ear[1] + r_ear[1]) / 2
    return (dist(lw, l_ear) / sw < 0.55 and dist(rw, r_ear) / sw < 0.55 and
            lw[1] >= ear_avg_y - 0.15 * sw and rw[1] >= ear_avg_y - 0.15 * sw and
            dist(le, re) / sw > 1.20)

GESTURES = [
    {"key": "AMBULANCE", "label": "AMBULANCE", "color": (0,   0,   255),
     "alert": "!! MEDICAL EMERGENCY !!", "detect": _detect_ambulance},
    {"key": "POLICE",    "label": "POLICE",    "color": (255, 80,  0),
     "alert": "!! POLICE / SECURITY !!", "detect": _detect_police},
    {"key": "FIRE",      "label": "FIRE",      "color": (0,   140, 255),
     "alert": "!! FIRE / DISASTER !!",  "detect": _detect_fire},
    {"key": "DISTRESS",  "label": "DISTRESS",  "color": (255, 0,   180),
     "alert": "!! DISTRESS / TRAPPED !!","detect": _detect_distress},
]

# ── Tracker ───────────────────────────────────────────────────────────────────
GHOST_FRAMES = 15

class PersonTracker:
    def __init__(self, max_dist=TRACK_MAX_DIST):
        self.next_id  = 1
        self.tracks   = {}
        self.max_dist = max_dist

    def _iou(self, a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0: return 0.0
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

    def update(self, detections):
        for t in self.tracks.values(): t["missing"] += 1
        self.tracks = {p: t for p, t in self.tracks.items()
                       if t["missing"] <= GHOST_FRAMES}
        if not detections: return {}
        if not self.tracks:
            result = {}
            for det in detections:
                pid = self.next_id
                self.tracks[pid] = {**det, "missing": 0}
                result[pid] = det
                self.next_id += 1
            return result
        used_pids = set(); result = {}
        for det in detections:
            best_pid = None; best_score = -1.0
            for pid, track in self.tracks.items():
                if pid in used_pids: continue
                iou   = self._iou(det["bbox"], track["bbox"])
                cx,cy = det["centroid"]; tx,ty = track["centroid"]
                cdist = math.hypot(cx-tx, cy-ty)
                score = iou * 10 - (cdist / self.max_dist)
                if score > best_score and cdist < self.max_dist:
                    best_score = score; best_pid = pid
            if best_pid is None:
                best_pid = self.next_id; self.next_id += 1
            self.tracks[best_pid] = {**det, "missing": 0}
            result[best_pid] = det; used_pids.add(best_pid)
        return result

# ── Helpers ───────────────────────────────────────────────────────────────────
def dist(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
def pt(lm, w, h): return (int(lm.x * w), int(lm.y * h))

def draw_skeleton(frame, pose, line_color=(180, 255, 0), point_color=(0, 255, 180)):
    kpts = pose["kpts"]
    conf = pose["kconf"]
    for a, b in COCO_CONNECTIONS:
        if conf[a] > 0.20 and conf[b] > 0.20:
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pa, pb, line_color, 2, cv2.LINE_AA)
    for i in range(len(kpts)):
        if conf[i] > 0.20:
            p = (int(kpts[i][0]), int(kpts[i][1]))
            cv2.circle(frame, p, 4, point_color, -1)

def fill_alpha(frame, x1, y1, x2, y2, color, alpha=0.45):
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)
    if x2<=x1 or y2<=y1: return
    roi = frame[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color, dtype=np.uint8), alpha,
                    roi, 1.0-alpha, 0, roi)


def _push_event(person: str, label: str, status: str):
    _event_log.appendleft({
        "ts": time.strftime("%H:%M:%S", time.localtime()),
        "person": person,
        "label": label,
        "status": status,
    })


def draw_event_ticker(frame):
    if not _event_log:
        return

    h, w = frame.shape[:2]
    panel_w = 340
    panel_x1 = max(0, w - panel_w - 10)
    panel_x2 = w - 10
    header_h = 24
    row_h = 22
    panel_h = header_h + len(_event_log) * row_h + 10
    panel_y1 = min(max(70, h // 2 - panel_h // 2), max(0, h - panel_h - 70))
    panel_y2 = panel_y1 + panel_h

    fill_alpha(frame, panel_x1, panel_y1, panel_x2, panel_y2, (5, 5, 20), 0.58)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (110, 110, 140), 1)
    cv2.putText(frame, "EVENT TIMELINE", (panel_x1 + 10, panel_y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 245), 1, cv2.LINE_AA)

    y = panel_y1 + header_h
    for idx, ev in enumerate(_event_log):
        fade = max(0.35, 1.0 - idx * 0.14)
        txt_col = tuple(int(220 * fade) for _ in range(3))
        if ev["status"].startswith("✓"):
            st_col = (0, int(230 * fade), int(120 * fade))
        elif ev["status"].startswith("x"):
            st_col = (0, int(120 * fade), int(230 * fade))
        else:
            st_col = (int(180 * fade), int(180 * fade), int(180 * fade))

        cv2.putText(frame, ev["ts"], (panel_x1 + 10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, txt_col, 1, cv2.LINE_AA)
        cv2.putText(frame, ev["person"], (panel_x1 + 90, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, txt_col, 1, cv2.LINE_AA)
        cv2.putText(frame, ev["label"], (panel_x1 + 130, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, txt_col, 1, cv2.LINE_AA)
        cv2.putText(frame, ev["status"], (panel_x2 - 74, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, st_col, 1, cv2.LINE_AA)
        y += row_h

def draw_hold_bar(frame, cx, top_y, elapsed, total, pid, label, color):
    bar_w, bar_h = 280, 22
    bar_x = max(0, cx - bar_w // 2); bar_y = max(30, top_y)
    ratio = min(elapsed / total, 1.0)
    fill  = color if ratio < 1.0 else (0, 255, 0)
    fill_alpha(frame, bar_x-4, bar_y-22, bar_x+bar_w+4, bar_y+bar_h+4, (0,0,0), 0.5)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*ratio), bar_y+bar_h), fill, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (220,220,220), 1)
    cv2.putText(frame, f"P{pid}  {label}  {elapsed:.1f}s / {total:.0f}s",
                (bar_x, bar_y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

def get_bbox(person_lm, fw, fh, pad=0.12):
    xs = [lm.x for lm in person_lm if lm.visibility > 0.1]
    ys = [lm.y for lm in person_lm if lm.visibility > 0.1]
    if not xs: return (0, 0, fw, fh)
    rx1,ry1 = min(xs),min(ys); rx2,ry2 = max(xs),max(ys)
    bw=rx2-rx1; bh=ry2-ry1
    return (int(max(0,(rx1-bw*pad)*fw)), int(max(0,(ry1-bh*pad)*fh)),
            int(min(fw,(rx2+bw*pad)*fw)), int(min(fh,(ry2+bh*pad)*fh)))

def assign_hands_to_people(hand_landmarks, id_map, fw, fh):
    assignments = {pid: [] for pid in id_map.keys()}
    if not hand_landmarks or not id_map:
        return assignments

    person_items = list(id_map.items())
    for hand_idx, hlm in enumerate(hand_landmarks):
        wx, wy = int(hlm[0].x * fw), int(hlm[0].y * fh)
        best_pid = None
        best_dist = float("inf")
        for pid, det in person_items:
            cx, cy = det["centroid"]
            d = math.hypot(wx - cx, wy - cy)
            if d < best_dist:
                best_dist = d
                best_pid = pid
        if best_pid is not None:
            assignments[best_pid].append((hand_idx, hlm))

    return assignments


def _run_pose_inference(frame_bgr):
    inf_frame = frame_bgr
    sx = 1.0
    sy = 1.0

    result = pose_model(inf_frame,
                        device=POSE_DEVICE,
                        conf=ARGS.pose_conf,
                        verbose=False,
                        imgsz=_runtime_imgsz,
                        half=(POSE_DEVICE != "cpu"),
                        max_det=8)[0]
    if result.keypoints is None or result.keypoints.xy is None:
        return []

    kxy = result.keypoints.xy.cpu().numpy()
    kxy[..., 0] *= sx
    kxy[..., 1] *= sy
    if result.keypoints.conf is not None:
        kcf = result.keypoints.conf.cpu().numpy()
    else:
        kcf = np.ones((kxy.shape[0], kxy.shape[1]), dtype=np.float32)

    if result.boxes is not None and result.boxes.xyxy is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
    else:
        boxes = np.zeros((kxy.shape[0], 4), dtype=np.float32)

    poses = []
    count = min(len(kxy), len(boxes))
    for i in range(count):
        x1, y1, x2, y2 = boxes[i]
        poses.append({
            "bbox": (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
            "kpts": kxy[i],
            "kconf": kcf[i],
        })
    return poses

# ── State ─────────────────────────────────────────────────────────────────────
tracker            = PersonTracker()
gesture_timers     = {}
_kpt_smooth        = {}
CANCEL_TIME        = 2.0
active_alerts      = {}
active_alert_times = {}
cancel_timers      = {}
_cancel_fist_last  = {}
_last_gesture_seen = {}
KPT_SMOOTH_ALPHA   = 0.40
GESTURE_GRACE      = 0.35
CANCEL_GRACE       = 0.30

if WEBCAM_TUNING:
    KPT_SMOOTH_ALPHA = 0.28
    GESTURE_GRACE = 0.50
    CANCEL_GRACE = 0.75
    print("[CAM] Webcam tuning enabled: pose_conf=%.2f hand_min_conf=%.2f fist_score=%.2f smooth=%.2f grace=%.2fs" % (
        ARGS.pose_conf, ARGS.hand_min_conf, FIST_SCORE_THRESHOLD, KPT_SMOOTH_ALPHA, GESTURE_GRACE
    ))


def smooth_keypoints(pid, kpts, kconf):
    if pid not in _kpt_smooth or _kpt_smooth[pid].shape != kpts.shape:
        _kpt_smooth[pid] = kpts.copy()
        return kpts
    prev = _kpt_smooth[pid]
    conf_mask = (kconf > 0.20).reshape(-1, 1)
    updated = prev * (1.0 - KPT_SMOOTH_ALPHA) + kpts * KPT_SMOOTH_ALPHA
    smoothed = np.where(conf_mask, updated, prev)
    _kpt_smooth[pid] = smoothed
    return smoothed

WINDOW_NAME = "Emergency Gesture Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

_frame_count = 0; _fps_time = time.time(); _fps_display = 0.0
_frame_serial = 0
_hand_visual_cache = ([], [])
_hand_visual_last_t = 0.0
HAND_VISUAL_TTL = 0.15   # proven working — do not increase

_perf_tier = 0
_runtime_imgsz = ARGS.imgsz
_runtime_pose_scale = ARGS.pose_scale
_runtime_hand_skip = ARGS.hand_skip
_runtime_draw_hands = ARGS.draw_hands
_low_fps_streak = 0
_high_fps_streak = 0

_pose_thread  = threading.Thread(target=_infer_worker_pose, daemon=True)
_pose_thread.start()
_hand_thread  = threading.Thread(target=_infer_worker_hands, daemon=True)
_hand_thread.start()

_GESTURE_BEEP = {"AMBULANCE":(880,180),"POLICE":(660,220),
                 "FIRE":(1100,140),"DISTRESS":(440,300)}


def _set_perf_tier(tier: int):
    global _perf_tier, _runtime_imgsz, _runtime_pose_scale
    global _runtime_hand_skip, _runtime_draw_hands

    base_img = ARGS.imgsz
    base_scale = ARGS.pose_scale
    base_hand_skip = ARGS.hand_skip
    base_draw_hands = ARGS.draw_hands

    t = max(0, min(3, int(tier)))
    if t == 0:
        _runtime_imgsz = base_img
        _runtime_pose_scale = base_scale
        _runtime_hand_skip = base_hand_skip
        _runtime_draw_hands = base_draw_hands
    elif t == 1:
        _runtime_imgsz = max(352, base_img - 96)
        _runtime_pose_scale = max(0.58, base_scale - 0.08)
        _runtime_hand_skip = max(base_hand_skip + 1, 2)
        _runtime_draw_hands = base_draw_hands
    elif t == 2:
        _runtime_imgsz = 320
        _runtime_pose_scale = max(0.52, base_scale - 0.15)
        _runtime_hand_skip = max(base_hand_skip + 2, 3)
        _runtime_draw_hands = base_draw_hands
    else:
        _runtime_imgsz = 320
        _runtime_pose_scale = 0.50
        _runtime_hand_skip = max(base_hand_skip + 2, 3)
        _runtime_draw_hands = base_draw_hands

    _perf_tier = t


_set_perf_tier(0)


def _update_udp_send_ms(sample_ms: float):
    global _udp_send_ms
    _udp_send_ms = (_udp_send_ms * 0.85 + sample_ms * 0.15) if _udp_send_ms > 0 else sample_ms


def _poll_qnx_echo():
    global _qnx_rtt_ms, _qnx_rtt_ok
    while True:
        try:
            data, _addr = _udp_sock.recvfrom(4096)
        except BlockingIOError:
            break
        except OSError:
            break

        try:
            payload = json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            continue

        echo_id = payload.get("echo_id")
        if not echo_id:
            continue

        sent_t = _qnx_rtt_pending.pop(echo_id, None)
        if sent_t is None:
            continue

        rtt_sample = (time.perf_counter() - sent_t) * 1000.0
        _qnx_rtt_ms = (_qnx_rtt_ms * 0.8 + rtt_sample * 0.2) if _qnx_rtt_ok else rtt_sample
        _qnx_rtt_ok = True

def _buzzer_worker():
    while True:
        if active_alerts:
            gdef = next(iter(active_alerts.values()))
            freq, dur = _GESTURE_BEEP.get(gdef["key"], (750, 200))
            try:
                winsound.Beep(freq, dur); time.sleep(0.06)
                winsound.Beep(max(freq-120,200), dur); time.sleep(0.25)
            except Exception: time.sleep(0.5)
        else:
            time.sleep(0.05)

threading.Thread(target=_buzzer_worker, daemon=True).start()


# ═══════════════════════════════════════════════════════════════
#  UDP SEND HELPERS — QNX is the ONLY destination
# ═══════════════════════════════════════════════════════════════

def send_alert_to_qnx(pid: int, gdef: dict, now_t: float):
    """Send a gesture alert through QNX. QNX will dispatch to
       io_daemon and forward (including any cascade) to dashboard."""
    global _udp_seq
    _udp_seq += 1
    echo_id = f"{pid}:{gdef['key']}:{_udp_seq}"
    msg = json.dumps({
        "type":      "ALERT",
        "gesture":   gdef["key"],
        "alert":     gdef["alert"],
        "person":    f"P{pid}",
        "timestamp": now_t,
        "echo_id":   echo_id,
    }).encode()
    try:
        send_t0 = time.perf_counter()
        _udp_sock.sendto(msg, (QNX_HOST, QNX_PORT))
        _update_udp_send_ms((time.perf_counter() - send_t0) * 1000.0)
        _qnx_rtt_pending[echo_id] = time.perf_counter()
        _push_event(f"P{pid}", gdef["key"], "✓ QNX")
        print(f"[UDP → {QNX_HOST}:{QNX_PORT}] ALERT {gdef['key']} P{pid}")
    except Exception as e:
        _push_event(f"P{pid}", gdef["key"], "x QNX")
        print(f"[UDP ERROR] {e}")


def send_cancel_to_qnx(pid: int, gesture_key: str, now_t: float):
    """Send a cancel through QNX. QNX stops the buzzer immediately
       and broadcasts the cancel to the dashboard so the alert
       card clears on all operators' screens simultaneously."""
    global _udp_seq
    _udp_seq += 1
    echo_id = f"{pid}:{gesture_key}:CANCEL:{_udp_seq}"
    msg = json.dumps({
        "type":      "CANCEL",
        "gesture":   gesture_key,
        "alert":     f"CANCEL {gesture_key}",
        "clear":     True,
        "state":     "CLEAR",
        "person":    f"P{pid}",
        "timestamp": now_t,
        "echo_id":   echo_id,
        "reason":    "operator_fist_gesture",
    }).encode()
    try:
        send_t0 = time.perf_counter()
        _udp_sock.sendto(msg, (QNX_HOST, QNX_PORT))
        _update_udp_send_ms((time.perf_counter() - send_t0) * 1000.0)
        _qnx_rtt_pending[echo_id] = time.perf_counter()
        _push_event(f"P{pid}", f"CANCEL {gesture_key}", "✓ QNX")
        print(f"[UDP → {QNX_HOST}:{QNX_PORT}] CANCEL {gesture_key} P{pid} (fist gesture)")
    except Exception as e:
        _push_event(f"P{pid}", f"CANCEL {gesture_key}", "x QNX")
        print(f"[UDP CANCEL ERROR] {e}")

    try:
        _udp_sock.sendto(msg, (DASHBOARD_HOST, DASHBOARD_PORT))
        print(f"[DASH → {DASHBOARD_HOST}:{DASHBOARD_PORT}] CANCEL forwarded")
    except Exception as e:
        print(f"[DASH CANCEL ERROR] {e}")


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════
_camera_read_fail_streak = 0
while True:
    ret, frame = cap.read()
    if not ret:
        _camera_read_fail_streak += 1
        if _camera_read_fail_streak == 1:
            print("[WARN] Camera frame read failed; retrying...")
        if _camera_read_fail_streak >= 120:
            print("[ERROR] Camera stream lost; exiting.")
            break
        time.sleep(0.01)
        continue
    _camera_read_fail_streak = 0
    _poll_qnx_echo()

    h, w, _ = frame.shape
    _frame_serial += 1
    _frame_count += 1
    now = time.time()
    if now - _fps_time >= 1.0:
        _fps_display = _frame_count / (now - _fps_time)
        _frame_count = 0; _fps_time = now

        if _fps_display < TARGET_RENDER_FPS:
            _low_fps_streak += 1
            _high_fps_streak = 0
        elif _fps_display > TARGET_RENDER_FPS + 4.0:
            _high_fps_streak += 1
            _low_fps_streak = 0
        else:
            _low_fps_streak = 0
            _high_fps_streak = 0

        if _low_fps_streak >= 1 and _perf_tier < 3:
            _set_perf_tier(_perf_tier + 1)
            _low_fps_streak = 0
            print(f"[PERF] Low FPS -> tier {_perf_tier} (imgsz={_runtime_imgsz}, scale={_runtime_pose_scale:.2f}, hand_skip={_runtime_hand_skip}, draw_hands={_runtime_draw_hands})")
        elif _high_fps_streak >= 4 and _perf_tier > 0:
            _set_perf_tier(_perf_tier - 1)
            _high_fps_streak = 0
            print(f"[PERF] Recovering quality -> tier {_perf_tier} (imgsz={_runtime_imgsz}, scale={_runtime_pose_scale:.2f}, hand_skip={_runtime_hand_skip}, draw_hands={_runtime_draw_hands})")

    if _frame_serial % (SKIP_FRAMES + 1) == 0:
        with _pose_lock:
            if _pose_latest_frame[0] is None:
                _pose_latest_frame[0] = frame
    if _frame_serial % (_runtime_hand_skip + 1) == 0:
        with _hand_lock:
            if _hand_latest_frame[0] is None:
                _hand_latest_frame[0] = frame.copy()

    with _pose_lock:
        all_landmarks = _pose_result[0] if _pose_result[0] is not None else []
    with _hand_lock:
        hand_result = _hand_result[0]
    if hand_result and hand_result.hand_landmarks:
        hand_landmarks_current = hand_result.hand_landmarks
        hand_gestures_current = [
            (g[0].category_name, g[0].score) if g else ("", 0.0)
            for g in (hand_result.gestures or [])
        ]
        _hand_visual_cache = (hand_landmarks_current, hand_gestures_current)
        _hand_visual_last_t = time.time()
    elif time.time() - _hand_visual_last_t <= HAND_VISUAL_TTL:
        hand_landmarks_current, hand_gestures_current = _hand_visual_cache
    else:
        hand_landmarks_current = []
        hand_gestures_current = []

    detections = []
    for det_idx, person_lm in enumerate(all_landmarks):
        ls = person_lm["kpts"][IDX["LS"]]; rs = person_lm["kpts"][IDX["RS"]]
        cx = int((ls[0] + rs[0]) / 2)
        cy = int((ls[1] + rs[1]) / 2)
        detections.append({
            "centroid": (cx, cy),
            "bbox": person_lm["bbox"],
            "det_idx": det_idx,
        })

    id_map = tracker.update(detections)
    for pid in list(gesture_timers.keys()):
        if pid not in id_map: del gesture_timers[pid]
    for pid in list(_kpt_smooth.keys()):
        if pid not in id_map:
            del _kpt_smooth[pid]
    hand_map = assign_hands_to_people(hand_landmarks_current, id_map, w, h)

    triggered = {}

    for person_idx, (pid, det) in enumerate(id_map.items()):
        centroid = det["centroid"]; bbox = det["bbox"]
        det_idx  = det.get("det_idx", person_idx)
        if det_idx >= len(all_landmarks):
            continue
        lm       = all_landmarks[det_idx]
        lm["kpts"] = smooth_keypoints(pid, lm["kpts"], lm["kconf"])
        required = [IDX["LW"],IDX["RW"],IDX["LE"],IDX["RE"],
                    IDX["LS"],IDX["RS"],IDX["NOSE"],IDX["L_EAR"],IDX["R_EAR"]]

        box_color = pid_color(pid)
        cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), box_color, 2)
        tag_label = f"P{pid}"; tag_y = max(bbox[1]-8, 16)
        cv2.rectangle(frame, (bbox[0],tag_y-16),
                      (bbox[0]+len(tag_label)*13, tag_y+4), box_color, -1)
        cv2.putText(frame, tag_label, (bbox[0]+3, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        skel_line = tuple(int(c * 0.9) for c in box_color)
        skel_point = tuple(int(min(c + 35, 255)) for c in box_color)
        draw_skeleton(frame, lm, skel_line, skel_point)

        if not all(lm["kconf"][k] >= MIN_VISIBILITY for k in required):
            continue

        pts = {
            "lw":   (int(lm["kpts"][IDX["LW"]][0]), int(lm["kpts"][IDX["LW"]][1])),
            "rw":   (int(lm["kpts"][IDX["RW"]][0]), int(lm["kpts"][IDX["RW"]][1])),
            "le":   (int(lm["kpts"][IDX["LE"]][0]), int(lm["kpts"][IDX["LE"]][1])),
            "re":   (int(lm["kpts"][IDX["RE"]][0]), int(lm["kpts"][IDX["RE"]][1])),
            "ls":   (int(lm["kpts"][IDX["LS"]][0]), int(lm["kpts"][IDX["LS"]][1])),
            "rs":   (int(lm["kpts"][IDX["RS"]][0]), int(lm["kpts"][IDX["RS"]][1])),
            "nose": (int(lm["kpts"][IDX["NOSE"]][0]), int(lm["kpts"][IDX["NOSE"]][1])),
            "lear": (int(lm["kpts"][IDX["L_EAR"]][0]), int(lm["kpts"][IDX["L_EAR"]][1])),
            "rear": (int(lm["kpts"][IDX["R_EAR"]][0]), int(lm["kpts"][IDX["R_EAR"]][1])),
        }
        sw = max(dist(pts["ls"], pts["rs"]), 1)

        if pid not in gesture_timers: gesture_timers[pid] = {}
        matched_gkey = None
        t_loop = time.time()

        for bar_slot, gdef in enumerate(GESTURES):
            gkey = gdef["key"]
            detected = gdef["detect"](pts, sw)
            state = gesture_timers[pid].get(gkey)

            if detected and matched_gkey is None:
                matched_gkey = gkey
                if state is None:
                    state = {"start": t_loop, "last_seen": t_loop}
                else:
                    state["last_seen"] = t_loop
                gesture_timers[pid][gkey] = state

                elapsed = t_loop - state["start"]
                draw_hold_bar(frame, centroid[0], bbox[1]+30+bar_slot*22,
                              elapsed, ALERT_TIME, pid, gdef["label"], gdef["color"])
                _last_gesture_seen[pid] = t_loop
                if elapsed >= ALERT_TIME:
                    triggered[pid] = gdef
                    cv2.rectangle(frame, (bbox[0],bbox[1]),
                                  (bbox[2],bbox[3]), gdef["color"], 3)
            else:
                if state is not None and (t_loop - state["last_seen"]) > GESTURE_GRACE:
                    gesture_timers[pid].pop(gkey, None)

        for point, col in [(pts["lw"],(0,255,255)),(pts["rw"],(0,255,255)),
                           (pts["le"],(255,165,0)),(pts["re"],(255,165,0))]:
            cv2.circle(frame, point, 10, col, -1)
            cv2.circle(frame, point, 12, (255,255,255), 2)

        panel_w = 300; panel_x = w-panel_w-10
        row_h   = 96;  panel_top = 10 + person_idx*(row_h+8)
        fill_alpha(frame, panel_x-6, panel_top, panel_x+panel_w,
                   panel_top+row_h, (10,10,30), 0.65)
        cv2.rectangle(frame, (panel_x-6,panel_top),
                      (panel_x+panel_w,panel_top+row_h), box_color, 1)
        cv2.putText(frame, f" PERSON  P{pid} ", (panel_x, panel_top+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)
        for gi, gdef in enumerate(GESTURES):
            active    = gdef["key"] in gesture_timers.get(pid, {})
            marker    = "\u25cf" if active else "\u25cb"
            cv2.putText(frame,
                        f"{marker} {gdef['label']}: {'HOLD' if active else '---'}",
                        (panel_x+6, panel_top+38+gi*16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        gdef["color"] if active else (90,90,90), 1, cv2.LINE_AA)

    for pid, gdef in triggered.items():
        active_alerts[pid] = gdef
        active_alert_times.setdefault(pid, time.time())

    # ── CANCEL gesture (fist below eye for 2s) ────────────────────────────────
    if active_alerts:
        t_now = time.time()
        canceling_pid = None
        cancel_progress = 0.0
        for pid in list(active_alerts.keys()):
            if pid not in id_map:
                if t_now - _cancel_fist_last.get(pid, 0) > CANCEL_GRACE:
                    cancel_timers.pop(pid, None)
                continue

            if t_now - _last_gesture_seen.get(pid, 0) < CANCEL_DEBOUNCE_AFTER_GESTURE:
                cancel_timers.pop(pid, None)
                continue

            if t_now - active_alert_times.get(pid, t_now) < 0.5:
                cancel_timers.pop(pid, None)
                continue

            det = id_map[pid]
            det_idx = det.get("det_idx")
            if det_idx is None or det_idx >= len(all_landmarks):
                cancel_timers.pop(pid, None)
                continue

            lm = all_landmarks[det_idx]
            if lm["kconf"][1] > 0.15 and lm["kconf"][2] > 0.15:
                eye_y = (lm["kpts"][1][1] + lm["kpts"][2][1]) / 2.0
            else:
                eye_y = lm["kpts"][IDX["NOSE"]][1]
            fists_below_eye = 0
            for hand_idx, hlm in hand_map.get(pid, []):
                gesture_name, gesture_score = (
                    hand_gestures_current[hand_idx]
                    if hand_idx < len(hand_gestures_current) else ("", 0.0)
                )
                if is_fist_from_gesture(gesture_name, gesture_score):
                    if (hlm[0].y * h) > eye_y:
                        fists_below_eye += 1

            if fists_below_eye >= CANCEL_REQUIRED_FISTS:
                _cancel_fist_last[pid] = t_now
                cancel_timers.setdefault(pid, t_now)
                c_elapsed = t_now - cancel_timers[pid]
                cancel_ratio = min(c_elapsed / CANCEL_TIME, 1.0)
                if cancel_ratio > cancel_progress:
                    cancel_progress = cancel_ratio
                    canceling_pid = pid

                bbox = det["bbox"]
                arc_cx = (bbox[0] + bbox[2]) // 2
                arc_cy = max(70, bbox[1] - 24)
                fill_alpha(frame, arc_cx-58, arc_cy-58, arc_cx+58, arc_cy+58, (0,0,0), 0.72)
                cv2.ellipse(frame, (arc_cx, arc_cy), (48, 48), -90, 0, 360, (40, 40, 40), 3)
                cv2.ellipse(frame, (arc_cx, arc_cy), (48, 48), -90, 0,
                            int(360 * min(c_elapsed / CANCEL_TIME, 1.0)), (0, 255, 100), 7)
                spin_angle = int((t_now * 420.0) % 360)
                cv2.ellipse(frame, (arc_cx, arc_cy), (58, 58), 0,
                            spin_angle, spin_angle + 110, (255, 255, 255), 2)
                cv2.putText(frame, f"{int(c_elapsed / CANCEL_TIME * 100)}%",
                            (arc_cx-22, arc_cy+8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (0,255,100), 2, cv2.LINE_AA)
                cv2.putText(frame, f"P{pid} DISMISS", (arc_cx-50, arc_cy+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,80), 1, cv2.LINE_AA)

                if c_elapsed >= CANCEL_TIME:
                    # ─────────────────────────────────────────────────────
                    # FIST HELD 2 SECONDS → send CANCEL through QNX backbone
                    # ─────────────────────────────────────────────────────
                    gdef_cancelled = active_alerts.get(pid)
                    if gdef_cancelled is not None:
                        send_cancel_to_qnx(pid, gdef_cancelled["key"], t_now)
                    _push_event(f"P{pid}", "CANCEL", "-")
                    active_alerts.pop(pid, None)
                    active_alert_times.pop(pid, None)
                    cancel_timers.pop(pid, None)
                    _cancel_fist_last.pop(pid, None)
                    gesture_timers.pop(pid, None)
            else:
                if t_now - _cancel_fist_last.get(pid, 0) > CANCEL_GRACE:
                    cancel_timers.pop(pid, None)

        if canceling_pid is not None:
            bx1, by1, bx2, by2 = w//2 - 210, h - 116, w//2 + 210, h - 72
            fill_alpha(frame, bx1, by1, bx2, by2, (0, 0, 0), 0.70)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 80, 80), 1)
            cv2.putText(frame, f"CANCEL BUFFERING  P{canceling_pid}",
                        (bx1 + 14, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (230, 230, 230), 1, cv2.LINE_AA)
            bar_x1, bar_y1 = bx1 + 14, by1 + 28
            bar_w, bar_h = (bx2 - bx1) - 28, 14
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (55, 55, 55), -1)
            cv2.rectangle(frame, (bar_x1, bar_y1),
                          (bar_x1 + int(bar_w * cancel_progress), bar_y1 + bar_h), (0, 210, 90), -1)
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (220, 220, 220), 1)
            cv2.putText(frame, f"{int(cancel_progress * 100)}%",
                        (bx2 - 64, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 255, 120), 1, cv2.LINE_AA)
    else:
        cancel_timers.clear()
        _cancel_fist_last.clear()

    if _runtime_draw_hands:
        for hand_idx, hlm in enumerate(hand_landmarks_current):
            gname, gscore = (
                hand_gestures_current[hand_idx]
                if hand_idx < len(hand_gestures_current) else ("", 0.0)
            )
            is_closed = is_fist_from_gesture(gname, gscore)
            _draw_hand(frame, hlm, w, h, is_closed)
            wx, wy = int(hlm[0].x * w), int(hlm[0].y * h)
            cv2.putText(frame, f"{gname} {gscore:.2f}", (wx + 10, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    with _stats_lock:
        pose_fps_now = _pose_fps_display
        hand_fps_now = _hand_fps_display
        pose_ms_now = _pose_ms_display
        hand_ms_now = _hand_ms_display

    fill_alpha(frame, 0, 0, 760, 58, (0,0,0), 0.62)
    cv2.putText(frame,
                f"Render FPS: {_fps_display:.1f}   Pose FPS: {pose_fps_now:.1f} ({pose_ms_now:.1f}ms)   "
                f"Hand FPS: {hand_fps_now:.1f} ({hand_ms_now:.1f}ms)   People: {len(all_landmarks)}",
                (12,34), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame,
                f"Perf Tier: {_perf_tier}  imgsz={_runtime_imgsz}  scale={_runtime_pose_scale:.2f}  hand_skip={_runtime_hand_skip}  "
                f"UDP send: {_udp_send_ms:.2f}ms  QNX RTT: {(_qnx_rtt_ms if _qnx_rtt_ok else 0.0):.2f}ms{'*' if _qnx_rtt_ok else ' (waiting)'}",
                (12,54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1, cv2.LINE_AA)
    draw_event_ticker(frame)

    legends  = ["Cross arms","Hands up (spread)","T-Pose","Hands behind head"]
    leg_rows = len(GESTURES) + 1
    fill_alpha(frame, 0, h-(leg_rows*26+22), 390, h, (0,0,0), 0.55)
    cv2.putText(frame, "  CANCEL       One closed fist below eye level  2s",
                (8, h-8-len(GESTURES)*26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,100), 1, cv2.LINE_AA)
    for gi, gdef in enumerate(GESTURES):
        cv2.putText(frame, f"  {gdef['label']:<12}  {legends[gi]}",
                    (8, h-8-gi*26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, gdef["color"], 1, cv2.LINE_AA)

    # ── UDP → QNX (alerts only — cancel handled above via fist gesture) ──────
    now_t = time.time()
    for pid, gdef in triggered.items():
        gkey = gdef["key"]
        if now_t - _last_alert_sent.get(pid, {}).get(gkey, 0) >= ALERT_COOLDOWN:
            send_alert_to_qnx(pid, gdef, now_t)
            _last_alert_sent.setdefault(pid, {})[gkey] = now_t

    # ── Alert overlay ─────────────────────────────────────────────────────────
    if active_alerts:
        first_gdef  = next(iter(active_alerts.values()))
        alert_color = first_gdef["color"]
        t_now       = time.time()
        pulse       = 0.5 + 0.5 * math.sin(t_now * 6.0)
        blink       = int(t_now * 2) % 2 == 0
        border_th   = int(6 + pulse * 12)

        cv2.rectangle(frame, (0,0), (w,h), alert_color, border_th)
        cv2.rectangle(frame, (border_th,border_th),
                      (w-border_th,h-border_th), (255,255,255), 1)

        banner_h = 120
        bc = tuple(int(c * 0.28) for c in alert_color)
        fill_alpha(frame, 0, 0, w, banner_h, bc, 0.90)
        cv2.line(frame, (0,banner_h), (w,banner_h), alert_color, 3)
        cv2.line(frame, (0,banner_h+3), (w,banner_h+3), (255,255,255), 1)

        if blink:
            cv2.circle(frame, (22,22), 9, (0,255,80), -1)
            cv2.putText(frame, "LIVE", (38,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,80), 2, cv2.LINE_AA)

        oldest_t  = min(active_alert_times.get(p, t_now) for p in active_alerts)
        badge_str = f"{int(t_now - oldest_t)}s"
        (bw2,_),_ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        fill_alpha(frame, w-bw2-24, 6, w-6, 36, (0,0,0), 0.55)
        cv2.putText(frame, badge_str, (w-bw2-14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, alert_color, 2, cv2.LINE_AA)

        alert_text = first_gdef["alert"]
        font, fscale = cv2.FONT_HERSHEY_DUPLEX, 1.85
        (atw,_),_ = cv2.getTextSize(alert_text, font, fscale, 3)
        ax, ay = (w-atw)//2, 82
        cv2.putText(frame, alert_text, (ax+4,ay+4), font, fscale, (0,0,0), 7, cv2.LINE_AA)
        cv2.putText(frame, alert_text, (ax,ay),     font, fscale, (255,255,255), 5, cv2.LINE_AA)
        cv2.putText(frame, alert_text, (ax,ay),     font, fscale, alert_color, 2, cv2.LINE_AA)

        pids_str = "  ●  ".join(f"P{p}: {g['label']}" for p,g in active_alerts.items())
        (ptw,_),_ = cv2.getTextSize(pids_str, cv2.FONT_HERSHEY_SIMPLEX, 0.88, 2)
        fill_alpha(frame,(w-ptw)//2-16,90,(w+ptw)//2+16,116,(0,0,0),0.52)
        cv2.putText(frame, pids_str, ((w-ptw)//2,112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255,255,255), 2, cv2.LINE_AA)

        hint = "[ HOLD ONE CLOSED FIST BELOW EYE LEVEL FOR 2s TO DISMISS ]"
        hval = int(160 + 95 * pulse)
        (hw2,_),_ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        fill_alpha(frame,(w-hw2)//2-14,h-42,(w+hw2)//2+14,h-8,(0,0,0),0.70)
        cv2.putText(frame, hint, ((w-hw2)//2, h-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (hval,hval,hval), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()