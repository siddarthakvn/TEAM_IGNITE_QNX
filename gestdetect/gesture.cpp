import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.framework.formats import landmark_pb2
import time
import math
import urllib.request
import os

# ─────────────────────────────────────────────
#  Auto-download model (runs once)
#  Using LITE model — optimised for Pi 4 / edge CPU
# ─────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
if not os.path.exists(MODEL_PATH):
    print("Downloading pose landmarker model (one-time, ~30 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ─────────────────────────────────────────────
#  MediaPipe Tasks — multi-person PoseLandmarker
# ─────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=4,                          # 4 people max — saves CPU on Pi
    min_pose_detection_confidence=0.4,    # slightly lower for CCTV distance
    min_pose_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

mp_draw      = mp.solutions.drawing_utils
mp_pose_conn = mp.solutions.pose.POSE_CONNECTIONS

# Works on both Windows (USB cam) and Pi (V4L2 /dev/video0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ─────────────────────────────────────────────
#  Tunable constants
# ─────────────────────────────────────────────
ALERT_TIME       = 3.0   # seconds the pose must be held
WRIST_DIST_RATIO = 0.80  # arms-crossed: max wrist-to-wrist / shoulder_width
ELBOW_DIST_RATIO = 1.20  # arms-crossed: max wrist-to-elbow / shoulder_width
MIN_VISIBILITY   = 0.20  # lower threshold — CCTV cameras are far
TRACK_MAX_DIST   = 150   # pixels — tighter to avoid ID swaps

# ── Pi performance settings ───────────────────
INFER_W      = 320
INFER_H      = 240
# Pi 4: set to 2 (infer every 3rd frame, ~15 FPS effective)
# Windows/fast machine: set to 1 (infer every 2nd frame, ~20+ FPS)
SKIP_FRAMES  = 2

# Landmark indices
IDX = {
    "LW": 15, "RW": 16,
    "LE": 13, "RE": 14,
    "LS": 11, "RS": 12,
    "NOSE": 0,
    "L_EAR": 7, "R_EAR": 8,
}

# ─────────────────────────────────────────────
#  Gesture definitions
#  Priority order matters — first match wins (mutual exclusion)
#  detect(pts, sw) -> bool
# ─────────────────────────────────────────────
def _detect_ambulance(pts, sw):
    """Arms crossed tight on chest.
    Wrists cross body-centre + elbows bent + wrists close together.
    """
    lw, rw, le, re, ls, rs = (
        pts["lw"], pts["rw"], pts["le"],
        pts["re"], pts["ls"], pts["rs"]
    )
    lw_cross = dist(lw, rs) < dist(lw, ls)   # lw near opposite shoulder
    rw_cross = dist(rw, ls) < dist(rw, rs)
    wd       = dist(lw, rw) / sw              # wrists close
    el       = dist(lw, le) / sw              # elbows bent
    er       = dist(rw, re) / sw
    # FIX: use named constants instead of hardcoded literals
    return lw_cross and rw_cross and wd < WRIST_DIST_RATIO and \
           el < ELBOW_DIST_RATIO and er < ELBOW_DIST_RATIO


def _detect_police(pts, sw):
    """Surrender / call for help — both wrists raised ABOVE the ears,
    AND wrists spread wide apart (not touching head).
    Ears are a higher and more stable anchor than nose.
    """
    lw, rw       = pts["lw"],  pts["rw"]
    l_ear, r_ear = pts["lear"], pts["rear"]
    # Use the higher of the two ears as the height threshold
    ear_y        = min(l_ear[1], r_ear[1])   # lower pixel value = higher in frame
    arms_up      = lw[1] < ear_y and rw[1] < ear_y
    # Wrists must be spread apart — rules out hands clasped over head
    spread       = dist(lw, rw) / sw > 0.80
    return arms_up and spread


def _detect_fire(pts, sw):
    """T-Pose — arms fully extended sideways at shoulder height.
    Uses pure DISTANCE (camera-mirror agnostic):
      - Each wrist far from its own shoulder (arm extended)
      - Total wrist spread > 2.2× shoulder width
      - Both wrists near shoulder height vertically
      - Wrists NOT above nose (rules out police overlap)
    """
    lw, rw, ls, rs = pts["lw"], pts["rw"], pts["ls"], pts["rs"]
    nose           = pts["nose"]
    # Arm extension: wrist-to-shoulder distance relative to shoulder width
    lw_ext     = dist(lw, ls) / sw > 0.90
    rw_ext     = dist(rw, rs) / sw > 0.90
    # Total spread between wrists
    wrist_span = dist(lw, rw) / sw > 2.20
    # Wrists near shoulder height (small vertical offset)
    lw_level   = abs(lw[1] - ls[1]) < 0.55 * sw
    rw_level   = abs(rw[1] - rs[1]) < 0.55 * sw
    # Must NOT be above nose — avoids police gesture overlap
    not_above  = lw[1] > nose[1] and rw[1] > nose[1]
    return lw_ext and rw_ext and wrist_span and lw_level and rw_level and not_above


def _detect_distress(pts, sw):
    """Hands behind head — prisoner / surrender distress pose.

    Each wrist is near its OWN ear with elbows flared wide.

    Why this can't overlap with other gestures:
      AMBULANCE : wrists cross the body at chest  → fails wrist-near-ear
      POLICE    : wrists must be ABOVE the ears   → blocked by not_above_ears
      FIRE      : wrists at shoulder height far   → fails wrist-near-ear
    """
    lw, rw         = pts["lw"],  pts["rw"]
    le, re         = pts["le"],  pts["re"]
    l_ear, r_ear   = pts["lear"], pts["rear"]

    # Each wrist close to its OWN ear (same-side)
    lw_near_ear = dist(lw, l_ear) / sw < 0.55
    rw_near_ear = dist(rw, r_ear) / sw < 0.55

    # Wrists must NOT be above ears — hard block vs POLICE gesture.
    # FIX: use max(dynamic, fixed_floor) so the tolerance doesn't collapse
    # to a few pixels when shoulder_width is small (CCTV far distance).
    ear_avg_y   = (l_ear[1] + r_ear[1]) / 2
    tolerance   = max(0.20 * sw, 10)          # at least 10 px buffer
    not_above_ears = lw[1] >= ear_avg_y - tolerance and \
                     rw[1] >= ear_avg_y - tolerance

    # Elbows flared wide — confirms arms are out, not tucked
    elbows_wide = dist(le, re) / sw > 1.20

    return lw_near_ear and rw_near_ear and not_above_ears and elbows_wide


GESTURES = [
    {
        "key":    "AMBULANCE",
        "label":  "AMBULANCE",
        "color":  (0,   0,   255),   # red
        "alert":  "!! MEDICAL EMERGENCY !!",
        "detect": _detect_ambulance,
    },
    {
        "key":    "POLICE",
        "label":  "POLICE",
        "color":  (255, 80,  0),     # blue
        "alert":  "!! POLICE / SECURITY !!",
        "detect": _detect_police,
    },
    {
        "key":    "FIRE",
        "label":  "FIRE",
        "color":  (0,   140, 255),   # orange
        "alert":  "!! FIRE / DISASTER !!",
        "detect": _detect_fire,
    },
    {
        "key":    "DISTRESS",
        "label":  "DISTRESS",
        "color":  (255, 0,   180),   # magenta
        "alert":  "!! DISTRESS / TRAPPED !!",
        "detect": _detect_distress,
    },
]

# ─────────────────────────────────────────────
#  SORT-inspired tracker  (IoU bbox + ghost buffer)
#
#  Each track stores:
#    centroid, bbox, frames_since_seen
#  IDs survive up to GHOST_FRAMES frames of no detection
#  before being permanently deleted — prevents ID churn
#  when MediaPipe drops a landmark for 1–2 frames.
# ─────────────────────────────────────────────
GHOST_FRAMES = 15   # frames to keep an ID alive after last detection

class PersonTracker:
    def __init__(self, max_dist: int = TRACK_MAX_DIST):
        self.next_id = 1
        # id -> {"centroid": (x,y), "bbox": (x1,y1,x2,y2), "missing": int}
        self.tracks  = {}
        self.max_dist = max_dist

    def _iou(self, a, b) -> float:
        """Intersection-over-Union of two (x1,y1,x2,y2) boxes."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter)

    def update(self, detections: list[dict]) -> dict:
        """
        detections: list of {"centroid":(x,y), "bbox":(x1,y1,x2,y2)}
        Returns {person_id: detection_dict} for currently visible people.
        """
        # Age all existing tracks
        for t in self.tracks.values():
            t["missing"] += 1

        # Remove tracks gone too long
        self.tracks = {pid: t for pid, t in self.tracks.items()
                       if t["missing"] <= GHOST_FRAMES}

        if not detections:
            return {}

        if not self.tracks:
            result = {}
            for det in detections:
                pid = self.next_id
                self.tracks[pid] = {**det, "missing": 0}
                result[pid] = det
                self.next_id += 1
            return result

        used_pids = set()
        result    = {}

        for det in detections:
            best_pid   = None
            best_score = -1.0

            for pid, track in self.tracks.items():
                if pid in used_pids:
                    continue
                # Prefer IoU match; fall back to centroid distance
                iou = self._iou(det["bbox"], track["bbox"])
                cx, cy = det["centroid"]
                tx, ty = track["centroid"]
                cdist  = math.hypot(cx - tx, cy - ty)
                # Combined score: IoU weighted heavily, distance as tiebreak
                score = iou * 10 - (cdist / self.max_dist)
                if score > best_score and cdist < self.max_dist:
                    best_score = score
                    best_pid   = pid

            if best_pid is None:
                best_pid = self.next_id
                self.next_id += 1

            self.tracks[best_pid] = {**det, "missing": 0}
            result[best_pid] = det
            used_pids.add(best_pid)

        return result


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def dist(p1: tuple, p2: tuple) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def pt(lm, w: int, h: int) -> tuple:
    return (int(lm.x * w), int(lm.y * h))


def to_proto(landmarks) -> landmark_pb2.NormalizedLandmarkList:
    """Convert Tasks API landmark list to proto for mp_draw."""
    proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        p = proto.landmark.add()
        p.x, p.y, p.z = lm.x, lm.y, lm.z
        p.visibility = lm.visibility if lm.visibility is not None else 0.0
    return proto


def draw_hold_bar(frame, cx: int, top_y: int, elapsed: float, total: float,
                  pid: int, label: str, color: tuple):
    """Draw a per-person gesture hold bar with gesture-type colour."""
    bar_w, bar_h = 200, 14
    bar_x = max(0, cx - bar_w // 2)
    bar_y = max(20, top_y)
    ratio  = min(elapsed / total, 1.0)
    fill   = color if ratio < 1.0 else (0, 255, 0)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * ratio), bar_y + bar_h), fill, -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    cv2.putText(frame, f"P{pid} {label} {elapsed:.1f}s/{total:.0f}s",
                (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, color, 1)


def get_bbox(person_lm, fw: int, fh: int, pad: float = 0.12) -> tuple:
    """
    Derive a bounding box from all visible pose landmarks.
    pad: fractional padding added around the tight landmark box.
    Returns (x1, y1, x2, y2) in pixel coords, clamped to frame.
    """
    xs = [lm.x for lm in person_lm if lm.visibility > 0.1]
    ys = [lm.y for lm in person_lm if lm.visibility > 0.1]
    if not xs:
        return (0, 0, fw, fh)
    rx1, ry1 = min(xs), min(ys)
    rx2, ry2 = max(xs), max(ys)
    bw = rx2 - rx1;  bh = ry2 - ry1
    x1 = int(max(0,  (rx1 - bw * pad) * fw))
    y1 = int(max(0,  (ry1 - bh * pad) * fh))
    x2 = int(min(fw, (rx2 + bw * pad) * fw))
    y2 = int(min(fh, (ry2 + bh * pad) * fh))
    return (x1, y1, x2, y2)


# ─────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────
tracker       = PersonTracker()
# {person_id: {gesture_key: start_time}}  — independent timer per person per gesture
gesture_timers: dict = {}

# FIX: use two separate counters so that the FPS reset every second does not
# accidentally set _frame_count to 0 and trigger an unscheduled inference on
# the very next line (0 % (SKIP_FRAMES+1) == 0 is always True after reset).
_frame_count  = 0   # counts frames within the current 1-second FPS window; resets
_infer_count  = 0   # ever-increasing; drives the skip-frame cadence; never resets
_fps_time     = time.time()
_fps_display  = 0.0
_last_result  = None

# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    _frame_count += 1
    _infer_count += 1

    # ── FPS counter ───────────────────────────────────────────────────
    now = time.time()
    if now - _fps_time >= 1.0:
        _fps_display = _frame_count / (now - _fps_time)
        _frame_count = 0
        _fps_time    = now

    # ── Frame skipping — uses _infer_count so the FPS reset above does
    #    not corrupt the skip cadence.
    if (_infer_count % (SKIP_FRAMES + 1)) == 0 or _last_result is None:
        small = cv2.resize(frame, (INFER_W, INFER_H))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        _last_result = landmarker.detect(mp_image)

    result = _last_result
    all_landmarks = result.pose_landmarks   # list[list[NormalizedLandmark]]

    # ── Build detection list (centroid + bbox) for tracker ────────────────
    detections = []
    for person_lm in all_landmarks:
        ls = person_lm[IDX["LS"]]
        rs = person_lm[IDX["RS"]]
        cx = int((ls.x + rs.x) / 2 * w)
        cy = int((ls.y + rs.y) / 2 * h)
        bbox = get_bbox(person_lm, w, h)
        detections.append({"centroid": (cx, cy), "bbox": bbox})

    id_map = tracker.update(detections)

    # ── Retire timers for persons gone past ghost window ──────────────
    # FIX: use tracker.tracks.keys() (all alive PIDs, including ghosts)
    # instead of id_map.keys() (only currently visible PIDs).  Previously,
    # a person unseen for even 1 frame had their gesture timer wiped, forcing
    # the 3-second hold to restart — defeating GHOST_FRAMES entirely.
    alive_ids = set(tracker.tracks.keys())
    for pid in list(gesture_timers.keys()):
        if pid not in alive_ids:
            del gesture_timers[pid]

    # ── Per-person processing ─────────────────────────────────────────
    # {person_id: gesture_dict}  filled when a gesture completes 3-s hold
    triggered: dict = {}

    for person_idx, (pid, det) in enumerate(id_map.items()):
        centroid = det["centroid"]
        bbox     = det["bbox"]
        lm       = all_landmarks[person_idx]

        required = [IDX["LW"], IDX["RW"], IDX["LE"],
                    IDX["RE"], IDX["LS"], IDX["RS"], IDX["NOSE"],
                    IDX["L_EAR"], IDX["R_EAR"]]

        # ── Default green box + label ─────────────────────────────────
        box_color = (0, 255, 0)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      box_color, 2)
        tag_label = f"P{pid}"
        tag_y     = max(bbox[1] - 8, 16)
        cv2.rectangle(frame, (bbox[0], tag_y - 16),
                      (bbox[0] + len(tag_label) * 13, tag_y + 4),
                      box_color, -1)
        cv2.putText(frame, tag_label, (bbox[0] + 3, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        mp_draw.draw_landmarks(frame, to_proto(lm), mp_pose_conn)

        if not all(lm[k].visibility >= MIN_VISIBILITY for k in required):
            continue

        # ── Build pixel-coord point dict ──────────────────────────────
        pts = {
            "lw":   pt(lm[IDX["LW"]],    w, h),
            "rw":   pt(lm[IDX["RW"]],    w, h),
            "le":   pt(lm[IDX["LE"]],    w, h),
            "re":   pt(lm[IDX["RE"]],    w, h),
            "ls":   pt(lm[IDX["LS"]],    w, h),
            "rs":   pt(lm[IDX["RS"]],    w, h),
            "nose": pt(lm[IDX["NOSE"]],  w, h),
            "lear": pt(lm[IDX["L_EAR"]], w, h),
            "rear": pt(lm[IDX["R_EAR"]], w, h),
        }
        sw = max(dist(pts["ls"], pts["rs"]), 1)

        if pid not in gesture_timers:
            gesture_timers[pid] = {}

        # ── Mutual exclusion: only the FIRST matching gesture fires ───
        # Gestures are checked in priority order (AMBULANCE first).
        # Once a gesture is detected for this person, all lower-priority
        # gestures have their timers cleared so they cannot overlap.
        matched_gkey = None

        # FIX: use gi from enumerate instead of GESTURES.index(gdef)
        # which is O(n) and redundant inside an already-enumerated loop.
        for gi, gdef in enumerate(GESTURES):
            gkey     = gdef["key"]
            gcolor   = gdef["color"]
            glabel   = gdef["label"]
            detected = gdef["detect"](pts, sw)

            if detected and matched_gkey is None:
                # This gesture wins for this frame
                matched_gkey = gkey
                if gkey not in gesture_timers[pid]:
                    gesture_timers[pid][gkey] = time.time()
                elapsed  = time.time() - gesture_timers[pid][gkey]
                bar_top  = bbox[1] + 30 + gi * 22
                draw_hold_bar(frame, centroid[0], bar_top,
                              elapsed, ALERT_TIME, pid, glabel, gcolor)

                if elapsed >= ALERT_TIME:
                    triggered[pid] = gdef
                    cv2.rectangle(frame,
                                  (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  gcolor, 3)
                    cv2.rectangle(frame, (bbox[0], tag_y - 16),
                                  (bbox[0] + len(tag_label) * 13, tag_y + 4),
                                  gcolor, -1)
                    cv2.putText(frame, tag_label, (bbox[0] + 3, tag_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                # Clear timer for any non-winning gesture
                gesture_timers[pid].pop(gkey, None)

        # ── Debug dots ────────────────────────────────────────────────
        for point, col in [
            (pts["lw"], (0, 255, 255)), (pts["rw"], (0, 255, 255)),
            (pts["le"], (255, 165, 0)), (pts["re"], (255, 165, 0)),
        ]:
            cv2.circle(frame, point, 7, col, -1)

        # ── Per-person debug panel (right side) ───────────────────────
        ox = w - 320
        oy = 30 + person_idx * 110
        cv2.putText(frame, f"-- P{pid} --", (ox, oy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
        for gi, gdef in enumerate(GESTURES):
            active    = gdef["key"] in gesture_timers.get(pid, {})
            label_str = f"{gdef['label']}: {'HOLD' if active else '---'}"
            cv2.putText(frame, label_str, (ox, oy + 18 + gi * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        gdef["color"] if active else (100, 100, 100), 1)

    # ── People count + FPS ────────────────────────────────────────────
    cv2.putText(frame, f"People: {len(all_landmarks)}  FPS: {_fps_display:.1f}",
                (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Gesture legend (bottom-left) ──────────────────────────────────
    legends = ["Cross arms", "Hands up (spread)", "T-Pose", "Hands behind head"]
    for gi, gdef in enumerate(GESTURES):
        cv2.putText(frame,
                    f"{gdef['label']}: {legends[gi]}",
                    (10, h - 20 - gi * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, gdef["color"], 1)

    # ── Emergency overlay ─────────────────────────────────────────────
    if triggered:
        first_gdef  = next(iter(triggered.values()))
        alert_color = first_gdef["color"]
        cv2.rectangle(frame, (0, 0), (w, h), alert_color, 6)
        cv2.putText(frame, first_gdef["alert"],
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.1, alert_color, 3)
        pids_str = ", ".join(
            f"P{pid} ({gdef['label']})" for pid, gdef in triggered.items()
        )
        cv2.putText(frame, pids_str,
                    (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)

    cv2.imshow("Emergency Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
