"""
Two ways to use this file:

1) TEST IT BY ITSELF with webcam:
       python posture_check.py
   Stand facing the camera with your whole body in view. Watch the numbers.
   Press Q to quit.  (Optional: P records your own posture as the reference.)

2) USE IT INSIDE THE HEIGHT TRACKER (see the bottom of this file).

How it works
------------
MediaPipe Pose gives 33 body landmarks, in the image (2D) and as rough
3D "world" coordinates in meters (origin = middle of the hips).
Five numbers are measured:

  shoulder_tilt  - shoulders not level (2D, degrees)
  head_tilt      - head rolled to one side (2D, degrees)
  torso_lean     - shoulders leaning toward/away from the camera vs hips (3D, degrees)
  head_forward   - head pushed toward/away from the camera vs shoulders (3D, cm)
  knee_bend      - knees not straight (3D, degrees, 180 = straight)

Slouching = torso_lean and/or head_forward too large.

IMPORTANT LIMIT: from a FRONT camera the forward/back numbers come from
MediaPipe's depth guess, which is noisy. This works for obvious slouching,
not small differences. The side camera (Gemini #2) sees slouching directly
and should be the final slouch detector.

TUNING (the easy way): leave LENIENCY = 1.0. If good posture gets flagged,
raise it (1.3, 1.6 ...). If real slouching is missed, lower it (0.8).
Better: stand tall and read the numbers on screen, then slouch on purpose
and read them again. Set each limit between the two.

Install:  pip install opencv-contrib-python mediapipe==0.10.14 numpy
"""

import math
import os
import sys
from collections import deque
from statistics import median

# Speeds up MSMF camera startup on Windows (harmless elsewhere)
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# ============================================
# SETTINGS
# ============================================
# One knob for everything: 1.0 = normal, >1 = more forgiving, <1 = stricter
LENIENCY = 1.0

# Base limits (multiplied by LENIENCY)
SHOULDER_TILT_MAX_DEG = 6.0     # shoulders more tilted than this = flag
HEAD_TILT_MAX_DEG = 10.0        # head rolled more than this = flag
TORSO_LEAN_MAX_DEG = 12.0       # torso leaning toward/away from camera
HEAD_FORWARD_MAX_CM = 10.0      # head in front of / behind shoulders
KNEE_MIN_DEG = 155.0            # knee angle below this = bent knees
                                # (for the knee limit, higher LENIENCY lowers it)

MIN_VISIBILITY = 0.5            # landmark confidence needed (0-1)
SMOOTH_FRAMES = 15              # frames averaged (median) for each number
BAD_FRAMES_TO_FLAG = 15         # must be bad this many frames in a row to flag
GOOD_FRAMES_TO_CLEAR = 8        # must be good this many frames to un-flag

# MediaPipe Pose landmark numbers
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK = 27, 28

REQUIRED = [NOSE, L_EAR, R_EAR, L_SH, R_SH, L_HIP, R_HIP,
            L_KNEE, R_KNEE, L_ANK, R_ANK]

METRICS = ["shoulder_tilt", "head_tilt", "torso_lean", "head_forward", "knee_bend"]


def _limits():
    """Limits after applying LENIENCY."""
    return {
        "shoulder_tilt": SHOULDER_TILT_MAX_DEG * LENIENCY,
        "head_tilt": HEAD_TILT_MAX_DEG * LENIENCY,
        "torso_lean": TORSO_LEAN_MAX_DEG * LENIENCY,
        "head_forward": HEAD_FORWARD_MAX_CM * LENIENCY,
        "knee_bend": 180.0 - (180.0 - KNEE_MIN_DEG) * LENIENCY,
    }


def _tilt_deg(p1, p2, w, h):
    """Angle (degrees) of the line p1->p2 away from horizontal, in pixels.
    0 = perfectly level. Result is between -90 and 90."""
    dx = (p2.x - p1.x) * w
    dy = (p2.y - p1.y) * h
    ang = math.degrees(math.atan2(dy, dx))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return ang


def _mid(a, b):
    return ((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)


def _angle_3d(a, b, c):
    """Angle at point b (degrees) between segments b->a and b->c."""
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    n1 = math.sqrt(sum(t * t for t in v1))
    n2 = math.sqrt(sum(t * t for t in v2))
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = sum(p * q for p, q in zip(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


class PostureChecker:
    """Call update(results, w, h) once per frame with the MediaPipe results."""

    def __init__(self):
        self.history = {m: deque(maxlen=SMOOTH_FRAMES) for m in METRICS}
        # Optional reference (P key). 0 = use absolute limits (default).
        self.baseline = {"torso_lean": 0.0, "head_forward": 0.0}
        self.baseline_set = False
        self.bad_count = 0
        self.good_count = 0
        self.flagged = False

    # ---------- measuring ----------
    def _measure(self, results, w, h):
        """Return one frame's raw numbers, or None if the body isn't clear."""
        if not results.pose_landmarks or not results.pose_world_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        wl = results.pose_world_landmarks.landmark

        if any(lm[i].visibility < MIN_VISIBILITY for i in REQUIRED):
            return None

        shoulder_tilt = abs(_tilt_deg(lm[L_SH], lm[R_SH], w, h))
        head_tilt = abs(_tilt_deg(lm[L_EAR], lm[R_EAR], w, h))

        sh = _mid(wl[L_SH], wl[R_SH])
        hip = _mid(wl[L_HIP], wl[R_HIP])
        ear = _mid(wl[L_EAR], wl[R_EAR])

        # Lean of the torso toward/away from the camera (signed degrees)
        torso_lean = math.degrees(math.atan2(sh[2] - hip[2], abs(sh[1] - hip[1])))

        # Head in front of / behind the shoulders (signed cm, camera axis)
        head_forward = (ear[2] - sh[2]) * 100.0

        # Smallest knee angle (180 = straight)
        knee_bend = min(
            _angle_3d(wl[L_HIP], wl[L_KNEE], wl[L_ANK]),
            _angle_3d(wl[R_HIP], wl[R_KNEE], wl[R_ANK]),
        )

        return {
            "shoulder_tilt": shoulder_tilt,
            "head_tilt": head_tilt,
            "torso_lean": torso_lean,
            "head_forward": head_forward,
            "knee_bend": knee_bend,
        }

    def _smoothed(self):
        return {m: median(self.history[m]) for m in METRICS if self.history[m]}

    # ---------- public ----------
    def set_baseline(self):
        """OPTIONAL: record the current posture as the reference for the two
        forward/back numbers. Not needed for normal use."""
        s = self._smoothed()
        if len(self.history["torso_lean"]) < SMOOTH_FRAMES // 2 or not s:
            return False
        self.baseline["torso_lean"] = s["torso_lean"]
        self.baseline["head_forward"] = s["head_forward"]
        self.baseline_set = True
        return True

    def update(self, results, w, h):
        """
        Returns a dict:
          status  : "ok", "bad" (flag it), or "unknown" (body not clearly visible)
          issues  : list of short messages for the person
          metrics : the smoothed numbers (plus torso_dev / head_dev)
          limits  : the limits currently in use
        """
        lim = _limits()
        raw = self._measure(results, w, h)

        if raw is None:
            for m in METRICS:
                self.history[m].clear()
            self.bad_count = self.good_count = 0
            self.flagged = False
            return {"status": "unknown", "issues": [], "metrics": {}, "limits": lim}

        for m in METRICS:
            self.history[m].append(raw[m])

        # Wait until the averaging window is mostly full, so a single
        # noisy first frame can't trigger a flag
        if len(self.history["torso_lean"]) < SMOOTH_FRAMES // 2:
            return {"status": "ok", "issues": [], "metrics": self._smoothed(),
                    "limits": lim}

        s = self._smoothed()
        torso_dev = s["torso_lean"] - self.baseline["torso_lean"]
        head_dev = s["head_forward"] - self.baseline["head_forward"]

        issues = []
        if abs(torso_dev) > lim["torso_lean"] or abs(head_dev) > lim["head_forward"]:
            issues.append("Stand up straight - don't slouch")
        if s["shoulder_tilt"] > lim["shoulder_tilt"]:
            issues.append("Level your shoulders")
        if s["head_tilt"] > lim["head_tilt"]:
            issues.append("Keep your head straight")
        if s["knee_bend"] < lim["knee_bend"]:
            issues.append("Straighten your knees")

        # Hysteresis: flag after many bad frames, clear after several good ones
        if issues:
            self.bad_count += 1
            self.good_count = 0
            if self.bad_count >= BAD_FRAMES_TO_FLAG:
                self.flagged = True
        else:
            self.good_count += 1
            self.bad_count = 0
            if self.good_count >= GOOD_FRAMES_TO_CLEAR:
                self.flagged = False

        metrics = dict(s)
        metrics["torso_dev"] = torso_dev
        metrics["head_dev"] = head_dev
        return {
            "status": "bad" if self.flagged else "ok",
            "issues": issues if self.flagged else [],
            "metrics": metrics,
            "limits": lim,
        }


# ============================================
# STANDALONE WEBCAM TEST
# ============================================
def _open_camera(index):
    import cv2
    if sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION]
    elif sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def run_demo(camera_index=0):
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = _open_camera(camera_index)
    if cap is None:
        print(f"ERROR: Camera index {camera_index} could not be opened.")
        pose.close()
        return

    checker = PostureChecker()
    name = "PREV-HUB Posture Check"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    print("Stand with your whole body in view. Press Q to quit.")
    print("Optional: press P while standing tall to use your own posture as reference.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read camera frame.")
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            state = checker.update(results, w, h)

            # Status banner
            if state["status"] == "unknown":
                color, text = (0, 165, 255), "Stand with your whole body in view"
            elif state["status"] == "bad":
                color, text = (0, 0, 255), "POSTURE: " + " | ".join(state["issues"])
            else:
                color, text = (0, 200, 0), "POSTURE OK"
            cv2.rectangle(frame, (0, h - 60), (w, h), (25, 25, 25), -1)
            cv2.putText(frame, text, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

            # Numbers panel: value (limit)
            m, lim = state["metrics"], state["limits"]
            ref = "own reference" if checker.baseline_set else "absolute limits"
            lines = [f"Mode: {ref}   Leniency: {LENIENCY:.1f}"]
            if m:
                lines += [
                    f"Shoulder tilt: {m['shoulder_tilt']:.1f} deg (limit {lim['shoulder_tilt']:.0f})",
                    f"Head tilt:     {m['head_tilt']:.1f} deg (limit {lim['head_tilt']:.0f})",
                    f"Torso lean:    {m.get('torso_dev', m['torso_lean']):+.1f} deg (limit {lim['torso_lean']:.0f})",
                    f"Head forward:  {m.get('head_dev', m['head_forward']):+.1f} cm (limit {lim['head_forward']:.0f})",
                    f"Knee angle:    {m['knee_bend']:.0f} deg (min {lim['knee_bend']:.0f})",
                ]
            cv2.rectangle(frame, (5, 5), (560, 20 + 28 * len(lines)), (25, 25, 25), -1)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (15, 30 + 28 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                if checker.set_baseline():
                    print("Reference set from your current posture.")
                else:
                    print("Can't set reference yet - stand in view for a moment first.")
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()


# ============================================
# HOW TO USE INSIDE THE HEIGHT TRACKER
# ============================================
# 1) Put this file next to the tracker and add near the top:
#        from posture_check import PostureChecker
#
# 2) Create one checker before the main loop:
#        posture = PostureChecker()
#
# 3) Right after  results = pose.process(rgb)  add:
#        posture_state = posture.update(results, w, h)
#
# 4) Only measure when posture is not flagged. In the  measuring = ( ... )
#    block add this line:
#            and posture_state["status"] != "bad"
#
# 5) Tell the patient. Before the other banner checks, add:
#        if posture_state["status"] == "bad":
#            draw_banner(frame, "STAND TALL\n" + posture_state["issues"][0],
#                        (0, 165, 255))
