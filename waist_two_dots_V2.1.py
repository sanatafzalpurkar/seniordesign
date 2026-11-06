"""
Waist from two edge dots during one full turn (separate from the other scripts).
 
Idea
  Two dots sit on the left and right edge of the waist. As the person turns, the
  dots slide around the body: one moves while the other barely does, then they
  swap. For every few degrees of turning the code works out how far the dots slid
  along the skin and adds it to a running total for each dot. Once the person is
  back where they started, waist = dot A total + dot B total.
  (This is the same answer a tape gives: dents like the spine groove are bridged.)
 
Start / stop
  Put a strip of tape on the floor at the standing spot with an arrow pointing
  at the camera. Stand on it facing the camera and press s. The code remembers
  that facing direction and stops by itself once you've turned all the way round. It counts the
  turn from the waist width: wide (facing) -> narrow (side) -> wide (back) ->
  narrow (side) -> wide (facing again).
 
Waist height
  Default is ALPHA of the way from the hip dots up to the shoulder dots. Use
  [ and ] to move the orange line up or down live until it sits at the belly
  button (midpoint between lowest rib and top of hip bone). The terminal prints
  the value; paste it into ALPHA so it's used every time.
 
Controls
  s  start (face the camera on the floor arrow, arms straight out to the sides)
  [  move the waist line down     ]  move the waist line up
  e  stop early                   q  quit
 
Scale: no calibration, from the camera's geometry (pinhole model):
  cm per pixel = 2 * distance * tan(field_of_view / 2) / image_width
 
Nothing is saved; results show on screen and in the terminal.
Requires Python 3.11 with: opencv-python, mediapipe (legacy mp.solutions API), numpy
"""
 
import math
import time
 
import cv2
import numpy as np
import mediapipe as mp
 
CAMERA_DISTANCE_CM = 106.7  # camera lens to the centre of the standing spot (3.5 ft)
CAMERA_HFOV_DEG = 66.8      # measured for Mustafa's MacBook webcam; change for the Orbbec
 
ALPHA = 0.45          # waist line: fraction of the way from hip dots up to shoulder dots
ALPHA_STEP = 0.02     # how far [ and ] move the line
TURN_FRAMES = 5       # frames in a row needed to count each quarter of the turn
NARROW = 0.85         # side-on: waist looks narrower than this fraction of the starting width
WIDE = 0.92           # facing / back: waist looks at least this wide again
BIN_DEG = 5.0         # turning is grouped into 5 degree steps
BAND_ROWS = 5
MASK_THRESH = 0.5
 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
PL = mp_pose.PoseLandmark
WIN = "Waist two dots"
 
 
def edge_dots(mask, y, center_x):
    """Left and right body edge on row y, following the body run through center_x."""
    h, w = mask.shape
    lefts, rights = [], []
    cx = int(np.clip(center_x, 0, w - 1))
    for row in range(max(0, y - BAND_ROWS // 2), min(h, y + BAND_ROWS // 2 + 1)):
        on = mask[row] > MASK_THRESH
        if not on[cx]:
            continue
        left = cx
        while left > 0 and on[left - 1]:
            left -= 1
        right = cx
        while right < w - 1 and on[right + 1]:
            right += 1
        lefts.append(left)
        rights.append(right + 1)
    if not lefts:
        return None
    return float(np.median(lefts)), float(np.median(rights))
 
 
def dot_travel(angles_deg, left_cm, right_cm, centre_cm):
    """
    How far each dot slid along the skin, and the waist.
 
    For each 5 degree step of the turn we know how far the body reaches out to
    each dot (measured from a body-fixed centre). When the body turns a little,
    the touching point slides along the skin by (reach + its curvature term) * step.
    Adding those slides up over the whole turn gives the distance each dot travelled.
    Dot A covers directions a, dot B covers a + 180. Over a full turn both dots
    see every direction, so each 5 degree step is counted once (median of both)
    and credited to the dots in proportion; dot A + dot B = the waist.
    Returns (dot_a_cm, dot_b_cm, waist_cm, outline_points_cm).
    """
    a = np.asarray(angles_deg, float)
    l, r, c = (np.asarray(v, float) for v in (left_cm, right_cm, centre_cm))
    dirs = np.concatenate([a, a + 180.0]) % 360.0
    reach = np.concatenate([r - c, c - l])
    dot = np.concatenate([np.zeros(len(a), int), np.ones(len(a), int)])
 
    n = int(round(360 / BIN_DEG))
    idx = (dirs // BIN_DEG).astype(int) % n
    h = np.full(n, np.nan)
    share_b = np.zeros(n)  # fraction of each step measured by dot B
    for i in range(n):
        m = idx == i
        if m.any():
            h[i] = np.median(reach[m])
            share_b[i] = dot[m].mean()
    ok = ~np.isnan(h)
    if ok.sum() < 4:
        return float("nan"), float("nan"), float("nan"), None
    centers = (np.arange(n) + 0.5) * BIN_DEG
    xs = np.concatenate([centers[ok] - 360, centers[ok], centers[ok] + 360])
    h = np.interp(centers, xs, np.tile(h[ok], 3))  # fill any gaps in the turn
 
    d = math.radians(BIN_DEG)
    h2 = (np.roll(h, -1) - 2 * h + np.roll(h, 1)) / d ** 2
    slide = (h + h2) * d                  # distance the touching dot slid in each step
    dot_a = float((slide * (1 - share_b)).sum())
    dot_b = float((slide * share_b).sum())
    waist = float(slide.sum())            # each direction counted once = one lap of the waist
 
    # outline for the map: where each dot touched the body
    h1 = (np.roll(h, -1) - np.roll(h, 1)) / (2 * d)
    t = np.radians(centers)
    pts = np.stack([h * np.cos(t) - h1 * np.sin(t), h * np.sin(t) + h1 * np.cos(t)], 1)
    return dot_a, dot_b, waist, pts
 
 
def draw_map(pts, waist):
    size, scale = 600, 7.0
    img = np.full((size, size, 3), 255, np.uint8)
    c = size // 2
    for g in range(-40, 41, 10):
        cv2.line(img, (int(c + g * scale), 0), (int(c + g * scale), size), (225, 225, 225), 1)
        cv2.line(img, (0, int(c + g * scale)), (size, int(c + g * scale)), (225, 225, 225), 1)
    if pts is not None:
        poly = (pts * scale + c).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, (0, 140, 255), 3)
    cv2.putText(img, f"Waist: {waist:.1f} cm   (grid = 10 cm)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img
 
 
def assign_angles(frames):
    """
    Turn angle for every kept frame. MediaPipe's own angle reads short when the
    person is side-on or facing away (a full turn can show as ~250 deg), so the
    angles are pinned to things the camera sees reliably instead:
      start = 0 deg (facing camera), end = 360 deg (facing camera again),
      widest view in the middle = 180 deg (back to camera),
      narrowest view before / after that = 90 / 270 deg (side-on).
    Between those, the angle is spread evenly by time.
    Falls back to stretching MediaPipe's angles to 360 if the pattern isn't clear.
    """
    t = np.array([f[4] for f in frames])
    width = np.array([f[2] - f[1] for f in frames])
    raw = np.array([f[0] for f in frames])
    n = len(frames)
    k = max(3, n // 40) | 1
    pad = np.pad(width, k // 2, mode="edge")
    smooth = np.array([np.median(pad[i:i + k]) for i in range(n)])
    lo, hi = int(n * 0.3), int(n * 0.7)
    if n >= 40 and hi > lo:
        i_back = lo + int(np.argmax(smooth[lo:hi]))
        i_side1 = int(np.argmin(smooth[:i_back])) if i_back > 2 else 0
        i_side2 = i_back + int(np.argmin(smooth[i_back:])) if n - i_back > 2 else n - 1
        anchors = [0, i_side1, i_back, i_side2, n - 1]
        sides_ok = smooth[i_side1] < 0.9 * smooth[i_back] and smooth[i_side2] < 0.9 * smooth[i_back]
        if sides_ok and 0 < i_side1 < i_back < i_side2 < n - 1:
            return np.interp(t, t[anchors], [0.0, 90.0, 180.0, 270.0, 360.0]), "width pattern"
    span = raw.max() - raw.min()
    return (raw - raw.min()) * (360.0 / span if span > 0 else 1.0), "stretched MediaPipe angle"
 
 
def facing_angle_deg(world):
    lh, rh = world[PL.LEFT_HIP], world[PL.RIGHT_HIP]
    return math.degrees(math.atan2(rh.z - lh.z, rh.x - lh.x))
 
 
def main():
    global ALPHA
    pose = mp_pose.Pose(model_complexity=1, enable_segmentation=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return
 
    cm_per_px = None
    recording = False
    frames = []           # (unwrapped angle, left_px, right_px, centre_px)
    skipped = 0
    prev2, unwrapped2 = None, 0.0
    msg = ""
    phase, run = 0, 0      # quarter turns completed (4 = nearly back, 5 = done)
    peak_w, peak_n, since_peak = 0.0, 0, 0
    print("s = start, [ / ] = move waist line, e = stop early, q = quit")
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w, _ = frame.shape
        if cm_per_px is None:
            cm_per_px = 2 * CAMERA_DISTANCE_CM * math.tan(math.radians(CAMERA_HFOV_DEG) / 2) / w
            print(f"Scale from camera geometry: {cm_per_px:.4f} cm/px")
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status = "Searching for body..."
        turned = 0.0
        face_visible = False
 
        if res.pose_landmarks and res.segmentation_mask is not None and res.pose_world_landmarks:
            lm = res.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            hip_y = (lm[PL.LEFT_HIP].y + lm[PL.RIGHT_HIP].y) / 2 * h
            sh_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2 * h
            centre_x = (lm[PL.LEFT_HIP].x + lm[PL.RIGHT_HIP].x) / 2 * w  # body-fixed reference, not drawn
            waist_y = int(hip_y - ALPHA * (hip_y - sh_y))
            face_visible = min(lm[PL.NOSE].visibility, lm[PL.LEFT_EYE].visibility,
                               lm[PL.RIGHT_EYE].visibility) > 0.6
 
            raw2 = 2 * facing_angle_deg(res.pose_world_landmarks.landmark)
            if prev2 is None:
                prev2 = raw2
            unwrapped2 += (raw2 - prev2 + 180) % 360 - 180
            prev2 = raw2
            angle = unwrapped2 / 2
 
            band = 0.08 * h
            arms_at_waist = any(abs(lm[p].y * h - waist_y) < band and lm[p].visibility > 0.5
                                for p in (PL.LEFT_WRIST, PL.RIGHT_WRIST, PL.LEFT_ELBOW, PL.RIGHT_ELBOW))
            hips_ok = min(lm[PL.LEFT_HIP].visibility, lm[PL.RIGHT_HIP].visibility) > 0.5
 
            cv2.line(frame, (0, waist_y), (w, waist_y), (0, 165, 255), 1)
            dots = edge_dots(res.segmentation_mask, waist_y, centre_x)
            if dots is not None:
                col = (0, 0, 255) if arms_at_waist else (0, 165, 255)
                cv2.circle(frame, (int(dots[0]), waist_y), 8, col, -1)  # dot B
                cv2.circle(frame, (int(dots[1]), waist_y), 8, col, -1)  # dot A
                width_cm = (dots[1] - dots[0]) * cm_per_px
                status = f"distance between dots {width_cm:5.1f} cm   line at {ALPHA:.2f}"
            if arms_at_waist:
                status += "   ARMS IN THE WAY - arms straight out"
 
            if recording:
                if dots is not None and hips_ok and not arms_at_waist:
                    frames.append((angle, dots[0], dots[1], centre_x, time.time()))
                else:
                    skipped += 1
            if frames:
                turned = max(f[0] for f in frames) - min(f[0] for f in frames)
 
        if recording:
            cv2.putText(frame, f"REC {len(frames)} kept, {skipped} skipped   " + f"quarter turns {min(phase, 4)}/4 - keep turning",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if len(frames) > 20 and turned >= 90:
                a = [f[0] for f in frames]
                da, db, so_far, _ = dot_travel(a, *(np.array([f[i] for f in frames]) * cm_per_px for i in (1, 2, 3)))
                cv2.putText(frame, f"dot A {da:5.1f} cm   dot B {db:5.1f} cm", (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if msg:
            cv2.putText(frame, msg, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow(WIN, frame)
 
        key = cv2.waitKey(1) & 0xFF
        if recording and res.pose_landmarks:
            # count the turn by the waist width only (MediaPipe often "finds" a face on the
            # back of the head, so the face can't be trusted): facing = wide, side-on = narrow.
            # wide -> narrow (side) -> wide (back) -> narrow (side) -> wide (facing again) = done
            if len(frames) >= 8:
                start_w = np.median([f[2] - f[1] for f in frames[:5]])
                now_w = np.median([f[2] - f[1] for f in frames[-3:]])
                want_narrow = phase in (0, 2)
                hit = now_w < NARROW * start_w if want_narrow else now_w > WIDE * start_w
                run = run + 1 if hit else 0
                if run >= TURN_FRAMES and phase < 4:
                    phase, run = phase + 1, 0
                    peak_w, peak_n, since_peak = now_w, len(frames), 0
                    print(["", "Side-on", "Back to the camera", "Side-on again", "Almost facing the camera"][phase])
                elif phase == 4:
                    # keep going until the waist is at its widest (squarely facing), then stop there
                    if now_w > peak_w:
                        peak_w, peak_n, since_peak = now_w, len(frames), 0
                    else:
                        since_peak += 1
                    if since_peak >= 10 or now_w < 0.97 * peak_w:
                        del frames[peak_n:]   # end the turn at the widest (facing) frame
                        phase = 5
        if recording and phase == 5:
            print("Back to the start - stopping automatically")
            key = ord("e")
        if key == ord("q"):
            break
        if key == ord("["):
            ALPHA = max(0.0, ALPHA - ALPHA_STEP)
            print(f"ALPHA = {ALPHA:.2f}")
        if key == ord("]"):
            ALPHA = min(1.0, ALPHA + ALPHA_STEP)
            print(f"ALPHA = {ALPHA:.2f}")
        if key == ord("s"):
            frames.clear()
            skipped, recording, msg = 0, True, ""
            phase, run = 0, 0
        if key == ord("e") and recording:
            recording = False
            if len(frames) < 20:
                msg = "Not enough frames - arms out and turn again"
                continue
            a, how = assign_angles(frames)
            turned = 360.0 if phase == 5 else a.max() - a.min()
            print(f"Turn angles from: {how}")
            left, right, centre = (np.array([f[i] for f in frames]) * cm_per_px for i in (1, 2, 3))
            da, db, waist, pts = dot_travel(a, left, right, centre)
            print(f"Kept {len(frames)} frames, skipped {skipped}, turned {turned:.0f} deg")
            if turned < 180:
                msg = f"Only {turned:.0f} deg - turn all the way round"
                print(msg)
                continue
            print(f"Dot A slid {da:.1f} cm, dot B slid {db:.1f} cm")
            print(f"WAIST: {waist:.1f} cm   (waist line at ALPHA = {ALPHA:.2f})")
            if phase < 5:
                print("Note: stopped before facing the camera again; result is less reliable")
            msg = f"Waist ~ {waist:.1f} cm"
            cv2.imshow("Waist map (top view)", draw_map(pts, waist))
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 