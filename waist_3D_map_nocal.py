"""
Waist 3D map from edge dots while the person turns (separate from waist_turn_scan.py).
 
Idea
  On each torso row (slice) the camera sees two edge dots: the left and right edge
  of the body. As the person turns, those dots slide along the body surface.
  The hip-centre dot is the pivot that stays put. Each frame says: "at this angle,
  the body reaches exactly this far left and this far right of the pivot".
  Stack those limits from every angle and only the true waist outline fits all of
  them, so the outline (a top-down map of the slice) is carved out frame by frame.
  Doing that on several rows from hips to chest gives a 3D map of the torso.
 
No calibration step
  Pixels are turned into cm from the camera's geometry (pinhole camera model):
      cm per pixel = 2 * distance * tan(field_of_view / 2) / image_width
  So you only set two numbers below once: how far the standing spot is from the
  camera, and the camera's horizontal field of view (from its spec sheet).
  The person must stand with the middle of their body over that spot.
 
Controls
  s  start recording (stand on the spot, hands on head, face the camera)
  e  end: turn at least 180 degrees first. Builds the map and prints the waist.
  q  quit
 
Output
  Nothing is saved to disk. The waist and every slice print in the terminal, and a
  top-view map of the slices (waist in yellow) opens in its own window.
 
Requires Python 3.11 with: opencv-python, mediapipe (legacy mp.solutions API), numpy
"""
 
import math
import traceback
 
import cv2
import numpy as np
import mediapipe as mp
 
CAMERA_DISTANCE_CM = 106.7  # camera lens to the centre of the standing spot (3.5 ft = 106.7 cm)
CAMERA_HFOV_DEG = 66.8      # measured for Mustafa's MacBook webcam (198 cm away, 261 cm wide); change for the Orbbec
 
# torso rows to map, as a fraction of the way from hip dots up to shoulder dots
SLICE_ALPHAS = [0.10, 0.20, 0.30, 0.40, 0.50]
WAIST_ALPHA = 0.30    # which slice is reported as "the waist"
BAND_ROWS = 5         # rows averaged per slice
MIN_SWEEP_DEG = 170
GRID_CM = 0.25        # map resolution
MASK_THRESH = 0.5
 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
PL = mp_pose.PoseLandmark
WIN = "Waist 3D map"
 
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
 
 
def carve_slice(angles_deg, left_cm, right_cm, pivot_cm, half_extent_cm=40.0, bin_deg=5.0):
    """
    Build the top-down outline of one slice from edge dots at many angles.
    angles_deg: body rotation per frame; left/right/pivot: image x positions in cm.
 
    Each frame says how far the body reaches from the pivot in two opposite
    directions (right edge at angle a, left edge at a + 180). Those reaches are
    grouped into bin_deg bins around the full circle and the median is kept, so a
    single noisy frame can't cut into the outline. The outline is then carved as
    the region that stays inside every binned reach.
    The perimeter comes from the binned reaches directly; the carved outline is the map.
    Returns (mask, outline_points_cm, perimeter_cm).
    """
    a = np.asarray(angles_deg, float)
    l, r, p = (np.asarray(v, float) for v in (left_cm, right_cm, pivot_cm))
    dirs = np.concatenate([a, a + 180.0]) % 360.0
    reach = np.concatenate([r - p, p - l])
    n_bins = int(round(360 / bin_deg))
    idx = (dirs // bin_deg).astype(int) % n_bins
    h = np.full(n_bins, np.nan)
    for i in range(n_bins):
        if (idx == i).any():
            h[i] = np.median(reach[idx == i])
    ok = ~np.isnan(h)
    if ok.sum() < 4:
        return None, None, float("nan")
    centers = (np.arange(n_bins) + 0.5) * bin_deg
    # fill empty bins around the circle, then perimeter = integral of reach over 360 deg
    # (support-function form of Cauchy's formula; robust to angle and pivot noise)
    xs_ = np.concatenate([centers[ok] - 360, centers[ok], centers[ok] + 360])
    h_full = np.interp(centers, xs_, np.tile(h[ok], 3))
    perimeter = float(h_full.sum() * math.radians(bin_deg))
 
    n = int(2 * half_extent_cm / GRID_CM)
    g = (np.arange(n) + 0.5) * GRID_CM - half_extent_cm
    X, Y = np.meshgrid(g, g)
    inside = np.ones_like(X, dtype=bool)
    for phi, reach_i in zip(centers[ok], h[ok]):
        t = math.radians(phi)
        inside &= X * math.cos(t) + Y * math.sin(t) <= reach_i
    if not inside.any():
        return inside, None, perimeter
    ys, xs = np.nonzero(inside)
    pts = np.stack([g[xs], g[ys]], axis=1).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    return inside, hull, perimeter
 
 
def draw_map(slices, waist_idx, perimeter):
    size, scale = 600, 7.0  # px per cm
    img = np.full((size, size, 3), 255, np.uint8)
    c = size // 2
    for gx in range(-40, 41, 10):
        cv2.line(img, (int(c + gx * scale), 0), (int(c + gx * scale), size), (225, 225, 225), 1)
        cv2.line(img, (0, int(c + gx * scale)), (size, int(c + gx * scale)), (225, 225, 225), 1)
    for i, (_, hull) in enumerate(slices):
        if hull is None:
            continue
        poly = (hull * scale + c).astype(np.int32).reshape(-1, 1, 2)
        color, thick = ((0, 190, 255), 3) if i == waist_idx else ((150, 150, 150), 1)
        cv2.polylines(img, [poly], True, color, thick)
    cv2.putText(img, f"Waist slice: {perimeter:.1f} cm   (grid = 10 cm)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img
 
 
def facing_angle_deg(world):
    lh, rh = world[PL.LEFT_HIP], world[PL.RIGHT_HIP]
    return math.degrees(math.atan2(rh.z - lh.z, rh.x - lh.x))
 
 
def cm_per_px_from_geometry(image_width_px):
    """Pinhole camera: width the camera sees at the standing spot, divided by image width."""
    seen_width_cm = 2 * CAMERA_DISTANCE_CM * math.tan(math.radians(CAMERA_HFOV_DEG) / 2)
    return seen_width_cm / image_width_px
 
 
def main():
    pose = mp_pose.Pose(model_complexity=1, enable_segmentation=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return
 
    cm_per_px = None  # set from the first frame's width
 
    recording = False
    frames = []           # (angle, slice_rows, [(left, right) or None per slice], pivot_x)
    skipped = 0
    prev2, unwrapped2 = None, 0.0
    msg = ""
 
    print("s = start, e = end + build map, q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w, _ = frame.shape
        if cm_per_px is None:
            cm_per_px = cm_per_px_from_geometry(w)
            print(f"Scale from camera geometry: {cm_per_px:.4f} cm/px "
                  f"({CAMERA_DISTANCE_CM:.0f} cm away, {CAMERA_HFOV_DEG:.0f} deg view, {w} px wide)")
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status = "Searching for body..."
 
        if res.pose_landmarks and res.segmentation_mask is not None and res.pose_world_landmarks:
            lm = res.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            hip_y = (lm[PL.LEFT_HIP].y + lm[PL.RIGHT_HIP].y) / 2 * h
            sh_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2 * h
            pivot_x = (lm[PL.LEFT_HIP].x + lm[PL.RIGHT_HIP].x) / 2 * w  # the dot that stays
 
            raw2 = 2 * facing_angle_deg(res.pose_world_landmarks.landmark)
            if prev2 is None:
                prev2 = raw2
            unwrapped2 += (raw2 - prev2 + 180) % 360 - 180
            prev2 = raw2
            angle = unwrapped2 / 2
 
            rows = [int(hip_y - a * (hip_y - sh_y)) for a in SLICE_ALPHAS]
            top, bottom = min(rows) - 0.05 * h, max(rows) + 0.05 * h
            arms_in_torso = any(top < lm[p].y * h < bottom and lm[p].visibility > 0.5
                                for p in (PL.LEFT_WRIST, PL.RIGHT_WRIST, PL.LEFT_ELBOW, PL.RIGHT_ELBOW))
            hips_ok = min(lm[PL.LEFT_HIP].visibility, lm[PL.RIGHT_HIP].visibility) > 0.5
 
            dots = [edge_dots(res.segmentation_mask, y, pivot_x) for y in rows]
            for y, d, a in zip(rows, dots, SLICE_ALPHAS):
                if d is None:
                    continue
                col = (0, 190, 255) if a == WAIST_ALPHA else (200, 200, 200)
                if arms_in_torso:
                    col = (0, 0, 255)
                cv2.circle(frame, (int(d[0]), y), 6, col, -1)  # left edge dot
                cv2.circle(frame, (int(d[1]), y), 6, col, -1)  # right edge dot
            cv2.circle(frame, (int(pivot_x), int(hip_y)), 7, (255, 0, 255), -1)  # pivot
 
            status = f"facing {angle:6.1f} deg"
            if arms_in_torso:
                status += "   ARMS IN THE WAY - hands on head"
            if recording:
                if hips_ok and not arms_in_torso and all(d is not None for d in dots):
                    frames.append((angle, rows, dots, pivot_x))
                else:
                    skipped += 1
 
        if recording:
            span = (max(f[0] for f in frames) - min(f[0] for f in frames)) if frames else 0
            cv2.putText(frame, f"REC {len(frames)} kept, {skipped} skipped   turned {span:5.1f} / {MIN_SWEEP_DEG}",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if msg:
            cv2.putText(frame, msg, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow(WIN, frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            frames.clear()
            skipped, recording, msg = 0, True, ""
        if key == ord("e") and recording:
            recording = False
            try:
                if len(frames) < 10:
                    msg = "Not enough frames - hands on head and turn again"
                    continue
                angles = np.array([f[0] for f in frames])
                angles -= angles.min()
                sweep = angles.max()
                print(f"Kept {len(frames)} frames, skipped {skipped}, turned {sweep:.0f} deg")
                if sweep < MIN_SWEEP_DEG:
                    msg = f"Only {sweep:.0f} deg usable - turn at least {MIN_SWEEP_DEG}"
                    print(msg)
                    continue
 
                waist_idx = SLICE_ALPHAS.index(WAIST_ALPHA)
 
                def build(scale, report):
                    pivot = np.array([f[3] for f in frames]) * scale
                    hip_rows = np.array([f[1][0] for f in frames], float)
                    out, waist = [], float("nan")
                    for k, a in enumerate(SLICE_ALPHAS):
                        left = np.array([f[2][k][0] for f in frames]) * scale
                        right = np.array([f[2][k][1] for f in frames]) * scale
                        _, hull, perim = carve_slice(angles, left, right, pivot)
                        # slice height above the lowest slice, in cm
                        z = float(np.median(hip_rows - np.array([f[1][k] for f in frames]))) * scale
                        out.append((z, hull))
                        if report:
                            print(f"Slice {a:.2f} (z = {z:5.1f} cm): perimeter {perim:6.1f} cm")
                        if k == waist_idx:
                            waist = perim
                    return out, waist
 
                slices, waist_perim = build(cm_per_px, True)
                img = draw_map(slices, waist_idx, waist_perim)
                cv2.imshow("Waist map (top view)", img)
                print(f"WAIST: {waist_perim:.1f} cm")
                msg = f"Waist ~ {waist_perim:.1f} cm (see map window)"
            except Exception:
                traceback.print_exc()
                msg = "Error building the map - see the terminal"
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 