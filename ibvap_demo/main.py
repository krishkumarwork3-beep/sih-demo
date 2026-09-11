import argparse
import json
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
SNAPSHOT_DIR = os.path.join(WEB_DIR, "snapshots")
ALERTS_PATH = os.path.join(WEB_DIR, "alerts.json")
DETECTIONS_PATH = os.path.join(WEB_DIR, "detections.json")
FRAME_PATH = os.path.join(WEB_DIR, "latest_frame.jpg")
TRACKER_CONFIG = os.path.join(BASE_DIR, "custom_tracker.yaml")

PERSON_CLASS = 0
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO ids)

BOX_COLOR_NORMAL = (0, 200, 0)     # green
BOX_COLOR_ALERT = (0, 0, 255)      # red — permanent once this track crosses the fence

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

try:
    import easyocr
    OCR_READER = easyocr.Reader(["en"], gpu=True)
except Exception as exc:  # pragma: no cover
    print(f"EasyOCR unavailable ({exc}) — ANPR will be skipped.")
    OCR_READER = None


def load_config(path):
    if not os.path.exists(path):
        print(f"No config found at {path} — fence crossing detection disabled.")
        return {}
    with open(path) as f:
        return json.load(f)


def side_of_line(point, a, b):
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


FENCE_DEADZONE_PX = 20  # a box center must be at least this far from the fence
                        # line (in pixels) before we trust which side it's on.
                        # Without this, someone simply walking PARALLEL to the
                        # line — or two different people swapped onto the same
                        # id right at the boundary — can flicker across the
                        # line by a pixel of detection noise and fire a false
                        # "crossing" alert every time it flickers back and forth.


def confirmed_side(point, a, b):
    """Which side of the fence line `point` is confidently on, or 0 ("too
    close to call") if it's within FENCE_DEADZONE_PX of the line."""
    cross = side_of_line(point, a, b)
    length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    if length == 0:
        return 0
    dist_px = cross / length  # signed perpendicular distance, in pixels
    if abs(dist_px) < FENCE_DEADZONE_PX:
        return 0
    return 1 if dist_px > 0 else -1


def load_alerts():
    if os.path.exists(ALERTS_PATH):
        with open(ALERTS_PATH) as f:
            return json.load(f)
    return []


def save_alerts(alerts):
    with open(ALERTS_PATH, "w") as f:
        json.dump(alerts[-50:], f, indent=2)


def load_detections():
    if os.path.exists(DETECTIONS_PATH):
        with open(DETECTIONS_PATH) as f:
            return json.load(f)
    return []


def save_detections(detections):
    with open(DETECTIONS_PATH, "w") as f:
        json.dump(detections[-30:], f, indent=2)


def compute_risk(off_hours):
    # Simplified stand-in for Step 29's weighted formula.
    zone_violation = 1  # every fence crossing counts as one, for demo purposes
    score = zone_violation * 30 + (10 if off_hours else 0)
    if score >= 40:
        severity = "High"
    elif score >= 30:
        severity = "Medium"
    else:
        severity = "Low"
    return score, severity


def is_off_hours(config):
    window = config.get("allowed_time_window", {"start_hour": 6, "end_hour": 18})
    hour = datetime.now().hour
    return not (window["start_hour"] <= hour < window["end_hour"])


def draw_outlined_text(img, text, origin, scale=1.1, thickness=2,
                        text_color=(255, 255, 255), outline_color=(0, 0, 0),
                        outline_thickness=3):
    """Draw text with a solid outline so it stays readable over any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, origin, font, scale, outline_color,
                thickness + outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, origin, font, scale, text_color,
                thickness, cv2.LINE_AA)


def read_plate(frame, box):
    if OCR_READER is None:
        return None
    x1, y1, x2, y2 = box
    h = y2 - y1
    crop = frame[y1 + int(h * 0.5):y2, x1:x2]  # lower half — rough plate region
    if crop.size == 0:
        return None
    results = OCR_READER.readtext(crop)
    if not results:
        return None
    best = max(results, key=lambda r: r[2])
    return best[1] if best[2] > 0.4 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to sample video, or 0 for webcam")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model weights")
    parser.add_argument("--imgsz", type=int, default=960,
                         help="Inference resolution. Higher = better small/distant object "
                              "detection but slower. Try 1280 if far-away vehicles/people "
                              "still flicker in and out. Default YOLO is 640.")
    parser.add_argument("--force-off-hours", action="store_true",
                         help="Treat every frame as off-hours, so you can show severity "
                              "escalate on demand instead of waiting for real nighttime")
    args = parser.parse_args()

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # Fresh run = fresh dashboard: wipe out whatever the previous run left behind
    # so old alerts/detections/snapshots don't linger when you reload localhost.
    for old_snap in os.listdir(SNAPSHOT_DIR):
        try:
            os.remove(os.path.join(SNAPSHOT_DIR, old_snap))
        except OSError:
            pass
    save_alerts([])
    save_detections([])

    config = load_config(args.config)
    fence = config.get("fence")
    if not fence:
        print("No fence line in config — run fence_setup.py first if you want crossing alerts.")

    source = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.video}")

    model = YOLO(args.model)
    prev_side = {}
    alerts = []       # start empty every run — no stale alerts from a previous session
    detections = []   # same for the recognition log
    face_last_logged = {}
    plate_last_logged = {}
    crossed_ids = set()  # track_ids that have ever crossed the fence — stay red for good
    LOG_COOLDOWN_SEC = 4  # avoid spamming the same track's face/plate every frame

    pending_side = {}     # track_id -> candidate "other side" being evaluated
    pending_count = {}    # track_id -> how many consecutive frames it's held
    CROSS_CONFIRM_FRAMES = 4  # side must persist this many frames straight to count

    # --- ID-switch diagnostics ---
    # Whenever a brand-new track_id appears close to where a different track_id
    # just vanished, log it. This doesn't fix anything — it just tells you WHY
    # a switch happened (occlusion? low confidence? overlap with another box?)
    # so we're not guessing from screenshots.
    ever_seen_ids = set()
    last_pos = {}          # track_id -> (cx, cy, cls, frame_no, conf)
    ID_SWITCH_LOG = os.path.join(BASE_DIR, "id_switch_log.txt")
    ID_SWITCH_DIST_PX = 120   # how close counts as "probably the same person"
    ID_SWITCH_WINDOW_SEC = 3  # how recently the old id must have vanished
    open(ID_SWITCH_LOG, "w").close()  # fresh log each run
    frame_no = 0

    # --- Identity-swap diagnostics ---
    # A different failure mode from above: the SAME track_id silently starts
    # tracking a different physical person (common when two people cross paths
    # very close together). No new id is created, so the check above can't see
    # it. We catch it by keeping a rough color-histogram "fingerprint" of each
    # id's crop and flagging a sharp, sustained change in appearance.
    track_hist = {}          # track_id -> last known color histogram
    track_height = {}        # track_id -> last known box height (px)
    swap_last_logged = {}    # track_id -> time.time() of last swap warning (cooldown)
    IDENTITY_SIM_THRESH = 0.45   # below this correlation = "looks like someone else"
    # A person's real height doesn't change frame to frame. We're only ever comparing
    # a track's height to ITS OWN height a few frames earlier (not to other people
    # elsewhere in the frame), so the person hasn't had time to move meaningfully
    # closer/farther from the camera — meaning we don't need perspective/depth
    # correction for this to be a valid check. Comparing height across the whole
    # frame (different people, different depths) would need that; this doesn't.
    HEIGHT_RATIO_MIN = 0.7   # more than ~30% shorter than a moment ago...
    HEIGHT_RATIO_MAX = 1.35  # ...or ~35% taller = probably a different, physical person
    IDENTITY_LOG_COOLDOWN_SEC = 5

    def crop_histogram(img, box):
        x1, y1, x2, y2 = box
        crop = img[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    print("Running. Press Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream.")
                break

            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                classes=list({PERSON_CLASS} | VEHICLE_CLASSES),
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            annotated = frame.copy()
            off_hours = args.force_off_hours or is_off_hours(config)
            frame_no += 1

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                clss = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                current_ids = set(ids.tolist())

                for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    label = "person" if cls == PERSON_CLASS else "vehicle"

                    if track_id not in ever_seen_ids:
                        ever_seen_ids.add(track_id)
                        # Is there a recently-vanished track sitting right near this
                        # brand-new one? If so, this new id is probably the SAME
                        # physical person/vehicle getting a fresh id — log why.
                        best_match = None
                        for old_id, (ox, oy, ocls, oframe, oconf) in last_pos.items():
                            if old_id in current_ids or old_id == track_id:
                                continue  # that old id is still alive this frame, not a switch
                            age_sec = (frame_no - oframe) / max(fps, 1)
                            if age_sec > ID_SWITCH_WINDOW_SEC:
                                continue
                            if ocls != cls:
                                continue
                            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                            if dist <= ID_SWITCH_DIST_PX:
                                if best_match is None or dist < best_match[1]:
                                    best_match = (old_id, dist, age_sec, oconf)
                        if best_match:
                            old_id, dist, age_sec, oconf = best_match
                            msg = (
                                f"[frame {frame_no}] possible ID switch: #{old_id} (conf {oconf:.2f}) "
                                f"vanished {age_sec:.2f}s ago -> new #{track_id} (conf {conf:.2f}) "
                                f"appeared {dist:.0f}px away, class={label}"
                            )
                            print(msg)
                            with open(ID_SWITCH_LOG, "a") as f:
                                f.write(msg + "\n")

                    if cls == PERSON_CLASS:
                        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                        if crop.size > 0:
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
                            if len(faces) > 0:
                                draw_outlined_text(annotated, "face", (x1, y2 + 16),
                                                    text_color=(255, 200, 0))
                                now = time.time()
                                if track_id not in face_last_logged or now - face_last_logged[track_id] > LOG_COOLDOWN_SEC:
                                    detections.append({
                                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                                        "type": "face",
                                        "track_id": int(track_id),
                                        "detail": "face detected",
                                    })
                                    save_detections(detections)
                                    face_last_logged[track_id] = now

                        hist = crop_histogram(frame, (x1, y1, x2, y2))
                        height_px = y2 - y1

                        appearance_changed = False
                        height_changed = False
                        sim = None
                        height_ratio = None

                        if hist is not None:
                            prev_hist = track_hist.get(track_id)
                            if prev_hist is not None:
                                sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                                appearance_changed = sim < IDENTITY_SIM_THRESH

                        prev_height = track_height.get(track_id)
                        if prev_height and prev_height > 0:
                            height_ratio = height_px / prev_height
                            height_changed = not (HEIGHT_RATIO_MIN <= height_ratio <= HEIGHT_RATIO_MAX)

                        if appearance_changed or height_changed:
                            # Whoever this id is tracking now is probably NOT who it was
                            # tracking a moment ago. Wipe this id's history so a swap
                            # doesn't leave a stale "already crossed -> stays red forever"
                            # flag stuck on an unrelated, innocent person.
                            was_crossed = track_id in crossed_ids
                            crossed_ids.discard(track_id)
                            prev_side.pop(track_id, None)
                            pending_side.pop(track_id, None)
                            pending_count.pop(track_id, None)
                            face_last_logged.pop(track_id, None)
                            plate_last_logged.pop(track_id, None)

                            last_logged = swap_last_logged.get(track_id, 0)
                            if time.time() - last_logged > IDENTITY_LOG_COOLDOWN_SEC:
                                reasons = []
                                if appearance_changed:
                                    reasons.append(f"appearance similarity {sim:.2f}")
                                if height_changed:
                                    reasons.append(
                                        f"height ratio {height_ratio:.2f}x "
                                        f"({int(prev_height)}px -> {int(height_px)}px)"
                                    )
                                msg = (
                                    f"[frame {frame_no}] possible IDENTITY SWAP: #{track_id} "
                                    f"({', '.join(reasons)}) — this id may now be tracking a "
                                    f"different person than a moment ago."
                                    + (" Cleared its red 'already crossed' flag." if was_crossed else "")
                                )
                                print(msg)
                                with open(ID_SWITCH_LOG, "a") as f:
                                    f.write(msg + "\n")
                                swap_last_logged[track_id] = time.time()

                        if hist is not None:
                            track_hist[track_id] = hist
                        track_height[track_id] = height_px

                        if fence:
                            a, b = fence[0], fence[1]
                            sign = confirmed_side((cx, cy), a, b)
                            stable = prev_side.get(track_id)

                            if sign == 0:
                                pass  # too close to the line to trust — don't touch any state
                            elif stable is None:
                                # first confident reading for this track: just establish
                                # a baseline, don't fire an alert off the very first frame
                                prev_side[track_id] = sign
                            elif sign == stable:
                                # still confidently on the same side — cancel any
                                # budding "other side" candidate from stray noise
                                pending_count[track_id] = 0
                            else:
                                # differs from the last confirmed side. Require this to
                                # hold for several frames in a row before trusting it —
                                # filters a single noisy/mixed-up frame (jitter, or a
                                # brief identity mixup) from counting as a real crossing.
                                if pending_side.get(track_id) == sign:
                                    pending_count[track_id] = pending_count.get(track_id, 0) + 1
                                else:
                                    pending_side[track_id] = sign
                                    pending_count[track_id] = 1

                                if pending_count[track_id] >= CROSS_CONFIRM_FRAMES:
                                    direction = "inbound" if sign > 0 else "outbound"
                                    score, severity = compute_risk(off_hours)
                                    snap_name = f"track{track_id}_{int(time.time())}.jpg"
                                    cv2.imwrite(os.path.join(SNAPSHOT_DIR, snap_name), frame)
                                    alert = {
                                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                                        "track_id": int(track_id),
                                        "event_type": "virtual_fence_crossing",
                                        "direction": direction,
                                        "off_hours": off_hours,
                                        "risk_score": score,
                                        "severity": severity,
                                        "snapshot": f"snapshots/{snap_name}",
                                    }
                                    alerts.append(alert)
                                    save_alerts(alerts)
                                    print(f"ALERT: {alert}")
                                    crossed_ids.add(track_id)  # box for this track stays red from now on
                                    prev_side[track_id] = sign
                                    pending_count[track_id] = 0

                    # Box turns red the moment this track crosses the fence, and stays
                    # red for the rest of the run (even if it crosses back).
                    box_color = BOX_COLOR_ALERT if track_id in crossed_ids else BOX_COLOR_NORMAL

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    draw_outlined_text(annotated, f"{label} #{track_id}", (x1, max(y1 - 8, 12)))

                    if cls in VEHICLE_CLASSES:
                        plate = read_plate(frame, (x1, y1, x2, y2))
                        if plate:
                            draw_outlined_text(annotated, f"plate: {plate}", (x1, y2 + 16),
                                                text_color=(0, 150, 255))
                            now = time.time()
                            if track_id not in plate_last_logged or now - plate_last_logged[track_id] > LOG_COOLDOWN_SEC:
                                detections.append({
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "type": "plate",
                                    "track_id": int(track_id),
                                    "detail": plate,
                                })
                                save_detections(detections)
                                plate_last_logged[track_id] = now

                    last_pos[track_id] = (cx, cy, cls, frame_no, float(conf))

            if fence:
                cv2.line(annotated, tuple(fence[0]), tuple(fence[1]), (0, 0, 255), 2)

            cv2.imwrite(FRAME_PATH, annotated)
            cv2.imshow("IBVAP demo (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()