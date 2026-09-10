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
TRACKER_CONFIG = os.path.join(BASE_DIR, "custom_bytetrack.yaml")

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
                verbose=False,
            )[0]

            annotated = frame.copy()
            off_hours = args.force_off_hours or is_off_hours(config)

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                clss = results.boxes.cls.cpu().numpy().astype(int)

                for box, track_id, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    label = "person" if cls == PERSON_CLASS else "vehicle"

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

                        if fence:
                            a, b = fence[0], fence[1]
                            s = side_of_line((cx, cy), a, b)
                            sign = 1 if s > 0 else (-1 if s < 0 else 0)
                            prev = prev_side.get(track_id)
                            if prev is not None and sign != 0 and prev != 0 and sign != prev:
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
                            if sign != 0:
                                prev_side[track_id] = sign

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