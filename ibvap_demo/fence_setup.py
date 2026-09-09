import argparse
import json
import sys

import cv2

MAX_DISPLAY_W = 960
MAX_DISPLAY_H = 720

points = []       # points in DISPLAY (resized) coordinates
WINDOW = "Click two points (press 'n' for next frame, Esc to cancel)"


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append([x, y])
        print(f"Point {len(points)}: ({x}, {y}) [display coords]")


def grab_frame(cap, seconds):
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    return ok, frame


def fit_to_display(frame):
    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)  # never upscale
    disp_w, disp_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (disp_w, disp_h)), scale


def main():
    global points
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to a sample video file, or 0 for webcam")
    parser.add_argument("--config", default="config.json", help="Config file to update")
    parser.add_argument("--seconds", type=float, default=2.0,
                         help="Start this many seconds into the video, to skip intro/title cards (default: 2)")
    args = parser.parse_args()

    source = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open {args.video}")
        sys.exit(1)

    current_time = 0.0 if source == 0 else args.seconds
    ok, frame = grab_frame(cap, current_time) if source != 0 else cap.read()
    if not ok:
        print(f"Could not read a frame from {args.video} at {current_time}s. "
              f"Try a smaller --seconds value, or a different video.")
        sys.exit(1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, click_event)

    print(f"Window sized to fit your screen (max {MAX_DISPLAY_W}x{MAX_DISPLAY_H}).")
    print("Press 'n' if this frame isn't useful (title card, black screen, etc).")
    print("Click two points to place the fence line, then press any other key to confirm.")

    scale = 1.0
    while True:
        display, scale = fit_to_display(frame)
        for p in points:
            cv2.circle(display, tuple(p), 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(display, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # Esc
            print("Cancelled — no changes saved.")
            sys.exit(0)

        if key == ord("n") and source != 0:
            current_time += 1.0
            ok, new_frame = grab_frame(cap, current_time)
            if ok:
                frame = new_frame
                points = []
                print(f"Advanced to ~{current_time:.0f}s — click points again.")
            else:
                print("Reached end of video — staying on this frame.")
            continue

        if len(points) == 2 and key != 255:
            break

    cv2.destroyAllWindows()
    cap.release()

    # Convert clicked points from display coordinates back to the video's
    # native resolution, since main.py and fence_setup work in native pixels.
    original_points = [[round(x / scale), round(y / scale)] for x, y in points]

    try:
        with open(args.config) as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}

    config["fence"] = original_points
    config.setdefault("zone", {"name": "Demo Zone", "required_authorization_level": 3})
    config.setdefault("allowed_time_window", {"start_hour": 6, "end_hour": 18})

    with open(args.config, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved fence line {original_points} (native resolution) to {args.config}")


if __name__ == "__main__":
    main()