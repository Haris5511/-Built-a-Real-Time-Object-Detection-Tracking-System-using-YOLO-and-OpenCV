
import argparse
import csv
import time
from datetime import datetime
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11 Object Detection and Tracking"
    )

    parser.add_argument(
        "--source",
        default="video.mp4",
        help="Video path, or 0 for webcam"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated video as output.mp4"
    )

    args = parser.parse_args()

    if not 0.0 <= args.conf <= 1.0:
        print("ERROR: Confidence must be between 0 and 1.")
        return

    # 1. Load YOLO model
    print("Loading YOLO11 nano model...")

    try:
        model = YOLO("yolo11n.pt")
    except Exception as error:
        print(f"ERROR loading YOLO model: {error}")
        return

    # 2. Open webcam or video
    source = 0 if args.source == "0" else args.source

    if isinstance(source, str) and not Path(source).is_file():
        print(f"ERROR: Video file not found: {source}")
        print("Place video.mp4 in the same folder as app.py.")
        return

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam/video.")
        return

    # Ask webcam for a moderate capture resolution
    if source == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps <= 0 or video_fps > 120:
        video_fps = 25.0

    print(f"Input: {args.source}")
    print(f"Resolution: {width}x{height}")
    print(f"Video FPS: {video_fps:.1f}")

    # 3. Create a small, resizable display window
    window_name = "YOLO11 Object Detection and Tracking"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    # 4. Optional annotated video output
    video_writer = None

    if args.save:
        if width <= 0 or height <= 0:
            print("ERROR: Invalid video dimensions.")
            cap.release()
            cv2.destroyAllWindows()
            return

        video_writer = cv2.VideoWriter(
            "output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height)
        )

        if not video_writer.isOpened():
            print("ERROR: Could not create output.mp4")
            video_writer.release()
            video_writer = None
            cap.release()
            cv2.destroyAllWindows()
            return

    # 5. Tracking statistics
    seen_tracks = set()
    unique_class_counts = Counter()
    frame_number = 0
    start_time = time.time()
    previous_time = time.perf_counter()
    fps = 0.0

    # 6. CSV logging
    try:
        with open(
            "tracking_log.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "timestamp",
                "frame_number",
                "tracking_id",
                "object_class",
                "confidence"
            ])

            print("\nTracking started. Press Q to quit.")

            while True:
                success, frame = cap.read()

                if not success:
                    print("Video finished or frame could not be read.")
                    break

                frame_number += 1

                # 7. Object detection and ByteTrack tracking
                result = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=args.conf,
                    imgsz=640,
                    verbose=False
                )[0]

                output_frame = result.plot()
                boxes = result.boxes
                current_class_counts = Counter()

                if boxes is not None and len(boxes) > 0:
                    class_ids = boxes.cls.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()

                    if boxes.id is not None:
                        track_ids = boxes.id.int().cpu().tolist()
                    else:
                        track_ids = [None] * len(class_ids)

                    timestamp = datetime.now().isoformat(
                        timespec="seconds"
                    )

                    for class_id, confidence, track_id in zip(
                        class_ids, confidences, track_ids
                    ):
                        class_name = model.names[int(class_id)]
                        current_class_counts[class_name] += 1

                        # Count unique tracked objects
                        if track_id is not None:
                            track_key = (
                                int(class_id),
                                int(track_id)
                            )

                            if track_key not in seen_tracks:
                                seen_tracks.add(track_key)
                                unique_class_counts[class_name] += 1

                        # Save detection details to CSV
                        csv_writer.writerow([
                            timestamp,
                            frame_number,
                            track_id if track_id is not None else "",
                            class_name,
                            round(float(confidence), 4)
                        ])

                # 8. Calculate processing FPS
                now = time.perf_counter()
                elapsed = now - previous_time

                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = (
                        instant_fps if frame_number == 1
                        else 0.8 * fps + 0.2 * instant_fps
                    )

                previous_time = now

                current_count = (
                    len(boxes) if boxes is not None else 0
                )

                cv2.putText(
                    output_frame,
                    f"FPS: {fps:.1f} | Current objects: {current_count}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    output_frame,
                    f"Unique tracked IDs: {len(seen_tracks)}",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                # 9. Show class-wise object counts
                y_position = 95

                for class_name, count in sorted(
                    current_class_counts.items()
                ):
                    cv2.putText(
                        output_frame,
                        f"{class_name}: {count}",
                        (12, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    y_position += 25

                # 10. Save full-resolution annotated video
                if video_writer is not None:
                    video_writer.write(output_frame)

                # 11. Resize ONLY the displayed frame
                display_frame = cv2.resize(
                    output_frame,
                    (640, 480),
                    interpolation=cv2.INTER_AREA
                )

                cv2.imshow(window_name, display_frame)

                if frame_number % 30 == 0:
                    csv_file.flush()

                # Press Q to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

    except KeyboardInterrupt:
        print("\nTracking interrupted by user.")

    except Exception as error:
        print(f"\nERROR during tracking: {error}")

    finally:
        cap.release()

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()

    # 12. Final report
    print("\nTracking finished.")
    print(f"Frames processed: {frame_number}")
    print(f"Unique tracked IDs: {len(seen_tracks)}")
    print(
        f"Processing time: {time.time() - start_time:.1f} seconds"
    )

    print("\nUnique tracked objects by class:")

    for class_name, count in sorted(unique_class_counts.items()):
        print(f"  {class_name}: {count}")

    print("\nCSV log: tracking_log.csv")

    if args.save:
        print("Annotated video: output.mp4")


if __name__ == "__main__":
    main()
