import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SeaDroneSee realtime tracking demo on frame sequence.")
    parser.add_argument("--model", type=Path, default=Path("SeaDroneSee/best.pt"))
    parser.add_argument("--frame-dir", type=Path, default=Path("SeaDroneSee/data/test"))
    parser.add_argument("--start", type=int, default=6701)
    parser.add_argument("--end", type=int, default=6881)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def frame_list(frame_dir: Path, start: int, end: int) -> list[Path]:
    frames = []
    for i in range(start, end + 1):
        p = frame_dir / f"{i}.jpg"
        if p.exists():
            frames.append(p)
    return frames


def draw_tracks(image: np.ndarray, result, names: dict | list) -> np.ndarray:
    out = image.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
    cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(xyxy), -1)

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        color = (0, 220, 0) if cls[i] == 1 else (255, 140, 0)
        if isinstance(names, dict):
            cls_name = names.get(int(cls[i]), str(int(cls[i])))
        else:
            cls_name = str(names[int(cls[i])]) if int(cls[i]) < len(names) else str(int(cls[i]))
        label = f"ID {ids[i]} | {cls_name} | {conf[i]:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    args = parse_args()
    frames = frame_list(args.frame_dir, args.start, args.end)
    if not frames:
        raise RuntimeError(f"No frames found in {args.frame_dir} for range {args.start}-{args.end}.")

    model = YOLO(str(args.model))
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read frame: {frames[0]}")

    for frame in frames:
        result = model.track(
            source=str(frame),
            tracker="bytetrack.yaml",
            persist=True,
            stream=False,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]

        drawn = draw_tracks(result.orig_img, result, model.names)
        cv2.imshow("SeaDroneSee SAR Tracking", drawn)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print(f"Processed frames: {len(frames)} ({frames[0].name} -> {frames[-1].name})")


if __name__ == "__main__":
    main()
