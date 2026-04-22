import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO pose model to TensorRT engine")
    parser.add_argument("--model", default="yolov8s-pose.pt", help="Input .pt pose model")
    parser.add_argument("--imgsz", type=int, default=416, help="Export image size")
    parser.add_argument("--device", default="0", help="CUDA device index")
    parser.add_argument("--half", type=int, choices=[0, 1], default=1, help="Use FP16 export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    engine_path = model.export(
        format="engine",
        half=bool(args.half),
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"[TRT] Export complete: {engine_path}")


if __name__ == "__main__":
    main()
