"""Train a YOLO or RTDETR model with configurable IOU type and dataset fraction.

Typical usage::

    python -m scripts.run -m yolo26n -i ciou -d data/lettuce/dataset.yaml -o runs/my_experiment -f 0.3
"""

import argparse
from pathlib import Path

from segab_yolo import YOLO, RTDETR


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO or RTDETR model."
    )
    parser.add_argument("-m", "--model", required=True, help="Model name (e.g. yolo26n)")
    # parser.add_argument(
    #     "-i", "--iou", default="ciou", help="IOU type (default: ciou)"
    # )
    parser.add_argument("-d", "--datapath", required=True, help="Dataset YAML path")
    parser.add_argument(
        "-o", "--outpath", required=True, help="Output project subdirectory"
    )
    parser.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=0.3,
        help="Dataset fraction to use (default: 0.3)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and launch training."""
    args = parse_args()

    if args.model.startswith("yolo"):
        model = YOLO(f"{args.model}.yaml")
    else:
        model = RTDETR(f"{args.model}.yaml")

    model.train(
        data=Path(args.datapath),
        project=Path("runs", args.outpath),
        name=args.model.split(".")[0],
        epochs=100,
        # imgsz=960,
        # iou_type=args.iou,
        fraction=args.fraction,
    )


if __name__ == "__main__":
    main()
