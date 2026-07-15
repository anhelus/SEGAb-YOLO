"""Convert image sequences to videos with codec fallback and optional rotation.

Typical usage::

    python -m scripts.process_bags --frames data/tomatoes --output videos --fps 30 --rotate 90
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from tqdm import tqdm


# Codec fallback chain: (fourcc_code, file_extension)
_CODEC_CANDIDATES: List[Tuple[str, str]] = [
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("H264", ".mp4"),
    ("XVID", ".avi"),
    ("MJPG", ".avi"),
    ("I420", ".avi"),
]

_ROTATION_MAP = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert image sequences to videos."
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="data/tomatoes",
        help="Directory containing image sequence subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for generated videos",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video frame rate (default: 30)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="*.png",
        help="Image file extension pattern (default: *.png)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotation to apply to each frame (default: 0 = no rotation)",
    )
    return parser.parse_args()


def _try_codec(
    out_path: Path, fps: int, frame_size: Tuple[int, int], codec: str
) -> Optional[cv2.VideoWriter]:
    """Try to open a VideoWriter with the given codec.

    Args:
        out_path: Full output path for the video file.
        fps: Frames per second.
        frame_size: (width, height) tuple for the video.
        codec: FourCC code string (e.g. ``"mp4v"``).

    Returns:
        An opened VideoWriter on success, or None if the codec is not
        available.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc=cv2.VideoWriter_fourcc(*codec),
        fps=fps,
        frameSize=frame_size,
    )
    if writer.isOpened():
        return writer
    writer.release()
    return None


def _open_writer(
    output_dir: Path,
    rel_path: Path,
    fps: int,
    frame_size: Tuple[int, int],
) -> Optional[cv2.VideoWriter]:
    """Open a VideoWriter by trying each codec candidate in order.

    Args:
        output_dir: Root output directory.
        rel_path: Relative path (without extension) used for the output
            filename.
        fps: Frames per second.
        frame_size: (width, height) tuple for the video.

    Returns:
        An opened VideoWriter, or None if no codec works.
    """
    for codec, ext in _CODEC_CANDIDATES:
        out_path = output_dir / f"{rel_path}{ext}"
        writer = _try_codec(out_path, fps, frame_size, codec)
        if writer is not None:
            return writer
    return None


def _read_first_frame(image_files: List[Path]) -> Tuple[Optional[cv2.Mat], int, int]:
    """Read the first image to determine frame dimensions.

    Args:
        image_files: Sorted list of image paths.

    Returns:
        A tuple ``(frame, height, width)``.  ``frame`` is None if the
        first image could not be read.
    """
    frame = cv2.imread(str(image_files[0]))
    if frame is None:
        return None, 0, 0
    h, w = frame.shape[:2]
    return frame, h, w


def process_sequence(
    video_dir: Path,
    output_dir: Path,
    frames_dir: Path,
    ext: str,
    fps: int,
    rotation: int,
) -> bool:
    """Convert a single image sequence directory into a video.

    Args:
        video_dir: Directory containing the image sequence.
        output_dir: Root directory where the video will be saved.
        frames_dir: Root directory of all frame sequences (used to
            compute the relative output path).
        ext: Glob pattern for image files (e.g. ``"*.png"``).
        fps: Output video frame rate.
        rotation: Clockwise rotation in degrees (0, 90, 180, 270).

    Returns:
        True if the video was written successfully, False otherwise.
    """
    image_files = sorted(video_dir.rglob(ext))
    if not image_files:
        return False

    _, h, w = _read_first_frame(image_files)
    if h == 0 or w == 0:
        return False

    # Apply rotation: swap dimensions if 90 or 270 degrees.
    if rotation in (90, 270):
        frame_size = (h, w)
    else:
        frame_size = (w, h)

    rel = video_dir.relative_to(frames_dir)
    writer = _open_writer(output_dir, rel, fps, frame_size)
    if writer is None:
        return False

    rot_func = _ROTATION_MAP.get(rotation)
    for image_file in tqdm(image_files, desc=f"  {video_dir.name}", leave=False):
        frame = cv2.imread(str(image_file))
        if frame is None:
            continue
        if rot_func is not None:
            frame = cv2.rotate(frame, rot_func)
        writer.write(frame)

    writer.release()
    return True


def main() -> None:
    """Entry point: parse args and process every image-sequence directory."""
    args = parse_args()

    frames_dir = Path(args.frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted({p.parent for p in frames_dir.rglob(args.ext)})
    if not video_dirs:
        print(f"No '{args.ext}' files found under {frames_dir}")
        return

    success = 0
    for video_dir in tqdm(video_dirs, desc="Videos"):
        ok = process_sequence(
            video_dir, output_dir, frames_dir, args.ext, args.fps, args.rotate
        )
        if ok:
            success += 1

    print(f"Done. {success}/{len(video_dirs)} videos created in {output_dir}")


if __name__ == "__main__":
    main()
