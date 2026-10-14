"""Inspect a rendered video without watching it: report its properties and extract frames.

    uv run python scenes/studio/inspect_video.py media/videos/x/480p15/MyVideo.mp4
    uv run python scenes/studio/inspect_video.py <video> --times 1.5 4 9
    uv run python scenes/studio/inspect_video.py <video> --per-beat 3     # 3 frames inside every beat

With no options it uses the <video>.beats.json written by BeatScene (start, middle and end of each beat).
Output goes to media/inspect/<video name>/ : one PNG per frame plus contact_sheet.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw

THUMB_WIDTH = 640  # contact-sheet cell width in pixels
COLUMNS = 3


def video_info(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(container.duration) / av.time_base if container.duration else None
        return {
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "fps": float(stream.average_rate) if stream.average_rate else None,
            "duration_s": duration,
            "frames": stream.frames,
            "codec": stream.codec_context.name,
        }


def frame_at(container: av.container.InputContainer, seconds: float) -> Image.Image:
    stream = container.streams.video[0]
    container.seek(int(seconds / stream.time_base), stream=stream, backward=True)
    last = None
    for frame in container.decode(stream):
        last = frame
        if frame.time is not None and frame.time >= seconds - 1e-3:
            break
    assert last is not None, f"no frame near {seconds}s"
    return last.to_image()


def plan_times(video: Path, times: list[float] | None, per_beat: int, margin: float) -> list[tuple[str, float]]:
    if times:
        return [(f"t={t:g}s", t) for t in times]
    sidecar = video.with_name(video.stem + ".beats.json")
    beats = json.loads(sidecar.read_text(encoding="utf-8"))["beats"]
    out: list[tuple[str, float]] = []
    for b in beats:
        if b["skipped"]:
            continue  # not in this video
        # Stay `margin` seconds inside the beat: its first and last frames are usually fades.
        first, last = b["start"] + margin, max(b["start"] + margin, b["end"] - margin)
        for i in range(per_beat):
            f = i / (per_beat - 1) if per_beat > 1 else 0.5
            out.append((f"{b['name']} {i + 1}/{per_beat}", first + f * (last - first)))
    return out


def contact_sheet(images: list[tuple[str, Image.Image]]) -> Image.Image:
    cell_h = int(THUMB_WIDTH * images[0][1].height / images[0][1].width)
    label_h = 24
    rows = (len(images) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGB", (COLUMNS * THUMB_WIDTH, rows * (cell_h + label_h)), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(images):
        x, y = (i % COLUMNS) * THUMB_WIDTH, (i // COLUMNS) * (cell_h + label_h)
        sheet.paste(img.resize((THUMB_WIDTH, cell_h)), (x, y + label_h))
        draw.text((x + 6, y + 6), label, fill=(230, 230, 235))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--times", type=float, nargs="*", help="seconds to extract")
    parser.add_argument("--per-beat", type=int, default=3, help="frames per beat when using beats.json")
    parser.add_argument("--margin", type=float, default=0.5, help="seconds to stay inside each beat's edges")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    info = video_info(args.video)
    print(json.dumps(info, indent=1))

    out = args.out or Path("media/inspect") / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    images: list[tuple[str, Image.Image]] = []
    with av.open(str(args.video)) as container:
        for label, t in plan_times(args.video, args.times, args.per_beat, args.margin):
            img = frame_at(container, t)
            name = f"{len(images):02d}_{label.replace(' ', '_').replace('/', 'of').replace('=', '')}.png"
            img.save(out / name)
            images.append((f"{label}  ({t:.2f}s)", img))
    contact_sheet(images).save(out / "contact_sheet.png")
    print(f"wrote {len(images)} frames and {out / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
