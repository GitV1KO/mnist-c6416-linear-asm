#!/usr/bin/env python3
"""Decode the project's binary pixel text into PNG images (standard library)."""
import argparse
import csv
from itertools import islice
from pathlib import Path
import struct
import zlib


def png_bytes(width, height, pixels):
    """Write an 8-bit grayscale PNG; pixels are in row-major order."""
    if len(pixels) != width * height:
        raise ValueError("incorrect pixel count")
    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff))
    rows = b"".join(b"\0" + pixels[y*width:(y+1)*width] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


FONT = {
    "0": (14,17,19,21,25,17,14), "1": (4,12,4,4,4,4,14),
    "2": (14,17,1,2,4,8,31), "3": (30,1,1,14,1,1,30),
    "4": (2,6,10,18,31,2,2), "5": (31,16,16,30,1,1,30),
    "6": (14,16,16,30,17,17,14), "7": (31,1,2,4,8,8,8),
    "8": (14,17,17,14,17,17,14), "9": (14,17,17,15,1,1,14),
    ":": (0,4,4,0,4,4,0),
}


def contact_sheet(images, labels, offset):
    """Nearest-neighbor enlargement preserves the actual binary pixels."""
    shown = min(len(images), 100)
    columns = min(shown, 5)
    cell_w, cell_h, margin, scale = 168, 188, 16, 5
    width = columns * cell_w + margin
    height = ((shown + columns - 1) // columns) * cell_h + margin
    canvas = bytearray([35]) * (width * height)
    for index in range(shown):
        x0 = margin + (index % columns) * cell_w
        y0 = margin + (index // columns) * cell_h
        # Caption syntax: sample index : true label.
        text = f"{index + offset}:{labels[index]}"
        for position, char in enumerate(text):
            for row, mask in enumerate(FONT[char]):
                for col in range(5):
                    if mask & (1 << (4-col)):
                        for yy in range(2):
                            for xx in range(2):
                                canvas[(y0+row*2+yy)*width+x0+position*12+col*2+xx] = 240
        for row in range(28):
            line = bytes(value*255 for value in images[index][row*28:(row+1)*28]
                         for _ in range(scale))
            for repeat in range(scale):
                start = (y0+24+row*scale+repeat)*width+x0
                canvas[start:start+len(line)] = line
    return png_bytes(width, height, bytes(canvas))


def read_range(path, start, stop):
    with path.open(encoding="ascii") as stream:
        return [int(line.strip()) for line in islice(stream, start, stop)]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "test_images")
    args = parser.parse_args()
    if args.count <= 0 or args.offset < 0:
        parser.error("count must be positive; offset must be nonnegative")
    stop = args.offset + args.count
    values = read_range(args.data_dir / "data.txt", args.offset*784, stop*784)
    labels = read_range(args.data_dir / "label.txt", args.offset, stop)
    if len(values) != args.count*784 or len(labels) != args.count:
        raise ValueError("not enough image/label data")
    if any(v not in (0,1) for v in values) or any(v not in range(10) for v in labels):
        raise ValueError("invalid binary pixels or class labels")
    images = [values[i*784:(i+1)*784] for i in range(args.count)]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "true_label", "png_file"])
        for i, image in enumerate(images):
            name = f"sample_{args.offset+i:05d}_label_{labels[i]}.png"
            (args.output / name).write_bytes(png_bytes(28, 28, bytes(v*255 for v in image)))
            writer.writerow([args.offset+i, labels[i], name])
    (args.output / "preview.png").write_bytes(contact_sheet(images, labels, args.offset))
    print(f"Exported {args.count} original-size 28x28 PNGs to {args.output}")
    print("preview.png captions: sample_index:true_label (not model predictions).")


if __name__ == "__main__":
    main()
