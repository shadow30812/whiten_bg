#!/usr/bin/env python3
"""Replace a photo's background with plain white (for passport-style photos).

Setup (once):   pip install "rembg[cpu]" pymatting opencv-python-headless numpy
Usage:          python3 whiten_bg.py photo1.jpg photo2.jpg -o out_folder

By default the photo is NOT cropped or resized: the aspect ratio is left exactly as it was
(only downscaled, same ratio, if the longest side exceeds --max-side, to keep it fast).

Options:
  --crop          auto-crop to a 35:45 passport frame around the face (white padding if the
                  frame extends past the photo). Needs a single, roughly front-facing face.
  --size WxH      with --crop, also resize the result (e.g. 630x810). Default: keep native size.
  --brighten      lift a dark/underexposed face
  --max-side N    cap on the longest side before processing (default 2400, 0 = no cap)

Originals are never modified; results are saved as <name>_white.jpg in the output folder.
The first run downloads a ~170 MB segmentation model.
"""

import argparse
import os

import cv2
import numpy as np
from rembg import new_session, remove

RATIO = 35 / 45  # width / height


def auto_crop(im):
    """Crop to a 35:45 frame with the head about 75% of the height, padding with white."""
    h, w = im.shape[:2]
    k = min(1.0, 3000 / max(h, w))
    g = cv2.cvtColor(
        cv2.resize(im, None, fx=k, fy=k) if k < 1 else im, cv2.COLOR_BGR2GRAY
    )
    fc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    fs = fc.detectMultiScale(g, 1.1, 6, minSize=(40, 40))
    if len(fs) == 0:
        raise SystemExit(
            "No face found - run without --crop on a photo you've already framed."
        )
    x, y, fw, fh = [v / k for v in max(fs, key=lambda f: f[2])]
    if fh * 1.8 < 600:
        print(
            f"  note: face is small, so the cropped photo will be only ~{int(fh * 1.8)}px tall (low resolution)"
        )
    ch = fh * 1.8
    cw = ch * RATIO
    cx, cy = x + fw / 2, y + fh / 2 - 0.02 * fh
    x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
    x1, y1 = int(round(x0 + cw)), int(round(y0 + ch))
    p = max(0, -x0, -y0, x1 - w, y1 - h)
    if p:
        im = cv2.copyMakeBorder(
            im, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    return im[y0 + p : y1 + p, x0 + p : x1 + p]


def whiten(path, session, brighten=False, crop=False, size=None, max_side=2400):
    im = cv2.imread(path)  # honours EXIF orientation
    if im is None:
        raise SystemExit(f"Cannot read {path}")
    if crop:
        im = auto_crop(im)
    if max_side and max(im.shape[:2]) > max_side:
        s = max_side / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgba = remove(
        cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=8,
    )
    a = rgba[..., 3].astype(np.float32) / 255
    a = cv2.GaussianBlur(cv2.erode(a, np.ones((3, 3))), (0, 0), 0.8)[
        ..., None
    ]  # trim halo, soften edge
    fg = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
    if brighten:
        lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
        lab[..., 0] = np.clip(255 * (lab[..., 0] / 255.0) ** 0.55, 0, 255).astype(
            np.uint8
        )
        lab[..., 0] = cv2.createCLAHE(1.5, (4, 4)).apply(lab[..., 0])
        fg = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = (fg.astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)
    if crop and size:
        out = cv2.resize(
            out,
            size,
            interpolation=cv2.INTER_AREA if out.shape[0] > size[1] else cv2.INTER_CUBIC,
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default="white_bg_out")
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--size", default=None)
    ap.add_argument("--brighten", action="store_true")
    ap.add_argument("--max-side", type=int, default=2400)
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    os.makedirs(a.out, exist_ok=True)
    ses = new_session("u2net_human_seg")
    for p in a.images:
        dst = os.path.join(
            a.out, os.path.splitext(os.path.basename(p))[0] + "_white.jpg"
        )
        cv2.imwrite(
            dst,
            whiten(p, ses, a.brighten, a.crop, size, a.max_side),
            [cv2.IMWRITE_JPEG_QUALITY, 97],
        )
        print("saved", dst)
