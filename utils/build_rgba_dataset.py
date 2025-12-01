#!/usr/bin/env python3
"""Utility to build RGBA (mask + edge) control tensors from raw images and masks."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PREVIEW_COLUMNS = 3  # original, mask, edge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute RGBA control tensors by fusing masks and Canny edges.")
    parser.add_argument("--img_dir", required=True, type=Path, help="Directory containing source images.")
    parser.add_argument("--mask_dir", required=True, type=Path, help="Directory containing mask images (grayscale labels).")
    parser.add_argument(
        "--dest_dir",
        required=True,
        type=Path,
        help="Base directory for RGBA tensors. The script will create a <canny_low>_<canny_high> subfolder automatically.",
    )
    parser.add_argument(
        "--fmt",
        nargs="+",
        choices=["npz", "png"],
        default=["npz"],
        help="Output format(s) to generate. Specify both to keep compressed tensors and 4ch PNG copies.",
    )
    parser.add_argument(
        "--preview_dir",
        type=Path,
        default=None,
        help="Optional directory to store preview panels (original/mask/edge). Defaults to <dest_dir>/preview.",
    )
    parser.add_argument("--preview-max", type=int, default=32, help="Maximum number of preview panels to save (per run).")
    parser.add_argument("--num-mask-classes", type=int, default=3, help="Number of mask classes to one-hot encode into RGB.")
    parser.add_argument("--canny-low", type=int, default=100, help="Lower hysteresis threshold for Canny.")
    parser.add_argument("--canny-high", type=int, default=200, help="Upper hysteresis threshold for Canny.")
    parser.add_argument("--beta-edge", type=float, default=1.0, help="Multiplier applied to the edge channel before normalization.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing RGBA outputs if they already exist.")
    return parser.parse_args()


def collect_file_map(root: Path, allowed_exts: Iterable[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in allowed_exts:
            continue
        stem = file_path.stem
        if stem in mapping:
            # Keep the first occurrence but warn later when mismatches happen
            continue
        mapping[stem] = file_path
    return mapping


def ensure_hw_match(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if image.shape[:2] == mask.shape[:2]:
        return image, mask
    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image, resized_mask


def mask_to_rgb(mask: np.ndarray, num_classes: int) -> np.ndarray:
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(min(num_classes, 3)):
        rgb[:, :, class_idx] = (mask == class_idx).astype(np.uint8) * 255
    return rgb


def normalize_mask(mask_rgb: np.ndarray) -> np.ndarray:
    return (mask_rgb.astype(np.float32) / 127.5) - 1.0


def normalize_edge(edge_map: np.ndarray, beta_edge: float) -> np.ndarray:
    edge_float = edge_map.astype(np.float32) / 255.0
    if beta_edge != 1.0:
        edge_float = np.clip(edge_float * beta_edge, 0.0, 1.0)
    return edge_float * 2.0 - 1.0


def save_npz(path: Path, rgba: np.ndarray, meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rgba=rgba.astype(np.float32), meta=json.dumps(meta))


def save_png(path: Path, rgba: np.ndarray) -> None:
    rgba_uint8 = np.clip((rgba + 1.0) * 127.5, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgba_uint8, cv2.COLOR_RGBA2BGRA))


def build_preview(original_bgr: np.ndarray, mask_rgb: np.ndarray, edge_gray: np.ndarray) -> np.ndarray:
    mask_preview = mask_rgb
    edge_preview = cv2.cvtColor(edge_gray, cv2.COLOR_GRAY2BGR)
    resized_mask = cv2.resize(mask_preview, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    resized_edge = cv2.resize(edge_preview, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([original_bgr, resized_mask, resized_edge], axis=1)


def main() -> None:
    args = parse_args()
    img_dir = args.img_dir.resolve()
    mask_dir = args.mask_dir.resolve()
    dest_dir = args.dest_dir.resolve()
    dest_suffix = f"{args.canny_low}_{args.canny_high}"
    if dest_dir.name != dest_suffix:
        dest_dir = dest_dir / dest_suffix
    print(f"[INFO] Saving outputs to {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = args.preview_dir.resolve() if args.preview_dir else dest_dir / "preview"
    if args.preview_max > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_map = collect_file_map(img_dir, SUPPORTED_IMAGE_EXTS)
    mask_map = collect_file_map(mask_dir, SUPPORTED_IMAGE_EXTS)

    total = len(image_map)
    processed = 0
    missing_masks = []
    previews_saved = 0

    for stem, image_path in sorted(image_map.items()):
        mask_path = mask_map.get(stem)
        if mask_path is None:
            missing_masks.append(stem)
            continue

        rgba_npz_path = dest_dir / f"{stem}.npz"
        rgba_png_path = dest_dir / f"{stem}.png"
        if not args.overwrite:
            outputs_exist = True
            if "npz" in args.fmt:
                outputs_exist &= rgba_npz_path.exists()
            if "png" in args.fmt:
                outputs_exist &= rgba_png_path.exists()
            if outputs_exist:
                processed += 1
                continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"[WARN] Failed to read pair ({image_path}, {mask_path}). Skipping.")
            continue

        image, mask = ensure_hw_match(image, mask)
        mask_rgb = mask_to_rgb(mask, args.num_mask_classes)
        mask_norm = normalize_mask(mask_rgb)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_raw = cv2.Canny(image_gray, args.canny_low, args.canny_high)
        edge_norm = normalize_edge(edge_raw, args.beta_edge)

        rgba = np.concatenate([mask_norm, edge_norm[..., None]], axis=-1)
        meta = {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "value_range": "neg_one_to_one",
            "canny_low": args.canny_low,
            "canny_high": args.canny_high,
            "beta_edge": args.beta_edge,
        }

        if "npz" in args.fmt:
            save_npz(rgba_npz_path, rgba, meta)
        if "png" in args.fmt:
            save_png(rgba_png_path, rgba)

        if args.preview_max > 0 and previews_saved < args.preview_max:
            preview = build_preview(image, mask_rgb, edge_raw)
            preview_path = preview_dir / f"{stem}.png"
            cv2.imwrite(str(preview_path), preview)
            previews_saved += 1

        processed += 1

    print(f"Processed {processed}/{total} inputs. Missing masks: {len(missing_masks)}")
    if missing_masks:
        missing_log = dest_dir / "missing_masks.txt"
        with open(missing_log, "w", encoding="utf-8") as f:
            for stem in missing_masks:
                f.write(stem + "\n")
        print(f"Wrote missing mask list to {missing_log}")

if __name__ == "__main__":
    main()
