"""The evaluation protocol. Every reported number goes through this module.

There is deliberately one implementation. During the work the same rules lived
in three places -- an evaluation script, the sweep driver and a notebook -- and
the three drifted. Anything that scores a model imports from here.

The rules, in order:

1. Annotated tiles are 1024 px. They are cut into a non-overlapping 4x4 grid of
   256 px sub-tiles. No overlap, no padding: an input whose side is not a
   multiple of 256 is an error rather than a silently dropped strip.
2. The CVAT mask is decoded through ``CVAT_COLOR_TO_CLASS``, merging
   Agriculture and Rangeland and sending unannotated background to Unknown.
3. A sub-tile is dropped entirely once its Unknown fraction reaches the
   threshold, because what is left is noise-dominated. Reported at 5%.
4. Unknown ground-truth pixels are excluded from scoring in the sub-tiles that
   survive. Predicting Unknown on a labeled pixel still counts as an error, so
   the model cannot buy accuracy by abstaining.
5. mIoU is the mean IoU over the classes present in the ground truth, computed
   from the confusion matrix. This is not Keras ``MeanIoU``: that averages over
   all 6 slots including Unknown, which is excluded here, so the two numbers
   are not comparable and only this one is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pitcic.taxonomy import (
    CLASS_NAMES,
    CVAT_COLOR_TO_CLASS,
    N_CLASSES,
    UNKNOWN_IDX,
)

TILE = 256


def rgb_mask_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 -> (H,W) int32 class index.

    Anything not in the palette stays Unknown rather than raising: CVAT exports
    occasionally carry a stray antialiased pixel on a polygon border.
    """
    labels = np.full(mask_rgb.shape[:2], UNKNOWN_IDX, dtype=np.int32)
    for color, idx in CVAT_COLOR_TO_CLASS.items():
        labels[np.all(mask_rgb == color, axis=-1)] = idx
    return labels


@dataclass
class TileSet:
    """Sub-tiles of one split, flattened, with their parent and Unknown share."""

    images: np.ndarray   # (N, 256, 256, 3) uint8
    labels: np.ndarray   # (N, 256, 256) int32
    parents: np.ndarray  # (N,) str, the 1024 px tile each came from
    unknown: np.ndarray  # (N,) float, Unknown fraction of the sub-tile

    def __len__(self) -> int:
        return len(self.images)

    def keep_mask(self, threshold: float) -> np.ndarray:
        """Sub-tiles strictly below the Unknown threshold."""
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold is a fraction in (0, 1], got {threshold}")
        return self.unknown < threshold


def cut(image: np.ndarray, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """One 1024 px tile -> its 4x4 grid of 256 px sub-tiles."""
    if image.shape[:2] != labels.shape:
        raise ValueError(f"image {image.shape[:2]} != mask {labels.shape}")
    h, w = labels.shape
    if h % TILE or w % TILE:
        raise ValueError(
            f"{h}x{w} is not a multiple of {TILE}; the right/bottom strip "
            "would be dropped without notice"
        )
    out = []
    for r in range(0, h - TILE + 1, TILE):
        for c in range(0, w - TILE + 1, TILE):
            out.append((image[r:r + TILE, c:c + TILE], labels[r:r + TILE, c:c + TILE]))
    return out


def load_tiles(pairs: list[tuple[str, Path, Path]]) -> TileSet:
    """Build a TileSet from (tile_id, image_png, mask_png) triples."""
    imgs, lbls, parents, unk = [], [], [], []
    for tid, img_path, mask_path in pairs:
        img = np.array(Image.open(img_path).convert("RGB"))
        lbl = rgb_mask_to_labels(np.array(Image.open(mask_path).convert("RGB")))
        for sub_img, sub_lbl in cut(img, lbl):
            imgs.append(sub_img)
            lbls.append(sub_lbl)
            parents.append(tid)
            unk.append(float((sub_lbl == UNKNOWN_IDX).mean()))
    if not imgs:
        raise ValueError("no tiles to load")
    return TileSet(
        images=np.stack(imgs),
        labels=np.stack(lbls),
        parents=np.asarray(parents),
        unknown=np.asarray(unk),
    )


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """(N_CLASSES, N_CLASSES) counts, rows = ground truth."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true.ravel(), y_pred.ravel()), 1)
    return cm


def metrics(cm: np.ndarray) -> dict:
    """Pixel accuracy, per-class IoU and mIoU over classes present in GT."""
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    present = cm.sum(1) > 0
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return {
        "pixel_accuracy": float(np.trace(cm) / cm.sum()),
        "miou": float(np.nanmean(np.where(present, iou, np.nan))),
        "iou": {
            name: (float(iou[i]) if present[i] else None)
            for i, name in enumerate(CLASS_NAMES)
        },
        "gt_pixels": {n: int(cm.sum(1)[i]) for i, n in enumerate(CLASS_NAMES)},
        "pred_pixels": {n: int(cm.sum(0)[i]) for i, n in enumerate(CLASS_NAMES)},
        "confusion": cm.tolist(),
    }


def score(tiles: TileSet, preds: np.ndarray, threshold: float = 0.05) -> dict:
    """Apply the protocol to a full set of predictions.

    ``preds`` is (N, 256, 256) int32, aligned with ``tiles.images``.
    """
    if preds.shape != tiles.labels.shape:
        raise ValueError(f"preds {preds.shape} != labels {tiles.labels.shape}")
    keep = tiles.keep_mask(threshold)
    if not keep.any():
        raise ValueError(
            f"every sub-tile is at or above the {threshold:.0%} Unknown "
            "threshold -- nothing to score"
        )
    lab, prd = tiles.labels[keep], preds[keep]
    valid = lab != UNKNOWN_IDX
    out = metrics(confusion(lab[valid], prd[valid]))
    out["unknown_threshold"] = threshold
    out["subtiles_kept"] = int(keep.sum())
    out["subtiles_total"] = int(len(keep))
    out["scored_pixel_fraction"] = float(valid.mean())
    return out


def format_report(m: dict) -> str:
    """The console table. Same layout the experiment logs carry."""
    lines = [
        f"kept {m['subtiles_kept']}/{m['subtiles_total']} sub-tiles at "
        f"{m['unknown_threshold']:.0%} Unknown",
        f"scored {m['scored_pixel_fraction']:.1%} of their pixels",
        f"pixel accuracy {m['pixel_accuracy']:.4f}",
        f"mIoU          {m['miou']:.4f}",
        "",
        f"{'class':<22}{'IoU':>9}{'GT px':>14}{'pred px':>14}",
    ]
    for name in CLASS_NAMES:
        iou = m["iou"][name]
        iou_s = f"{iou:.4f}" if iou is not None else "n/a"
        lines.append(
            f"{name:<22}{iou_s:>9}{m['gt_pixels'][name]:>14,}"
            f"{m['pred_pixels'][name]:>14,}"
        )
    cm = np.asarray(m["confusion"], dtype=np.float64)
    row = cm.sum(1, keepdims=True)
    rec = np.where(row > 0, cm / np.maximum(row, 1), np.nan)
    lines += ["", "row-normalized confusion (recall per GT class)",
              " " * 22 + "".join(f"{n[:10]:>11}" for n in CLASS_NAMES)]
    for i, name in enumerate(CLASS_NAMES):
        cells = "".join(
            "        n/a" if np.isnan(v) else f"{v:>11.2f}" for v in rec[i]
        )
        lines.append(f"{name:<22}{cells}")
    return "\n".join(lines)
