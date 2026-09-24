#!/usr/bin/env python3
"""Render the image / ground truth / prediction panel used in the README.

    python scripts/make_qualitative_figure.py --tile <tile_id>

Predicts one annotated 1024 px tile through the same path the protocol uses:
the tile is cut into the 4x4 grid of 256 px sub-tiles, each is predicted, and
the sub-tiles are reassembled. Predicting the 1024 px tile in one pass would be
a different computation from the one every reported number comes from.

The tile is passed explicitly rather than chosen by the script. Picking the
best-scoring tile automatically is how a qualitative figure ends up flattering
the model; the caption in the README states what this tile actually scores and
how that compares to the split as a whole.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from pitcic.data import pampa
from pitcic.eval import protocol
from pitcic.taxonomy import CLASS_NAMES, CLASS_TO_COLOR, UNKNOWN_IDX

DEFAULT_TILE = "CBERS_4A_WPM_20220729_212_150_L4_43_44"


PANEL_TITLES = {
    "en": ["CBERS-4A composite", "annotation", "prediction"],
    "pt": ["composição CBERS-4A", "anotação de referência", "predição"],
}


def colorize(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for idx, color in enumerate(CLASS_TO_COLOR):
        out[labels == idx] = color
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", default=DEFAULT_TILE)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: docs/img/qualitative.png inside the repository")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--lang", choices=("en", "pt"), default="en",
                    help="language of the panel titles")
    ap.add_argument("--no-title", action="store_true",
                    help="omit the figure title, for use with a caption that "
                         "already carries the tile id and the metrics")
    args = ap.parse_args()

    from pitcic import config

    out = args.out or config.REPO_ROOT / "docs" / "img" / "qualitative.png"
    model_path = args.model or config.require(
        "checkpoints.finetuned", "run: python scripts/fetch/checkpoints.py"
    )
    triples = [t for t in pampa.annotated_tiles("test") if t[0] == args.tile]
    if not triples:
        raise SystemExit(f"{args.tile} is not an annotated test tile")
    tiles = protocol.load_tiles(triples)

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise SystemExit(
            f"model outputs {model.output_shape[-1]} classes, expected "
            f"{len(CLASS_NAMES)}"
        )
    preds = np.empty(tiles.images.shape[:3], dtype=np.int32)
    for start in range(0, len(tiles), 4):
        chunk = tiles.images[start:start + 4].astype(np.float32)
        preds[start:start + len(chunk)] = np.argmax(
            model(chunk, training=False).numpy(), axis=-1
        )

    # Reassemble in the order protocol.cut produced: r outer, c inner. The
    # shape comes from the parent mask rather than from sqrt(len(tiles)), which
    # would be wrong for any tile that is not square.
    from PIL import Image
    h, w = np.array(Image.open(triples[0][2])).shape[:2]
    n_rows, n_cols = h // protocol.TILE, w // protocol.TILE

    def grid(flat):
        rows = [np.concatenate(list(flat[r * n_cols:(r + 1) * n_cols]), axis=1)
                for r in range(n_rows)]
        return np.concatenate(rows, axis=0)

    image = grid(tiles.images)
    truth = grid(tiles.labels)
    pred = grid(preds)

    # The panel shows the whole tile, so the figures printed here cover every
    # sub-tile of it. The protocol drops sub-tiles at 5% Unknown; when it would
    # drop any of this tile's, say so rather than quietly reporting a number
    # that is not the protocol's.
    dropped = int((~tiles.keep_mask(0.05)).sum())
    if dropped:
        print(f"note: {dropped}/{len(tiles)} sub-tiles of this tile are at or "
              "above the 5% Unknown threshold and would be dropped by the "
              "protocol. The figures below cover the whole tile and are "
              "therefore not the protocol's.")

    valid = truth != UNKNOWN_IDX
    if not valid.any():
        raise SystemExit(f"{args.tile} is entirely Unknown, nothing to score")
    m = protocol.metrics(protocol.confusion(truth[valid], pred[valid]))
    acc = m["pixel_accuracy"]

    # The legend covers what either panel shows. Filtering on ground truth
    # alone would leave a class the model predicted, and that the reader can
    # see, with no entry naming it.
    shown = [
        n for n in CLASS_NAMES
        if m["gt_pixels"][n] > 0 or int((pred == CLASS_NAMES.index(n)).sum()) > 0
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, data, title in zip(
        axes,
        [image, colorize(truth), colorize(pred)],
        PANEL_TITLES[args.lang],
    ):
        ax.imshow(data)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for side_ in ax.spines.values():
            side_.set_edgecolor("#999999")

    handles = [
        Patch(facecolor=np.array(CLASS_TO_COLOR[CLASS_NAMES.index(n)]) / 255,
              edgecolor="#555555", label=n.replace("_", " + "))
        for n in shown
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    n_scored = len([n for n in CLASS_NAMES if m["gt_pixels"][n] > 0])
    if args.no_title:
        fig.tight_layout(rect=(0, 0.07, 1, 1.0))
    else:
        fig.suptitle(
            f"{args.tile}   {w * 2} x {h * 2} m at 2 m/px   "
            f"this tile: pixel accuracy {acc:.1%}, "
            f"mIoU {m['miou']:.1%} over its {n_scored} classes",
            fontsize=10, y=0.97,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}  acc {acc:.4f}  mIoU {m['miou']:.4f}")
    for n in shown:
        if m["gt_pixels"][n] == 0:
            print(f"  {n:<22} predicted but absent from this tile's annotation")
            continue
        print(f"  {n:<22} IoU {m['iou'][n]:.4f}  "
              f"{m['gt_pixels'][n] / valid.sum():6.1%} of scored px")


if __name__ == "__main__":
    main()
