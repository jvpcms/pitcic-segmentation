#!/usr/bin/env python3
"""Score a checkpoint on the annotated CBERS-4A tiles.

    python -m pitcic.eval.cli                                   # released fine-tuned weights
    python -m pitcic.eval.cli --model artifacts/checkpoints/deepglobe_unet_base.keras
    python -m pitcic.eval.cli --split train --threshold 0.05 0.25

Reports at every requested threshold. The default reports both the strict 5%
used everywhere in the report and the 25% the protocol keeps available: on this
dataset a cloudy parent tile is all-or-nothing at 5%, so quoting a single
threshold would quietly turn the figure into a clear-sky-only baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from pitcic import config
from pitcic.data import pampa
from pitcic.eval import protocol
from pitcic.taxonomy import N_CLASSES


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", type=Path, default=None,
                    help="default: the released fine-tuned checkpoint")
    ap.add_argument("--split", choices=("test", "train", "all"), default="test")
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.05, 0.25])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--json", type=Path, help="also write the metrics here")
    args = ap.parse_args(argv)

    model_path = args.model or config.require(
        "checkpoints.finetuned",
        "run: python scripts/fetch/checkpoints.py --only finetuned",
    )
    triples = pampa.annotated_tiles(args.split)
    tiles = protocol.load_tiles(triples)
    print(f"model {model_path}")
    print(f"data  {len(triples)} tiles -> {len(tiles)} sub-tiles ({args.split})")

    import tensorflow as tf  # imported late: keeps --help fast

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.output_shape[-1] != N_CLASSES:
        raise SystemExit(
            f"model outputs {model.output_shape[-1]} classes, expected {N_CLASSES}"
        )

    # Raw uint8 in: EfficientNetB0 rescales and normalizes inside the graph.
    preds = np.empty(tiles.images.shape[:3], dtype=np.int32)
    for start in range(0, len(tiles), args.batch_size):
        chunk = tiles.images[start:start + args.batch_size].astype(np.float32)
        preds[start:start + len(chunk)] = np.argmax(
            model(chunk, training=False).numpy(), axis=-1
        )
        print(f"\r  predicting {min(start + len(chunk), len(tiles))}/{len(tiles)}",
              end="", flush=True)
    print()

    out = {}
    for t in args.threshold:
        m = protocol.score(tiles, preds, t)
        out[f"unk{int(t * 100):02d}"] = m
        print(f"\n{'=' * 72}")
        print(protocol.format_report(m))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
