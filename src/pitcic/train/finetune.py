#!/usr/bin/env python3
"""Run one arm of the CBERS-4A adaptation experiment.

    python -m pitcic.train.finetune --arm zeroshot
    python -m pitcic.train.finetune --arm finetune --lr 3e-4
    python -m pitcic.train.finetune --arm scratch  --lr 1e-4 --train-subset 40

Three arms. Identical data, protocol and seed; only the weights and the amount
of training data differ.

``zeroshot``
    The DeepGlobe checkpoint, scored on ``test`` without any training. The floor
    the other two must beat.
``scratch``
    EfficientNetB0 with ImageNet weights and a fresh decoder, trained only on the
    Pampa tiles. Measures what the annotation buys with no DeepGlobe contact.
``finetune``
    The DeepGlobe checkpoint, trained on the same tiles. The proposed setup.

Things that are easy to get wrong, and that this file is careful about:

* Unknown (index 5) is an ORDINARY TRAINED CLASS. No loss mask, no sample
  weights. It is excluded only when scoring.
* The 5% Unknown cutoff is an EVALUATION rule. Training sees every sub-tile.
* ``val_miou`` (Keras MeanIoU, six classes, every sub-tile) and the protocol
  mIoU are different quantities. Model selection runs on the first; only the
  second is ever reported.
* ``test`` doubles as the validation set, so every number is a validation score
  and never a held-out one. That is equally true of all three arms, which is
  what keeps them comparable, but it does mean the absolute figures are optimistic.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from pitcic import config
from pitcic.data import pampa
from pitcic.eval import protocol
from pitcic.taxonomy import CLASS_NAMES, N_CLASSES

SEED = 42


def load_split(split: str, keep: set[str] | None = None):
    """(n_tiles, images, labels) for one split, cut to 256 px sub-tiles."""
    triples = pampa.annotated_tiles(split)
    if keep is not None:
        have = {t for t, _, _ in triples}
        missing = keep - have
        if missing:
            raise ValueError(f"{len(missing)} requested tiles are not in {split}")
        triples = [t for t in triples if t[0] in keep]
    tiles = protocol.load_tiles(triples)
    return len(triples), tiles


def build_arm(arm: str, dropout: float):
    import tensorflow as tf

    from pitcic.models.unet import build_unet

    if arm == "scratch":
        model, _ = build_unet(dropout)
    else:
        base = config.require(
            "checkpoints.base",
            "the DeepGlobe base model is not redistributed (see "
            "scripts/fetch/checkpoints.py). Build it with: make base",
        )
        model = tf.keras.models.load_model(base, compile=False)
    if model.output_shape[-1] != N_CLASSES:
        raise ValueError(
            f"model outputs {model.output_shape[-1]} classes, expected {N_CLASSES}"
        )
    return model


def compile_model(model, lr: float) -> None:
    import tensorflow as tf

    from pitcic.models.losses import bce_dice_loss, training_metrics

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=bce_dice_loss,
        metrics=training_metrics(),
    )


def predict_labels(model, images: np.ndarray, batch_size: int) -> np.ndarray:
    """argmax predictions, in chunks.

    ``model.predict`` concatenates every (256,256,6) softmax map before
    returning, which is the largest allocation in the run and exhausts a 4 GB
    card well before the model itself does.
    """
    preds = np.empty(images.shape[:3], dtype=np.int32)
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size].astype(np.float32)
        preds[start:start + len(chunk)] = np.argmax(
            model(chunk, training=False).numpy(), axis=-1
        )
    return preds


def parse_args(argv=None):
    cfg = config.load("finetune")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True, choices=("zeroshot", "scratch", "finetune"))
    ap.add_argument("--lr", type=float, default=cfg["lr"])
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument(
        "--batch-size", type=int, default=cfg["batch_size"],
        help="4 GB card: do not raise",
    )
    ap.add_argument("--es-patience", type=int, default=cfg["es_patience"])
    ap.add_argument("--lr-patience", type=int, default=cfg["lr_patience"])
    ap.add_argument("--lr-factor", type=float, default=cfg["lr_factor"])
    ap.add_argument("--dropout", type=float, default=cfg["dropout"])
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    ap.add_argument(
        "--train-subset", type=int, metavar="N", default=cfg["train_subset"],
        help="train on the first N annotated tiles by train_rank instead of all of "
             "them, to measure how the arms separate at a smaller annotation "
             "budget. The test split is never reduced, so runs stay comparable.",
    )
    ap.add_argument(
        "--eval-batch-size", type=int, default=cfg["eval_batch_size"],
        help="smaller than --batch-size: cuDNN picks a conv algorithm by "
             "profiling, and on a 4 GB card every candidate can fail on "
             "workspace allocation",
    )
    ap.add_argument(
        "--eval-only", metavar="CHECKPOINT",
        help="skip training and score this checkpoint instead. Exists because "
             "evaluating in the process that just trained can exhaust the card: "
             "the optimizer slots and the training graph are still resident and "
             "cuDNN has no workspace left to profile into. Intermittent, not "
             "deterministic; when a run dies at the eval step, re-score its "
             "best.keras with this in a fresh process.",
    )
    ap.add_argument("--runs-root", type=Path, default=config.resolve("runs_root"))
    return ap.parse_args(argv), cfg


def main(argv=None) -> None:
    args, cfg = parse_args(argv)
    import tensorflow as tf

    name = args.arm if args.arm == "zeroshot" else f"{args.arm}-lr{args.lr:g}"
    if args.train_subset:
        name += f"-tr{args.train_subset}"
    if args.tag:
        name += f"-{args.tag}"
    out = args.runs_root / name
    out.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("no GPU visible; this is not meant to run on CPU")

    keep = pampa.train_rank_prefix(args.train_subset) if args.train_subset else None
    n_train_tiles, train_tiles = load_split("train", keep)
    n_val_tiles, val_tiles = load_split("test")
    x_train, y_train = train_tiles.images, train_tiles.labels
    x_val, y_val = val_tiles.images, val_tiles.labels

    print(f"== {name}")
    print(f"train {n_train_tiles} tiles -> {len(x_train)} sub-tiles | "
          f"test {n_val_tiles} tiles -> {len(x_val)} sub-tiles")

    if args.eval_only:
        model = tf.keras.models.load_model(args.eval_only, compile=False)
        if model.output_shape[-1] != N_CLASSES:
            raise ValueError(
                f"model outputs {model.output_shape[-1]} classes, expected {N_CLASSES}"
            )
        print(f"eval-only: {args.eval_only}")
    else:
        model = build_arm(args.arm, args.dropout)
        compile_model(model, args.lr)

    record = {
        "arm": args.arm,
        "name": name,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs_max": args.epochs,
        "seed": SEED,
        "n_train_tiles": n_train_tiles,
        "train_subset": args.train_subset,
        "train_selection": (
            "first N by train_rank" if args.train_subset else "all annotated"
        ),
        "n_val_tiles": n_val_tiles,
        "n_train_subtiles": int(len(x_train)),
        "n_val_subtiles": int(len(x_val)),
        "encoder_init": "imagenet" if args.arm == "scratch" else "deepglobe",
        "unknown_policy": "trained as an ordinary class; excluded only when scoring",
        "tiling": "4x4 non-overlapping 256px grid",
    }

    if args.arm != "zeroshot" and not args.eval_only:
        from pitcic.train.augment import augment_fn

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(len(x_train), seed=SEED, reshuffle_each_iteration=True)
            .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((x_val, y_val))
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(out / "best.keras"), monitor="val_miou", mode="max",
                save_best_only=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_miou", mode="max", factor=args.lr_factor,
                patience=args.lr_patience, min_lr=1e-7, verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_miou", mode="max", patience=args.es_patience,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
        ]
        started = time.time()
        hist = model.fit(
            train_ds, validation_data=val_ds, epochs=args.epochs,
            callbacks=callbacks, verbose=2,
        )
        record["epochs_run"] = len(hist.history["loss"])
        record["train_seconds"] = round(time.time() - started, 1)
        best = int(np.argmax(hist.history["val_miou"]))
        record["best_epoch"] = best + 1
        record["keras_val_miou"] = float(hist.history["val_miou"][best])
        record["keras_val_accuracy"] = float(hist.history["val_accuracy"][best])

    preds = predict_labels(model, x_val, args.eval_batch_size)
    record["protocol"] = {
        f"unk{int(t * 100):02d}": protocol.score(val_tiles, preds, t)
        for t in cfg["protocol"]["thresholds"]
    }

    metrics_path = out / "metrics.json"
    if args.eval_only and metrics_path.exists():
        # Keep the training fields from the run that produced the checkpoint;
        # only the protocol block is being recomputed.
        previous = json.loads(metrics_path.read_text())
        previous["protocol"] = record["protocol"]
        record = previous
    record["eval_checkpoint"] = str(args.eval_only) if args.eval_only else "in-process"
    metrics_path.write_text(json.dumps(record, indent=2))

    for key, sc in record["protocol"].items():
        print(f"\n-- {key}: {sc['subtiles_kept']}/{sc['subtiles_total']} sub-tiles")
        print(f"   px acc {sc['pixel_accuracy']:.4f}   mIoU {sc['miou']:.4f}")
        for cls in CLASS_NAMES:
            v = sc["iou"][cls]
            print(f"     {cls:<22} {'absent' if v is None else f'{v:.4f}'}")
    print(f"\nwrote {metrics_path}")


if __name__ == "__main__":
    main()
