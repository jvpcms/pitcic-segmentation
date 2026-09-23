#!/usr/bin/env python3
"""Stage 1: train the base U-Net on DeepGlobe at 2 m/px.

    python -m pitcic.train.deepglobe
    python -m pitcic.train.deepglobe --epochs 100 --batch-size 16 --lr 5e-5

Defaults come from ``conf/deepglobe.yaml`` and are the settings of the run that
produced the released ``deepglobe_unet_base.keras``. Reproducing that checkpoint
takes many hours on a 4 GB card; if you only want the fine-tuning results,
fetch the weights instead of retraining them.

``--freeze-epochs`` runs a decoder-only warmup with the encoder frozen and then
recompiles at a reduced rate. It is off by default: measured on this dataset it
bought nothing, and it is kept only because turning it back on is a one-flag
experiment.

Checkpoint selection runs on Keras ``val_miou``, which is not the reported
metric. See ``pitcic.models.losses`` for why the two differ.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from pitcic import config
from pitcic.data import deepglobe as data
from pitcic.taxonomy import N_CLASSES


def make_datasets(batch_size: int):
    import tensorflow as tf

    from pitcic.train.augment import augment_fn

    train_pairs = data.load_pairs("train")
    val_pairs = data.load_pairs("val")
    print(f"train {len(train_pairs)} tiles | val {len(val_pairs)} tiles")

    def _load(sat_path, mask_path):
        img = np.load(sat_path.numpy().decode()).astype(np.uint8)
        # Merge Agriculture and Rangeland here, not on disk.
        label = data.LABEL_REMAP[np.load(mask_path.numpy().decode())]
        return img, label

    def tf_load(sat_path, mask_path):
        img, label = tf.py_function(_load, [sat_path, mask_path], [tf.uint8, tf.int32])
        img.set_shape([data.TILE_SIZE, data.TILE_SIZE, 3])
        label.set_shape([data.TILE_SIZE, data.TILE_SIZE])
        return img, label

    def build(pairs, augment):
        sat_p, mask_p = map(list, zip(*pairs))
        ds = tf.data.Dataset.from_tensor_slices((sat_p, mask_p))
        if augment:
            ds = ds.shuffle(len(sat_p), seed=data.SEED, reshuffle_each_iteration=True)
        ds = ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return build(train_pairs, True), build(val_pairs, False)


def parse_args(argv=None):
    cfg = config.load("deepglobe")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    ap.add_argument("--lr", type=float, default=cfg["lr"])
    ap.add_argument("--lr-factor", type=float, default=cfg["lr_factor"])
    ap.add_argument("--lr-patience", type=int, default=cfg["lr_patience"])
    ap.add_argument("--es-patience", type=int, default=cfg["es_patience"])
    ap.add_argument("--dropout", type=float, default=cfg["dropout"])
    ap.add_argument("--freeze-epochs", type=int, default=cfg["freeze_epochs"],
                    help="decoder-only warmup epochs before unfreezing (0 = off)")
    ap.add_argument("--unfreeze-lr-div", type=float, default=4.0,
                    help="phase 2 runs at lr / this, to protect encoder features")
    ap.add_argument("--out", type=Path,
                    default=config.resolve("checkpoints.root") / "deepglobe")
    return ap.parse_args(argv), cfg


def main(argv=None) -> None:
    args, cfg = parse_args(argv)
    import tensorflow as tf

    from pitcic.models.losses import bce_dice_loss, training_metrics
    from pitcic.models.unet import build_unet
    from pitcic.train.augment import DESCRIPTION as AUG

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("no GPU visible; this is not meant to run on CPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(cfg["seed"])

    args.out.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = make_datasets(args.batch_size)
    model, backbone = build_unet(args.dropout)

    def compile_at(lr):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=bce_dice_loss,
            metrics=training_metrics(),
        )

    compile_at(args.lr)
    print(f"params {sum(int(tf.size(w)) for w in model.weights):,}")

    (args.out / "config.json").write_text(json.dumps({
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "lr_factor": args.lr_factor, "lr_patience": args.lr_patience,
        "es_patience": args.es_patience, "decoder_dropout": args.dropout,
        "encoder": cfg["encoder"], "n_classes": N_CLASSES, "loss": "bce_dice",
        "optimizer": "adam", "augmentation": AUG, "seed": cfg["seed"],
        "tiling": "3x3 overlapping 256px grid at 2 m/px",
        "class_merge": "Agriculture+Rangeland merged at load time (7 -> 6)",
        "freeze_epochs": args.freeze_epochs,
    }, indent=2))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(args.out / "best.keras"), monitor="val_miou", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_miou", mode="max", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_miou", mode="max", patience=args.es_patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(args.out / "history.csv")),
    ]

    started = time.time()
    if args.freeze_epochs > 0:
        # Phase 1: BN layers run in inference mode while the backbone is frozen.
        backbone.trainable = False
        compile_at(args.lr)
        print(f"phase 1: encoder frozen for {args.freeze_epochs} epochs")
        model.fit(train_ds, validation_data=val_ds, epochs=args.freeze_epochs)
        # Recompile is required after a trainable change.
        backbone.trainable = True
        compile_at(args.lr / args.unfreeze_lr_div)
        print(f"phase 2: unfrozen at lr={args.lr / args.unfreeze_lr_div:g}")

    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        initial_epoch=args.freeze_epochs, callbacks=callbacks,
    )
    model.save(args.out / "final.keras")
    print(f"wrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
