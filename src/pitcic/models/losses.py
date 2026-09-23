"""Loss and the in-training metrics.

Cross-entropy alone under-weights the rare classes, and this dataset is very
unbalanced -- Barren is under 2% of labeled pixels. Dice is computed per class
and averaged, so a class contributes to the loss in proportion to nothing but
its own overlap, and the sum of the two is what every run trains on.

A warning about ``miou`` in the training logs: Keras ``MeanIoU`` averages over
all six slots, Unknown included, and scores every pixel. The protocol in
``pitcic.eval.protocol`` excludes Unknown ground truth and drops noise-dominated
sub-tiles. The two numbers are not comparable, and only the protocol one is
reported. ``val_miou`` is here to pick checkpoints, not to be quoted.
"""

from __future__ import annotations

import tensorflow as tf

from pitcic.taxonomy import CLASS_NAMES, N_CLASSES


def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), N_CLASSES)
    axes = [1, 2]
    intersection = tf.reduce_sum(y_true_oh * y_pred, axis=axes)
    union = tf.reduce_sum(y_true_oh, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def training_metrics() -> list:
    per_class = [
        tf.keras.metrics.IoU(
            num_classes=N_CLASSES,
            target_class_ids=[i],
            name=f"iou_{CLASS_NAMES[i].lower()}",
            sparse_y_pred=False,
        )
        for i in range(N_CLASSES)
    ]
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.MeanIoU(num_classes=N_CLASSES, name="miou", sparse_y_pred=False),
        *per_class,
    ]
