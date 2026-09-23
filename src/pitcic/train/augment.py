"""Geometric and photometric augmentation, shared by both training stages.

One implementation on purpose. The arms of the experiment differ in weights and
in training data; a third moving part would make the comparison unreadable.

Flips and rot90 are applied to image and label together by concatenating them
into one tensor, which is why the label rides along as a float channel and is
cast back afterwards. Colour jitter touches the image only.
"""

from __future__ import annotations

import tensorflow as tf


def augment_fn(img, label):
    img_f = tf.cast(img, tf.float32)
    combined = tf.concat(
        [img_f, tf.cast(tf.expand_dims(label, -1), tf.float32)], axis=-1
    )
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k)
    img = tf.cast(combined[:, :, :3], tf.uint8)
    label = tf.cast(combined[:, :, 3], tf.int32)

    img_f = tf.cast(img, tf.float32) / 255.0
    img_f = tf.image.random_brightness(img_f, max_delta=0.2)
    img_f = tf.image.random_contrast(img_f, lower=0.8, upper=1.2)
    img_f = tf.image.random_saturation(img_f, lower=0.8, upper=1.2)
    img_f = tf.image.random_hue(img_f, max_delta=0.05)
    img = tf.cast(tf.clip_by_value(img_f * 255.0, 0, 255), tf.uint8)
    return img, label


DESCRIPTION = "hflip+vflip+rot90+brightness+contrast+saturation+hue"
