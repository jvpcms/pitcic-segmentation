"""U-Net with an ImageNet-pretrained EfficientNetB0 encoder.

The encoder choice is a 4 GB VRAM decision: EfficientNetB0 is the largest
backbone that trains at 256 px with a batch big enough for BatchNorm to behave.
Skips are taken at the four expand activations, so the decoder sees strides
2, 4, 8 and 32, and a final upsample + conv block returns to full resolution.

Input is raw uint8. EfficientNet rescales and normalizes inside the graph, so
nothing upstream should divide by 255 -- doing it twice is a silent 100x
brightness error that still trains, just badly.
"""

from __future__ import annotations

import tensorflow as tf

from pitcic.taxonomy import N_CLASSES

INPUT_SHAPE = (256, 256, 3)
SKIP_LAYERS = [
    "block2a_expand_activation",
    "block3a_expand_activation",
    "block4a_expand_activation",
    "block6a_expand_activation",
]


def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def decoder_block(x, skip, filters, dropout):
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return tf.keras.layers.Dropout(dropout)(x)


def build_unet(dropout: float = 0.3, pretrained: bool = True):
    """Returns (model, backbone). The backbone is returned so the caller can
    freeze it for a warmup phase; it is a view on the same weights, not a copy.

    ``pretrained=False`` is the "from scratch" arm: same architecture, random
    encoder weights. That arm exists to test whether DeepGlobe pretraining is
    necessary at all, so the only thing allowed to differ is this flag.
    """
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_tensor=inputs,
    )
    skips = [backbone.get_layer(n).output for n in SKIP_LAYERS]
    x = conv_block(backbone.output, 256)
    x = decoder_block(x, skips[3], 256, dropout)
    x = decoder_block(x, skips[2], 128, dropout)
    x = decoder_block(x, skips[1], 64, dropout)
    x = decoder_block(x, skips[0], 32, dropout)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = conv_block(x, 16)
    outputs = tf.keras.layers.Conv2D(N_CLASSES, 1, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), backbone
