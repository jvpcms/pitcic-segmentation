# The evaluation protocol

Every number reported for this model goes through
`src/pitcic/eval/protocol.py`. This document states the rules the code
implements and why each one exists.

The protocol was fixed before any fine-tuning result was read.

## 1. Tiling

Annotated tiles are 1024 px, georeferenced, at 2 m/px. They are cut into a
non-overlapping 4×4 grid of 256 px sub-tiles: 16 per tile, 1600 in total across
the 100 annotated tiles.

No overlap and no padding. An input whose side is not a multiple of 256 raises
rather than silently dropping the right and bottom strip.

The split is assigned per 1024 px tile, so sub-tiles of one tile never straddle
the train/test boundary.

## 2. Decoding the masks

The CVAT export keeps the original 7-colour DeepGlobe palette. It is decoded
through `CVAT_COLOR_TO_CLASS` in `src/pitcic/taxonomy.py`, which merges
Agriculture and Rangeland into one index and sends unannotated background to
Unknown. A pixel whose colour is not in the palette stays Unknown: CVAT
occasionally leaves a stray antialiased pixel on a polygon border.

The merge happens at load time. No stored mask is ever rewritten, so changing
the taxonomy never means re-annotating or re-tiling.

## 3. Dropping noise-dominated sub-tiles

A sub-tile is dropped entirely once its Unknown fraction reaches the threshold.
Reported at **5%**.

The threshold is also computed at 25% and both are kept in every `metrics.json`.
This is not hedging: on this dataset a cloudy parent tile is all-or-nothing at
5%, so quoting a single strict threshold would quietly turn the figure into a
clear-sky-only baseline. Carrying both makes that visible.

The cutoff is an **evaluation** rule. Training sees every sub-tile.

## 4. Excluding Unknown ground truth

In the sub-tiles that survive, pixels whose ground truth is Unknown are excluded
from the confusion matrix.

Unknown carries two meanings here: genuinely unknowable ground (cloud, shadow)
and merely unannotated background. It is not a reliable label either way, and a
label that means two things cannot be scored as one.

Two consequences the code is careful about:

- Unknown is **trained as an ordinary class**. No loss mask, no sample weights.
  The network must be able to say "cloud"; otherwise it is forced to assign
  clouds to a land cover class.
- Predicting Unknown on a *labeled* pixel is still an error and is counted as
  one. The model cannot buy accuracy by abstaining.

## 5. The metric

From the confusion matrix over the scored pixels:

- **pixel accuracy** = trace / total
- **IoU per class** = `TP / (TP + FP + FN)`
- **mIoU** = mean IoU over the classes *present in the ground truth*

A class absent from the ground truth of a split reports `null` rather than 0,
and does not enter the mean.

## What `val_miou` is not

The training logs carry a `val_miou` from Keras `MeanIoU`. It averages over all
six slots including Unknown, and scores every sub-tile with no threshold. It is
a different quantity from the protocol mIoU and the two are not comparable.

`val_miou` selects checkpoints. It is never reported.

## What this protocol does not establish

The test split doubles as the validation set. Early stopping and checkpoint
selection both read it, so every reported figure is a validation score and never
a held-out one. This is true of all arms equally, which is what keeps them
comparable to each other, but it means the absolute values are optimistic and
should not be read as an estimate of performance on unseen imagery.

With 100 annotated tiles, a three-way split would have left a test set too small
to separate the arms at all. The choice was made knowingly.
