# The downstream interface

This repository stops at a land cover raster. What consumes it is the
traversability estimator in
[`lcoandrade/costmap_learning`](https://github.com/lcoandrade/costmap_learning),
described in the operational model report for the south of Brazil. The estimator
was **not** run in this work. This document states the contract the model output
is built to satisfy, and what the segmentation error is worth once it passes
through that contract.

## The contract

The estimator combines several terrain layers into a traversability index by
weighted mean: slope, pedology, topographic indices, weather, vehicle type and
land cover. Land cover enters with a weight of **0,2411**.

Each land cover class maps to a traversability **rank on a 0–9 scale**, and the
rank depends on the manoeuvre element profile (road tyre, mixed tyre,
all-terrain, tracks, on foot). The mapping is a lookup table, not a learned
component, so plugging this model in means producing a class raster on the same
grid and supplying a lookup entry per class.

Two adjustments are needed for the six classes here:

- **The merged Agriculture+Rangeland class has no single rank.** The published
  table ranks `vegetacao_baixa` and `veg_cultivada` separately, and the merge
  spans both. Every downstream figure below is therefore given as an interval
  over the two hypotheses rather than a point value.
- **Unknown has no rank.** It should be treated as missing data by the
  estimator, not as a traversable or impassable class.

After the weighted mean, the estimator applies vector masks for hydrography and
transport that zero the corresponding areas outright. That happens *after* the
land cover layer contributes, which matters for reading the error below.

## What the segmentation error is worth downstream

Taking the best run's confusion matrix on the test split, and for each
(true, predicted) pair the absolute difference between the ranks the estimator
assigns to the two classes, averaged over pixels:

| profile | rank error | interval from the merge | contribution to the index |
|---|---|---|---|
| road tyre | 0,231 | 0,151 – 0,231 | 0,056 |
| mixed tyre | 0,279 | 0,199 – 0,279 | 0,067 |
| all-terrain | 0,301 | 0,194 – 0,301 | 0,073 |
| tracks | 0,261 | 0,208 – 0,261 | 0,063 |
| on foot | 0,248 | 0,147 – 0,248 | 0,060 |

About a quarter of a rank on a scale of nine, and under 0,08 of a rank in the
final index once the 0,2411 weight is applied.

Three things are worth reading off this, because none of them is visible in the
IoU table.

**Barren is the worst class and nearly harmless.** It is the weakest class by a
wide margin in every run, and contributes about 2% of the rank error: 37,8% of
its true pixels go to the merged class, which is one rank away on the vehicle
profiles and zero on tracks.

**The dominant term is Forest confused with the merged class.** It reaches 11,1%
of forest pixels and accounts on its own for about 42% of the rank error on the
all-terrain profile, because forest and low vegetation are seven ranks apart.
That is the expensive error, and `Forest` looks healthy at 84–87% IoU.

**The impassable classes are partly covered by external data.** In the best run,
3,56% of Water and Urban pixels are predicted as a traversable class and 0,48%
go the other way. The practical effect is smaller than it looks, because the
hydrography and transport vector masks zero those areas after the weighted mean.
The corollary is where this model actually adds: forest, low vegetation and bare
ground, the classes no vector mask covers, and also the ones where going from
30 m to 2 m resolution changes anything.

## Caveat

Every figure here is computed from the confusion matrix and the published rank
tables. Nothing was measured end to end against the estimator, which was not
executed in this work.
