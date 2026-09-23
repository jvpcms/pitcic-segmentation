# PITCIC — land cover segmentation for terrain traversability

Reproduction code for the segmentation stage of **PITCIC** (*Inteligência
Artificial aplicada ao Processo de Integração Terreno, Condições Meteorológicas,
Inimigo e Considerações Civis*), a final-year project at the Instituto Militar
de Engenharia.

The pipeline the project sits in is:

```
satellite image → semantic segmentation → cost heuristic → movement difficulty heatmap
```

**This repository covers the first arrow only.** It produces a per-pixel land
cover raster in six classes. The heuristic and the heatmap are not built here:
they already exist, in
[`lcoandrade/costmap_learning`](https://github.com/lcoandrade/costmap_learning),
and the output of this model is designed to drop into them as the land cover
layer. That interface is described in [docs/interface.md](docs/interface.md).

## What it does

A U-Net with an ImageNet-pretrained EfficientNetB0 encoder is trained on
DeepGlobe resampled to 2 m/px, then adapted to CBERS-4A WPM imagery over the
Brazilian Pampa using 100 hand-annotated tiles.

| | pixel accuracy | mIoU |
|---|---|---|
| DeepGlobe model, zero-shot on CBERS-4A | 42,1% | 23,7% |
| after fine-tuning (lr 3e-4) | 94,8% | 82,0% |

Both under the protocol in `src/pitcic/eval/protocol.py`, at the 5% Unknown
threshold, on the 20-tile test split.

The headline result of the sweep is a negative one and the code is arranged so
it can be checked: a model trained on the same 80 tiles from ImageNet weights
alone, with no DeepGlobe contact, reaches 80,4% mIoU. The gap to fine-tuning is
about 1,5 points, which is close to the run-to-run spread. The evidence does not
support the claim that DeepGlobe pretraining is necessary.

## Reproducing the numbers without a GPU-week

```sh
make setup      # venv + install
make data       # the annotated dataset, from Zenodo
make weights    # the two released checkpoints
make eval       # score the fine-tuned model under the protocol
```

`make eval` runs on CPU if it has to, slowly. Everything it needs is fetched;
nothing in this repository is data.

To re-run the experiment instead of checking it:

```sh
make sweep       # 9 runs: zero-shot, 5 fine-tune LRs, 3 from-scratch LRs
make sweep-half  # the same arms on the first 40 train tiles
make results     # fold runs/*/metrics.json into results/*.csv
```

One run at a time, always. There is one card, and two jobs on it fail in ways
that look like bad hyperparameters.

## Data

Nothing is committed. `data/` and `artifacts/` are gitignored and populated by
the scripts in `scripts/fetch/`.

| what | where from | licence |
|---|---|---|
| 100 annotated CBERS-4A tiles | [doi:10.5281/zenodo.22922625](https://doi.org/10.5281/zenodo.22922625) | CC BY-SA 4.0 |
| DeepGlobe Land Cover | Kaggle, needs your own credentials | its own terms, no redistribution |
| trained checkpoints | this repository's GitHub release | CC BY-SA 4.0 |

The annotated dataset is published separately, with its own README, annotation
guidelines and known-issues list:
[jvpcms/pampa-cbers-landcover](https://github.com/jvpcms/pampa-cbers-landcover).
The version DOI pinned in `conf/paths.yaml` is the exact data every number here
was computed from; the concept DOI above is the one to cite.

DeepGlobe cannot be mirrored, so `scripts/fetch/deepglobe.py` downloads it from
the source under the terms you accept there.

## Layout

```
conf/            YAML: paths, hyperparameters, the sweep grid, the tile ranks
scripts/fetch/   every download lives here
scripts/         run_sweep.py, collect_results.py
src/pitcic/
  taxonomy.py    the 6 classes and the CVAT palette, in one place
  config.py      path resolution, with PITCIC_* environment overrides
  data/          deepglobe.py (tiling), pampa.py (the annotated tiles)
  models/        unet.py, losses.py
  train/         deepglobe.py (stage 1), finetune.py (stage 2), augment.py
  eval/          protocol.py, cli.py
results/         committed CSVs: the sweep, and IoU per class
notebooks/       figure generation only, outputs stripped
docs/            the evaluation protocol, and the downstream interface
```

`src/pitcic/eval/protocol.py` is the single implementation of the evaluation
rules. During the work those rules lived in three places and the three drifted;
anything that scores a model imports from there now.

## The evaluation protocol, in short

Annotated tiles are 1024 px, cut into a non-overlapping 4×4 grid of 256 px
sub-tiles. A sub-tile is dropped when its Unknown fraction reaches 5%. In the
ones that survive, Unknown ground-truth pixels are excluded from scoring, but
*predicting* Unknown on a labeled pixel still counts as an error. mIoU is the
mean over the classes present in the ground truth.

Unknown is trained as an ordinary class — the network must be able to say
"cloud" — and excluded only when scoring, because the label means two different
things (unknowable ground, and merely unannotated background) and cannot be
scored as one. Full statement in [docs/protocol.md](docs/protocol.md).

Keras `val_miou` in the training logs is a different quantity: it averages over
all six classes and scores every sub-tile. It selects checkpoints. It is never
reported.

## Caveats worth reading before quoting a number

- **The test split doubles as the validation set.** Every figure is a validation
  score, never held out. It is equally true of all arms, which is what keeps
  them comparable, but the absolute values are optimistic.
- **One biome, one sensor, one season.** The Pampa, CBERS-4A WPM, a narrow
  acquisition window. Nothing here measures generalisation beyond that.
- **Barren is weak** (IoU 0,50 at best) and rare, under 2% of labeled pixels.
- **Forest→Agriculture confusion matters more than its IoU suggests**: those two
  classes are three ranks apart on the traversability scale, so the class that
  looks healthy at 86% IoU contributes about 42% of the downstream rank error,
  while Barren contributes about 2%. See [docs/interface.md](docs/interface.md).

## Hardware

Everything was run on an RTX 3050 Laptop GPU, 4 GB. That is why the fine-tuning
batch size is 8 and the evaluation batch size is 4, and why runs are separate
processes. `conf/*.yaml` says "do not raise" where raising it will fail.

## Citing

The dataset and the code are separate artifacts. Cite the dataset by its concept
DOI above. For the code, `CITATION.cff` in this repository.

## Licence

Code under the MIT licence (`LICENSE`). The released checkpoints and the
annotated dataset are CC BY-SA 4.0, inherited from INPE's CBERS-4A imagery.
