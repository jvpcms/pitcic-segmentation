"""Reproduction code for the PITCIC land cover segmentation module.

Two stages. ``pitcic.train.deepglobe`` trains a U-Net on DeepGlobe resampled to
2 m/px, and ``pitcic.train.finetune`` adapts it to CBERS-4A imagery over the
Pampa biome. ``pitcic.eval.protocol`` is the single implementation of the
evaluation rules and is what every reported number goes through.
"""

__version__ = "1.0.0"
