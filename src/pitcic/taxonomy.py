"""The 6-class taxonomy, in one place.

DeepGlobe ships 7 classes. Agriculture and Rangeland are merged into a single
class here because the CBERS-4A imagery at 2 m/px does not separate them
reliably and, downstream, they land within one rank of each other on the
traversability scale. The merge happens at load time on both datasets, so no
stored mask on disk is ever rewritten.

Unknown is index 5 and is ambiguous by construction: it covers genuinely
unknowable ground (cloud, shadow) and merely unannotated background. It is
trained as an ordinary class -- the network must be able to say "cloud" -- but
Unknown ground-truth pixels are excluded from scoring, because a label that
means two different things cannot be scored as one.
"""

from __future__ import annotations

N_CLASSES = 6
UNKNOWN_IDX = 5
CLASS_NAMES = [
    "Urban",
    "Agriculture_Rangeland",
    "Forest",
    "Water",
    "Barren",
    "Unknown",
]

# CVAT "Segmentation mask 1.1" export colors -> merged 6-class index.
# The export keeps the original 7-class DeepGlobe palette, so Agriculture and
# Rangeland are two distinct colors mapping to the same index.
# background (0,0,0) == Unknown: unannotated pixels are treated as Unknown.
CVAT_COLOR_TO_CLASS = {
    (0, 255, 255): 0,    # Urban
    (255, 255, 0): 1,    # Agriculture
    (255, 0, 255): 1,    # Rangeland (merged)
    (0, 255, 0): 2,      # Forest
    (0, 0, 255): 3,      # Water
    (255, 255, 255): 4,  # Barren
    (0, 0, 0): 5,        # Unknown / background
}

# Same palette, for rendering a predicted label map back to RGB.
CLASS_TO_COLOR = [
    (0, 255, 255),
    (255, 255, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
]
