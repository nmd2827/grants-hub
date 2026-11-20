"""Symbol classes shared by the training data, the model and the browser converter.

The order of CLASSES is the model's class index order. Do not reorder without retraining,
and keep pdf-to-flat/index.html's HW_CLASSES list identical.
"""

CLASSES = [
    "noteheadFull", "noteheadHalf", "noteheadWhole", "noteheadGrace",
    "stem", "beam", "flag8th", "flag16th", "augmentationDot", "ledgerLine",
    "restWhole", "restHalf", "restQuarter", "rest8th", "rest16th", "multiRest",
    "sharp", "flat", "natural", "doubleSharp",
    "gClef", "fClef", "cClef", "percClef",
    "digit0", "digit1", "digit2", "digit3", "digit4", "digit5", "digit6", "digit7", "digit8", "digit9",
    "timeSigCommon", "timeSigCut",
    "barline", "repeatDot", "tie", "slur", "crescendo", "diminuendo",
    "dynP", "dynF", "dynM", "dynS", "dynZ",
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# MUSCIMA++ v2 class name -> our class (anything not listed is ignored)
MUSCIMA_MAP = {
    "noteheadFull": "noteheadFull", "noteheadHalf": "noteheadHalf", "noteheadWhole": "noteheadWhole",
    "noteheadFullSmall": "noteheadGrace",
    "stem": "stem", "beam": "beam",
    "flag8thUp": "flag8th", "flag8thDown": "flag8th", "flag16thUp": "flag16th", "flag16thDown": "flag16th",
    "augmentationDot": "augmentationDot", "legerLine": "ledgerLine",
    "restWhole": "restWhole", "restHalf": "restHalf", "restQuarter": "restQuarter",
    "rest8th": "rest8th", "rest16th": "rest16th", "multiMeasureRest": "multiRest",
    "accidentalSharp": "sharp", "accidentalFlat": "flat", "accidentalNatural": "natural",
    "accidentalDoubleSharp": "doubleSharp",
    "gClef": "gClef", "fClef": "fClef", "cClef": "cClef",
    **{f"numeral{d}": f"digit{d}" for d in range(10)},
    "timeSigCommon": "timeSigCommon", "timeSigCutCommon": "timeSigCut",
    "barline": "barline", "barlineHeavy": "barline", "repeatDot": "repeatDot",
    "tie": "tie", "slur": "slur",
    "dynamicCrescendoHairpin": "crescendo", "dynamicDiminuendoHairpin": "diminuendo",
    "dynamicLetterP": "dynP", "dynamicLetterF": "dynF", "dynamicLetterM": "dynM",
    "dynamicLetterS": "dynS", "dynamicLetterZ": "dynZ",
}

# Geometry used everywhere: staff space normalised to this many pixels, strips padded to this size.
SPACE_PX = 22
TILE_W, TILE_H = 1024, 320
TILE_STRIDE = 896


def coco_categories():
    return [{"id": i, "name": c, "supercategory": "music"} for i, c in enumerate(CLASSES)]
