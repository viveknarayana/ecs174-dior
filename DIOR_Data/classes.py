"""Shared DIOR label names and XML-name normalization."""

import re


# Label IDs are index + 1; ID 0 is reserved for background.
CLASS_NAMES = [
    "airplane",
    "airport",
    "baseball field",
    "basketball court",
    "bridge",
    "chimney",
    "dam",
    "expressway service area",
    "expressway toll station",
    "golf course",
    "ground track field",
    "harbour",
    "overpass",
    "ship",
    "stadium",
    "storage tank",
    "tennis court",
    "train station",
    "vehicle",
    "windmill",
]


def normalize_class_name(name):
    """Normalize XML labels to the shared names used by the project."""
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    aliases = {
        "harbor": "harbour",
        # DIOR's XML uses "golffield" for the golf-course category.
        "golffield": "golfcourse",
    }
    return aliases.get(normalized, normalized)


CLASS_TO_ID = {
    normalize_class_name(name): index + 1
    for index, name in enumerate(CLASS_NAMES)
}
