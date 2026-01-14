from collections.abc import Sequence
from enum import auto, StrEnum


# Format: left, top, width, height
# All the coordinates are ratios in [0.0, 1.0]
BBox = tuple[float, float, float, float]

class DatasetType(StrEnum):
    TRAINING = auto()
    VALIDATION = auto()
    TEST = auto()

class Label(StrEnum):
    NON_FIRE = auto()
    FIRE = auto()

    def to_int(self) -> int:
        for i, e in enumerate(__class__): # type: ignore[name-defined]
            if e.value == self:
                return i
        raise LookupError


FeatureVector = Sequence[float]
Sample = tuple[Sequence[FeatureVector], Label]
