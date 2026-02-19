"""BicopFamily enum — copula family identifiers."""

from __future__ import annotations

from enum import Enum


class BicopFamily(str, Enum):
    indep = "indep"
    gaussian = "gaussian"
    student = "student"
    clayton = "clayton"
    gumbel = "gumbel"
    frank = "frank"
    joe = "joe"
    bb1 = "bb1"
    bb6 = "bb6"
    bb7 = "bb7"
    bb8 = "bb8"
    tawn = "tawn"
    tll = "tll"


_FAMILY_CAN_ROTATE = {
    BicopFamily.indep: False,
    BicopFamily.gaussian: False,
    BicopFamily.student: False,
    BicopFamily.frank: False,
    BicopFamily.tll: False,
}


def family_can_rotate(fam: BicopFamily) -> bool:
    return _FAMILY_CAN_ROTATE.get(fam, True)


def normalize_family(fam: str | BicopFamily) -> BicopFamily:
    if isinstance(fam, BicopFamily):
        return fam
    try:
        return BicopFamily(str(fam).lower())
    except Exception as e:
        raise ValueError(f"Unknown BicopFamily: {fam!r}") from e
