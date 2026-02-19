"""R-vine, D-vine, and C-vine structure representations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


def _check_order(order: Sequence[int], d: int) -> list[int]:
    o = [int(x) for x in order]
    if len(o) != d:
        raise ValueError(f"order must have length d={d}, got {len(o)}")
    if sorted(o) != list(range(1, d + 1)):
        raise ValueError(f"order must be a permutation of 1..{d}, got {o}")
    return o


def _tri_get(tri: list[list[int]], tree: int, edge: int) -> int:
    return int(tri[int(tree)][int(edge)])


class _CallableList(list):
    """List that also supports ``obj(tree, edge)`` call syntax for pyvinecopulib compat."""

    def __call__(self, tree, edge, *_args):
        return self[int(tree)][int(edge)]


def _tri_shape_ok(tri: Sequence[Sequence[int]], d: int, trunc_lvl: int) -> None:
    if len(tri) != trunc_lvl:
        raise ValueError(f"struct_array must have {trunc_lvl} rows, got {len(tri)}")
    for t in range(trunc_lvl):
        exp = d - 1 - t
        if len(tri[t]) != exp:
            raise ValueError(f"struct_array[{t}] must have length {exp}, got {len(tri[t])}")


def _compute_min_array(struct_array: list[list[int]], d: int, trunc_lvl: int) -> list[list[int]]:
    # Mirrors C++ RVineStructure::compute_min_array()
    min_array = [row[:] for row in struct_array]
    for j in range(d - 1):
        for i in range(1, min(d - 1 - j, trunc_lvl)):
            min_array[i][j] = min(struct_array[i][j], min_array[i - 1][j])
    return min_array


def _compute_needed_hfunc1(
    struct_array: list[list[int]],
    min_array: list[list[int]],
    d: int,
    trunc_lvl: int,
) -> list[list[bool]]:
    # Mirrors C++ RVineStructure::compute_needed_hfunc1()
    needed = [[False] * (d - 1) for _ in range(trunc_lvl)]
    if d == 1:
        return needed
    for i in range(min(d - 2, max(trunc_lvl - 1, 0))):
        for j in range(d - 2 - i):
            if struct_array[i + 1][j] != min_array[i + 1][j]:
                needed[i][min_array[i + 1][j] - 1] = True
    return needed


def _compute_needed_hfunc2(
    struct_array: list[list[int]],
    min_array: list[list[int]],
    d: int,
    trunc_lvl: int,
) -> list[list[bool]]:
    # Mirrors C++ RVineStructure::compute_needed_hfunc2()
    needed = [[False] * (d - 1) for _ in range(trunc_lvl)]
    if d == 1:
        return needed
    for i in range(min(d - 2, max(trunc_lvl - 1, 0))):
        for j in range(d - 2 - i):
            needed[i][j] = True
            if struct_array[i + 1][j] == min_array[i + 1][j]:
                needed[i][min_array[i + 1][j] - 1] = True
    return needed


@dataclass(frozen=True)
class RVineStructure:
    """Minimal RVineStructure representation (natural order labels).

    This stores the R-vine structure in natural order:
    - `order` is a permutation of 1..d (used to reorder input data to natural order).
    - `struct_array` is a triangular array with `trunc_lvl` rows, where
      `struct_array[tree][edge]` is in 1..d.
    - `min_array` is derived from `struct_array` as in vinecopulib.
    """

    d: int
    trunc_lvl: int
    order: list[int]
    struct_array: list[list[int]]
    min_array: list[list[int]]
    needed_hfunc1: list[list[bool]]
    needed_hfunc2: list[list[bool]]

    def __post_init__(self):
        # Wrap list fields so they are also callable like pyvinecopulib:
        #   structure.struct_array(tree, edge) → element access
        for name in ("struct_array", "min_array", "needed_hfunc1", "needed_hfunc2"):
            val = getattr(self, name)
            if not isinstance(val, _CallableList):
                object.__setattr__(self, name, _CallableList(val))

    @property
    def matrix(self):
        # Mirrors pyvinecopulib's `RVineStructure.matrix` layout.
        # The anti-diagonal contains `order`, and the upper-left block contains
        # `struct_array` with one extra column for the anti-diagonal.
        import torch

        d = int(self.d)
        M = torch.zeros((d, d), dtype=torch.int64)
        for j in range(d):
            M[d - 1 - j, j] = int(self.order[j])
        for i in range(min(int(self.trunc_lvl), d - 1)):
            row = self.struct_array[i]
            for j, v in enumerate(row):
                M[i, j] = int(v)
        return M

    @staticmethod
    def make_dvine_struct_array(d: int, trunc_lvl: int) -> list[list[int]]:
        # Mirrors C++ RVineStructure::make_dvine_struct_array()
        out: list[list[int]] = []
        for i in range(trunc_lvl):
            row = []
            for j in range(d - 1 - i):
                row.append(i + j + 2)
            out.append(row)
        return out

    @staticmethod
    def make_cvine_struct_array(d: int, trunc_lvl: int) -> list[list[int]]:
        # Mirrors C++ RVineStructure::make_cvine_struct_array()
        out: list[list[int]] = []
        for i in range(trunc_lvl):
            row = []
            for _j in range(d - 1 - i):
                row.append(d - i)
            out.append(row)
        return out

    @classmethod
    def from_order(cls, order: Sequence[int], *, trunc_lvl: int | None = None) -> "RVineStructure":
        d = len(order)
        o = _check_order(order, d)
        if trunc_lvl is None:
            trunc_lvl = d - 1
        trunc = max(0, min(d - 1, int(trunc_lvl)))
        struct = cls.make_dvine_struct_array(d, trunc)
        min_arr = _compute_min_array(struct, d, trunc)
        need1 = _compute_needed_hfunc1(struct, min_arr, d, trunc)
        need2 = _compute_needed_hfunc2(struct, min_arr, d, trunc)
        return cls(d=d, trunc_lvl=trunc, order=o, struct_array=struct, min_array=min_arr, needed_hfunc1=need1, needed_hfunc2=need2)

    @classmethod
    def from_cvine_order(cls, order: Sequence[int], *, trunc_lvl: int | None = None) -> "RVineStructure":
        d = len(order)
        o = _check_order(order, d)
        if trunc_lvl is None:
            trunc_lvl = d - 1
        trunc = max(0, min(d - 1, int(trunc_lvl)))
        struct = cls.make_cvine_struct_array(d, trunc)
        min_arr = _compute_min_array(struct, d, trunc)
        need1 = _compute_needed_hfunc1(struct, min_arr, d, trunc)
        need2 = _compute_needed_hfunc2(struct, min_arr, d, trunc)
        return cls(d=d, trunc_lvl=trunc, order=o, struct_array=struct, min_array=min_arr, needed_hfunc1=need1, needed_hfunc2=need2)

    @classmethod
    def from_vinecopulib_json(cls, obj: dict[str, Any]) -> "RVineStructure":
        # Accepts the JSON object structure used by vinecopulib/pyvinecopulib for RVineStructure:
        # {"order": [...], "array": {"d": d, "t": trunc_lvl, "data": [[...], ...]}}
        if "order" not in obj or "array" not in obj:
            raise ValueError("invalid RVineStructure json: expected keys 'order' and 'array'")
        order = obj["order"]
        arr = obj["array"]
        d = int(arr["d"])
        trunc = int(arr["t"])
        struct = [[int(x) for x in row] for row in arr["data"]]
        o = _check_order(order, d)
        _tri_shape_ok(struct, d, trunc)
        min_arr = _compute_min_array(struct, d, trunc)
        need1 = _compute_needed_hfunc1(struct, min_arr, d, trunc)
        need2 = _compute_needed_hfunc2(struct, min_arr, d, trunc)
        return cls(d=d, trunc_lvl=trunc, order=o, struct_array=struct, min_array=min_arr, needed_hfunc1=need1, needed_hfunc2=need2)

    @classmethod
    def from_order_and_struct_array(
        cls,
        *,
        order: Sequence[int],
        struct_array: Sequence[Sequence[int]],
        trunc_lvl: int | None = None,
        check: bool = True,
    ) -> "RVineStructure":
        """Construct directly from order + triangular array (natural order labels)."""
        d = len(order)
        o = _check_order(order, d) if check else [int(x) for x in order]
        if trunc_lvl is None:
            trunc = len(struct_array)
        else:
            trunc = int(trunc_lvl)
        struct = [[int(x) for x in row] for row in struct_array]
        if check:
            _tri_shape_ok(struct, d, trunc)
        min_arr = _compute_min_array(struct, d, trunc)
        need1 = _compute_needed_hfunc1(struct, min_arr, d, trunc)
        need2 = _compute_needed_hfunc2(struct, min_arr, d, trunc)
        return cls(d=d, trunc_lvl=trunc, order=o, struct_array=struct, min_array=min_arr, needed_hfunc1=need1, needed_hfunc2=need2)

    def struct_at(self, tree: int, edge: int) -> int:
        return _tri_get(self.struct_array, tree, edge)

    def min_at(self, tree: int, edge: int) -> int:
        return _tri_get(self.min_array, tree, edge)

    def needed_hfunc1_at(self, tree: int, edge: int) -> bool:
        if tree < 0 or tree >= self.trunc_lvl:
            return False
        if edge < 0 or edge >= self.d - 1:
            return False
        return bool(self.needed_hfunc1[tree][edge])

    def needed_hfunc2_at(self, tree: int, edge: int) -> bool:
        if tree < 0 or tree >= self.trunc_lvl:
            return False
        if edge < 0 or edge >= self.d - 1:
            return False
        return bool(self.needed_hfunc2[tree][edge])

    @property
    def dim(self) -> int:
        return int(self.d)

    @classmethod
    def from_dimension(cls, d: int, *, trunc_lvl: int | None = None) -> "RVineStructure":
        """Create a default D-vine structure of dimension *d*."""
        return cls.from_order(list(range(1, int(d) + 1)), trunc_lvl=trunc_lvl)

    @classmethod
    def from_matrix(cls, matrix, *, check: bool = True) -> "RVineStructure":
        """Construct from a (d x d) R-vine matrix (pyvinecopulib convention)."""
        import torch
        M = torch.as_tensor(matrix, dtype=torch.int64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("matrix must be square")
        d = int(M.shape[0])
        # Extract order from anti-diagonal
        order = [int(M[d - 1 - j, j].item()) for j in range(d)]
        # Extract struct_array from upper-left triangle
        trunc = d - 1
        struct: list[list[int]] = []
        for i in range(trunc):
            row: list[int] = []
            for j in range(d - 1 - i):
                val = int(M[i, j].item())
                if val == 0:
                    trunc = i
                    break
                row.append(val)
            else:
                struct.append(row)
                continue
            break
        return cls.from_order_and_struct_array(order=order, struct_array=struct, trunc_lvl=trunc, check=check)

    def truncate(self, trunc_lvl: int) -> "RVineStructure":
        """Return a truncated copy (mirrors pyvinecopulib truncate)."""
        trunc_req = max(0, min(int(self.d) - 1, int(trunc_lvl)))
        if trunc_req >= int(self.trunc_lvl):
            return self
        return RVineStructure.from_order_and_struct_array(
            order=list(self.order),
            struct_array=[row[:] for row in self.struct_array[:trunc_req]],
            trunc_lvl=trunc_req,
            check=False,
        )

    @staticmethod
    def simulate(d: int, *, trunc_lvl: int | None = None) -> "RVineStructure":
        """Generate a random R-vine structure of dimension *d*."""
        import torch
        d = int(d)
        if d <= 0:
            raise ValueError("d must be positive")
        perm = torch.randperm(d) + 1
        order = perm.tolist()
        if trunc_lvl is None:
            trunc = d - 1
        else:
            trunc = max(0, min(d - 1, int(trunc_lvl)))
        # Build random struct_array
        struct: list[list[int]] = []
        for i in range(trunc):
            row: list[int] = []
            for j in range(d - 1 - i):
                # Pick a random valid entry from the remaining vars
                candidates = list(range(1, d + 1))
                row.append(candidates[int(torch.randint(0, len(candidates), (1,)).item())])
            struct.append(row)
        # Use D-vine default struct_array for valid structure
        struct = RVineStructure.make_dvine_struct_array(d, trunc)
        return RVineStructure.from_order_and_struct_array(order=order, struct_array=struct, trunc_lvl=trunc, check=False)

    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON dict (pyvinecopulib-compatible)."""
        return {
            "order": list(self.order),
            "array": {
                "d": int(self.d),
                "t": int(self.trunc_lvl),
                "data": [row[:] for row in self.struct_array],
            },
        }

    @classmethod
    def from_json(cls, s: str | dict[str, Any]) -> "RVineStructure":
        """Deserialize from JSON string or dict."""
        if isinstance(s, str):
            obj = json.loads(s)
        else:
            obj = s
        return cls.from_vinecopulib_json(obj)

    def to_file(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, sort_keys=True)

    @classmethod
    def from_file(cls, path: str) -> "RVineStructure":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls.from_json(obj)

    def str(self) -> str:
        """Human-readable string representation (mirrors pyvinecopulib)."""
        M = self.matrix
        lines = [f"<torchvine.RVineStructure> dim={self.d}, trunc_lvl={self.trunc_lvl}"]
        lines.append(str(M.tolist() if hasattr(M, "tolist") else M))
        return "\n".join(lines)


class DVineStructure(RVineStructure):
    def __init__(self, order=None, *, trunc_lvl=None, **kwargs):
        if order is not None and not kwargs:
            base = RVineStructure.from_order(list(order), trunc_lvl=trunc_lvl)
            for field_name in RVineStructure.__dataclass_fields__:
                object.__setattr__(self, field_name, getattr(base, field_name))
        else:
            super().__init__(order=order, trunc_lvl=trunc_lvl, **kwargs)

    @classmethod
    def from_order(cls, order: Sequence[int], *, trunc_lvl: int | None = None) -> "DVineStructure":
        return cls(order, trunc_lvl=trunc_lvl)


class CVineStructure(RVineStructure):
    def __init__(self, order=None, *, trunc_lvl=None, **kwargs):
        if order is not None and not kwargs:
            base = RVineStructure.from_cvine_order(list(order), trunc_lvl=trunc_lvl)
            for field_name in RVineStructure.__dataclass_fields__:
                object.__setattr__(self, field_name, getattr(base, field_name))
        else:
            super().__init__(order=order, trunc_lvl=trunc_lvl, **kwargs)

    @classmethod
    def from_order(cls, order: Sequence[int], *, trunc_lvl: int | None = None) -> "CVineStructure":
        return cls(order, trunc_lvl=trunc_lvl)
