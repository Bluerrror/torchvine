"""Vine structure and pair-copula family selection (MST-based)."""

from __future__ import annotations

import math
import concurrent.futures
from dataclasses import dataclass, field
from typing import Iterable

import torch

from .bicop import Bicop
from .fit_controls import FitControlsVinecop, FitControlsBicop
from .rvine_structure import RVineStructure
from . import stats
from .families import BicopFamily


@dataclass
class _Vertex:
    # Data needed to form pseudo observations in the next tree.
    hfunc1: torch.Tensor | None = None
    hfunc2: torch.Tensor | None = None
    hfunc1_sub: torch.Tensor | None = None
    hfunc2_sub: torch.Tensor | None = None
    prev_edge_indices: list[int] = field(default_factory=list)  # size 2
    conditioned: list[int] = field(default_factory=list)
    conditioning: list[int] = field(default_factory=list)
    all_indices: list[int] = field(default_factory=list)
    var_types: tuple[str, str] = ("c", "c")


@dataclass
class _Edge:
    u: int
    v: int
    weight: float = 1.0
    crit: float = 0.0
    pc_data: torch.Tensor | None = None  # (n,2)
    conditioned: list[int] = field(default_factory=list)  # size 2, 0-based var indices
    conditioning: list[int] = field(default_factory=list)
    all_indices: list[int] = field(default_factory=list)
    pair_copula: Bicop | None = None
    hfunc1: torch.Tensor | None = None
    hfunc2: torch.Tensor | None = None
    hfunc1_sub: torch.Tensor | None = None
    hfunc2_sub: torch.Tensor | None = None
    var_types: tuple[str, str] = ("c", "c")
    loglik: float = 0.0
    npars: float = 0.0


class _Graph:
    def __init__(self, n_vertices: int):
        self.vertices: list[_Vertex] = [_Vertex() for _ in range(int(n_vertices))]
        self._edges: dict[tuple[int, int], _Edge] = {}
        self._nbrs: list[set[int]] = [set() for _ in range(int(n_vertices))]
        # Preserve insertion order of undirected edges (canonicalized key).
        # Boost's adjacency_list iterators are deterministic; this is the closest
        # analogue and avoids tie-breaking drift in MST and finalize().
        self._edge_order: list[tuple[int, int]] = []

    def copy_structure(self) -> "_Graph":
        g = _Graph(len(self.vertices))
        # vertices shallow copy, edges deep copy
        for i, v in enumerate(self.vertices):
            g.vertices[i] = _Vertex(
                hfunc1=v.hfunc1,
                hfunc2=v.hfunc2,
                prev_edge_indices=list(v.prev_edge_indices),
                conditioned=list(v.conditioned),
                conditioning=list(v.conditioning),
                all_indices=list(v.all_indices),
            )
        g._edge_order = list(self._edge_order)
        for k, e in self._edges.items():
            g._edges[k] = _Edge(
                u=e.u,
                v=e.v,
                weight=float(e.weight),
                crit=float(e.crit),
                pc_data=e.pc_data,
                conditioned=list(e.conditioned),
                conditioning=list(e.conditioning),
                all_indices=list(e.all_indices),
                pair_copula=e.pair_copula,
                hfunc1=e.hfunc1,
                hfunc2=e.hfunc2,
            )
            g._nbrs[e.u].add(e.v)
            g._nbrs[e.v].add(e.u)
        return g

    def add_edge(self, u: int, v: int) -> _Edge:
        u = int(u)
        v = int(v)
        if u == v:
            raise ValueError("self-loop")
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in self._edges:
            return self._edges[(a, b)]
        # Keep insertion orientation in (u,v) to mirror boost::source/target
        # ordering, which impacts which h-function is considered "first".
        e = _Edge(u=u, v=v)
        self._edges[(a, b)] = e
        self._edge_order.append((a, b))
        self._nbrs[a].add(b)
        self._nbrs[b].add(a)
        return e

    def remove_edge(self, u: int, v: int) -> None:
        a, b = (u, v) if u < v else (v, u)
        e = self._edges.pop((a, b), None)
        if e is None:
            return
        self._nbrs[a].discard(b)
        self._nbrs[b].discard(a)

    def edges(self) -> Iterable[_Edge]:
        for k in self._edge_order:
            e = self._edges.get(k)
            if e is not None:
                yield e

    def edge(self, u: int, v: int) -> _Edge | None:
        a, b = (u, v) if u < v else (v, u)
        return self._edges.get((a, b))

    def num_vertices(self) -> int:
        return len(self.vertices)

    def num_edges(self) -> int:
        return len(self._edges)

    def degree(self, u: int) -> int:
        return len(self._nbrs[int(u)])


def _intersect(a: list[int], b: list[int]) -> list[int]:
    sb = set(b)
    return [x for x in a if x in sb]


def _sym_diff_ordered(a: list[int], b: list[int]) -> list[int]:
    sb = set(b)
    out0 = [x for x in a if x not in sb]
    sa = set(a)
    out1 = [x for x in b if x not in sa]
    return out0 + out1


def _is_same_set(a: list[int], b: list[int]) -> bool:
    return set(a) == set(b)


def _find_position(x: int, arr: list[int]) -> int:
    for i, v in enumerate(arr):
        if int(v) == int(x):
            return i
    return -1


def _find_common_neighbor(v0: _Vertex, v1: _Vertex) -> int:
    inter = _intersect(v0.prev_edge_indices, v1.prev_edge_indices)
    return int(inter[0]) if inter else -1


def _get_hfunc(v: _Vertex, is_first: bool) -> torch.Tensor:
    if is_first:
        if v.hfunc1 is None:
            raise RuntimeError("missing hfunc1")
        return v.hfunc1
    if v.hfunc2 is None:
        raise RuntimeError("missing hfunc2")
    return v.hfunc2


def _get_hfunc_sub(v: _Vertex, is_first: bool) -> torch.Tensor:
    # Mirrors VinecopSelector::get_hfunc_sub
    if is_first:
        if v.hfunc1_sub is not None:
            return v.hfunc1_sub
        return _get_hfunc(v, True)
    if v.hfunc2_sub is not None:
        return v.hfunc2_sub
    return _get_hfunc(v, False)


def _get_pc_data(v0i: int, v1i: int, tree: _Graph) -> torch.Tensor:
    v0 = tree.vertices[int(v0i)]
    v1 = tree.vertices[int(v1i)]
    cn = _find_common_neighbor(v0, v1)
    if cn < 0:
        raise RuntimeError("proximity condition violated (no common neighbor)")
    pos0 = _find_position(cn, v0.prev_edge_indices)
    pos1 = _find_position(cn, v1.prev_edge_indices)
    u1 = _get_hfunc(v0, pos0 == 0)
    u2 = _get_hfunc(v1, pos1 == 0)
    return torch.stack([u1, u2], dim=1)


def _remove_nans_2d(xy: torch.Tensor, weights: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    mask = torch.isfinite(xy[:, 0]) & torch.isfinite(xy[:, 1])
    if weights is not None and weights.numel() > 0:
        mask = mask & torch.isfinite(weights)
    if not bool(mask.any()):
        return xy[:0], (weights[:0] if weights is not None and weights.numel() > 0 else weights), 0.0
    xy2 = xy[mask]
    w2 = weights[mask] if weights is not None and weights.numel() > 0 else weights
    freq = float(xy2.shape[0]) / float(xy.shape[0])
    return xy2, w2, freq


def _criterion_wdm(x: torch.Tensor, y: torch.Tensor, method: str, weights: torch.Tensor | None) -> float:
    # This torch port intentionally does not depend on pyvinecopulib.
    #
    # The original C++ uses a "wdm" helper for some dependence measures. Here we
    # implement the methods needed for vine structure selection:
    # - "rho": Spearman's rho (computed as Pearson correlation of ranks).
    # - "hoeffd": Hoeffding's D (fast O(n log n) estimator; ties are broken
    #            arbitrarily, which is fine for continuous pseudo-observations).

    def _rank_1_to_n(v: torch.Tensor) -> torch.Tensor:
        # Returns 1..n ranks (ties deterministically broken by stable sort order).
        # This is fully torch-native and works on GPU.
        v = torch.as_tensor(v)
        idx = torch.argsort(v, stable=True)
        inv = torch.empty_like(idx)
        inv[idx] = torch.arange(int(v.numel()), device=v.device, dtype=idx.dtype)
        return inv.to(torch.int64) + 1

    def _spearman_rho(xx: torch.Tensor, yy: torch.Tensor, ww: torch.Tensor | None) -> float:
        # Spearman's rho = Pearson correlation of ranks.
        rx = _rank_1_to_n(xx).to(xx.dtype)
        ry = _rank_1_to_n(yy).to(yy.dtype)
        return float(stats.pearson_cor(rx, ry, weights=ww))

    def _hoeffding_d_approx(xx: torch.Tensor, yy: torch.Tensor) -> float:
        # GPU-friendly approximation of Hoeffding's D via binned empirical copula C_n.
        # This avoids CPU-only Fenwick tree logic and keeps the heavy work on-device.
        #
        # We approximate:
        #   D ~= 30 * mean_i ( C_n(u_i, v_i) - u_i v_i )^2
        # where u_i,v_i are normalized ranks.
        n = int(xx.numel())
        if n < 20:
            return 0.0

        R = _rank_1_to_n(xx).to(xx.dtype)
        S = _rank_1_to_n(yy).to(yy.dtype)
        u = (R - 0.5) / float(n)
        v = (S - 0.5) / float(n)

        # Choose a small grid (trade-off speed/accuracy).
        B = int(max(32, min(128, round(math.sqrt(n)))))
        bx = torch.clamp((u * float(B)).to(torch.int64), 0, B - 1)
        by = torch.clamp((v * float(B)).to(torch.int64), 0, B - 1)
        idx = bx * B + by

        counts = torch.bincount(idx, minlength=B * B).reshape(B, B).to(xx.dtype)
        cdf = counts.cumsum(dim=0).cumsum(dim=1) / float(n)
        cn = cdf[bx, by]
        d = 30.0 * torch.mean((cn - (u * v)) ** 2)
        return float(d.item())

    if method == "rho":
        return _spearman_rho(x, y, weights)
    if method == "hoeffd":
        # weights are ignored for this estimator in the torch port (keeps it GPU-friendly).
        return _hoeffding_d_approx(x, y)
    raise NotImplementedError(f"wdm method {method!r} not implemented")


def _calculate_criterion(pc_data: torch.Tensor, name: str, weights: torch.Tensor | None) -> float:
    # Port of vinecopulib::tools_select::calculate_criterion()
    xy = torch.as_tensor(pc_data)
    xy2, w2, freq = _remove_nans_2d(xy, weights)
    if xy2.shape[0] <= 10:
        return 0.0

    x = xy2[:, 0]
    y = xy2[:, 1]
    w = 0.0
    if name == "mcor":
        from .tll_fit import pairwise_mcor

        w = float(pairwise_mcor(torch.stack([x, y], dim=1), weights=w2))
    elif name == "joe":
        z = stats.qnorm(stats.clamp_unit(torch.stack([x, y], dim=1)))
        r = stats.pearson_cor(z[:, 0], z[:, 1], weights=w2)
        rr = max(0.0, min(1.0 - 1e-15, r * r))
        w = -0.5 * math.log(1.0 - rr)
    elif name == "tau":
        w = float(stats.kendall_tau(x, y, weights=w2))
    elif name == "rho":
        # wdm's rho is Spearman's rho with tie correction; for parity use pv.wdm.
        w = _criterion_wdm(x, y, "rho", w2)
    elif name == "hoeffd":
        w = _criterion_wdm(x, y, "hoeffd", w2)
    else:
        raise NotImplementedError(f"tree_criterion={name!r} not implemented")

    if math.isnan(w):
        w = 0.0
    return abs(float(w)) * math.sqrt(freq)


def _mst_prim(graph: _Graph) -> set[tuple[int, int]]:
    n = graph.num_vertices()
    if n <= 1:
        return set()
    in_tree = [False] * n
    best_w = [float("inf")] * n
    best_p = [-1] * n
    best_w[0] = 0.0
    for _ in range(n):
        # pick next vertex
        v = -1
        wv = float("inf")
        for i in range(n):
            if (not in_tree[i]) and best_w[i] < wv:
                v = i
                wv = best_w[i]
        if v < 0:
            break
        in_tree[v] = True
        # relax neighbors
        for u in sorted(graph._nbrs[v]):
            if in_tree[u]:
                continue
            e = graph.edge(v, u)
            if e is None:
                continue
            w = float(e.weight)
            bw = best_w[u]
            if (w < bw - 1e-15) or (abs(w - bw) <= 1e-15 and (best_p[u] < 0 or v < best_p[u])):
                best_w[u] = w
                best_p[u] = v
    out: set[tuple[int, int]] = set()
    for u in range(1, n):
        p = best_p[u]
        if p >= 0:
            a, b = (p, u) if p < u else (u, p)
            out.add((a, b))
    return out


def _mst_kruskal(graph: _Graph) -> set[tuple[int, int]]:
    n = graph.num_vertices()
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    edges = list(graph.edges())
    edges.sort(key=lambda e: (float(e.weight), min(int(e.u), int(e.v)), max(int(e.u), int(e.v))))
    out: set[tuple[int, int]] = set()
    for e in edges:
        a, b = (e.u, e.v) if e.u < e.v else (e.v, e.u)
        if union(a, b):
            out.add((a, b))
            if len(out) == n - 1:
                break
    return out


def _random_spanning_tree_wilson(graph: _Graph, *, weighted: bool, seed: int = 0) -> set[tuple[int, int]]:
    # Loop-erased random walk (Wilson's algorithm).
    # Returns a spanning tree as a set of undirected canonical edges.
    n = graph.num_vertices()
    if n <= 1:
        return set()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    in_tree = [False] * n
    root = int(torch.randint(low=0, high=n, size=(1,), generator=rng).item())
    in_tree[root] = True
    tree_edges: set[tuple[int, int]] = set()

    def pick_next(cur: int) -> int:
        nbrs = sorted(graph._nbrs[cur])
        if not nbrs:
            raise RuntimeError("graph is disconnected")
        if not weighted:
            j = int(torch.randint(low=0, high=len(nbrs), size=(1,), generator=rng).item())
            return int(nbrs[j])
        # Weighted: probability proportional to conductance = 1/(cost+eps),
        # where cost is e.weight (smaller cost -> more likely).
        wts = []
        for nb in nbrs:
            e = graph.edge(cur, nb)
            if e is None:
                wts.append(0.0)
            else:
                wts.append(1.0 / (float(e.weight) + 1e-10))
        probs = torch.tensor(wts, device=xx.device, dtype=xx.dtype)
        s = float(probs.sum().item())
        if s <= 0.0:
            j = int(torch.randint(low=0, high=len(nbrs), size=(1,), generator=rng).item())
            return int(nbrs[j])
        probs = probs / s
        j = int(torch.multinomial(probs, 1, generator=rng).item())
        return int(nbrs[j])

    for start in range(n):
        if in_tree[start]:
            continue
        # random walk until hit tree
        path = [start]
        pos = {start: 0}
        cur = start
        while not in_tree[cur]:
            nxt = pick_next(cur)
            if nxt in pos:
                # loop erase
                cut = pos[nxt]
                path = path[: cut + 1]
                pos = {v: i for i, v in enumerate(path)}
            else:
                path.append(nxt)
                pos[nxt] = len(path) - 1
            cur = nxt

        # add loop-erased path to tree
        for i in range(len(path) - 1):
            a = int(path[i])
            b = int(path[i + 1])
            in_tree[a] = True
            in_tree[b] = True
            x, y = (a, b) if a < b else (b, a)
            tree_edges.add((x, y))
    return tree_edges


class VinecopSelectorTorch:
    def __init__(
        self,
        data: torch.Tensor,
        controls: FitControlsVinecop,
        *,
        var_types: list[str],
        vine_struct: RVineStructure | None = None,
        pair_copulas: list[list[Bicop]] | None = None,
    ):
        # Data can be n x d (all continuous) or include a second block for left-limits
        # for discrete variables (n x 2d) or (n x (d + k)).
        self.data_raw = torch.as_tensor(data)
        self.controls = controls
        self.var_types = var_types
        self.d = int(len(var_types))
        self.vine_struct = vine_struct
        self.structure_unknown = vine_struct is None or int(vine_struct.trunc_lvl) == 0
        self.trees: list[_Graph] = []
        self.pair_copulas = pair_copulas
        self.u_main, self.u_sub_full = self._format_data(self.data_raw)
        # Clamp to open unit interval (matches vinecopulib behavior).
        self.u_main = stats.clamp_unit(self.u_main)
        self.u_sub_full = stats.clamp_unit(self.u_sub_full)

    def _format_data(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors vinecopulib data layout for discrete variables.
        u = torch.as_tensor(u)
        if u.ndim != 2:
            raise ValueError("data must be 2D")
        n = int(u.shape[0])
        d = int(self.d)
        k = sum(1 for t in self.var_types if t == "d")
        if u.shape[1] == d:
            u_main = u
            u_sub = u  # no discrete left-limits provided; treat as identical
            return u_main, u_sub
        if u.shape[1] == 2 * d:
            u_main = u[:, :d]
            u_sub = u[:, d:]
            return u_main, u_sub
        if u.shape[1] == d + k:
            u_main = u[:, :d]
            u_sub = u_main.clone()
            disc_count = 0
            for j, t in enumerate(self.var_types):
                if t == "d":
                    u_sub[:, j] = u[:, d + disc_count]
                    disc_count += 1
            return u_main, u_sub
        raise ValueError(f"data must have shape (n,{d}), (n,{2*d}), or (n,{d+k}); got {tuple(u.shape)}")

    def _make_base_tree(self) -> _Graph:
        # Mimics C++ make_base_tree: star with root node at index d.
        g = _Graph(self.d + 1)
        root = self.d
        order = list(range(1, self.d + 1))
        if self.vine_struct is not None:
            order = list(self.vine_struct.order)

        for target in range(self.d):
            e = g.add_edge(root, target)
            # Initialize with u[:, order[target]-1] (natural order)
            col = self.u_main[:, int(order[target]) - 1].clone()
            e.hfunc1 = col
            # Make orientation irrelevant for the base tree: the "other" hfunc
            # is never conceptually used, but iterator/source-target conventions
            # can differ; setting it to the same values keeps pc_data stable.
            e.hfunc2 = col
            # For discrete variables, also initialize left-limits.
            col_sub = self.u_sub_full[:, int(order[target]) - 1].clone()
            if self.var_types[int(order[target]) - 1] == "d":
                e.hfunc1_sub = col_sub
                e.hfunc2_sub = col_sub
                e.var_types = ("d", "d")
            else:
                e.hfunc1_sub = None
                e.hfunc2_sub = None
                e.var_types = ("c", "c")
            e.conditioned = [int(order[target]) - 1]
            e.conditioning = []
            e.all_indices = list(e.conditioned)
        return g

    def _edges_as_vertices(self, prev_tree: _Graph) -> _Graph:
        # Each edge becomes one vertex. Deterministic ordering by edge key.
        edges = list(prev_tree.edges())
        new_tree = _Graph(len(edges))
        for i, e in enumerate(edges):
            v = new_tree.vertices[i]
            v.hfunc1 = e.hfunc1
            v.hfunc2 = e.hfunc2
            v.hfunc1_sub = e.hfunc1_sub
            v.hfunc2_sub = e.hfunc2_sub
            v.conditioned = list(e.conditioned)
            v.conditioning = list(e.conditioning)
            v.all_indices = list(e.all_indices)
            v.prev_edge_indices = [int(e.u), int(e.v)]
            v.var_types = e.var_types
        return new_tree

    def _add_allowed_edges(self, tree: _Graph) -> None:
        crit_name = self.controls.tree_criterion
        thr = float(self.controls.threshold)
        weights = self.controls.weights

        if self.structure_unknown:
            # Add all edges satisfying proximity condition.
            for v0 in range(tree.num_vertices()):
                for v1 in range(v0):
                    cn = _find_common_neighbor(tree.vertices[v0], tree.vertices[v1])
                    if cn < 0:
                        continue
                    e = tree.add_edge(v0, v1)
                    pc = _get_pc_data(v0, v1, tree)
                    wts = weights
                    if wts is not None and wts.numel() > 0 and wts.device != pc.device:
                        wts = wts.to(pc.device)
                    crit = _calculate_criterion(pc, crit_name, wts)
                    w = 1.0 - (1.0 if crit >= thr else 0.0) * crit
                    e.weight = float(w)
                    e.crit = float(crit)
        else:
            # Structure fixed for this tree (until we surpass its truncation level).
            assert self.vine_struct is not None
            tree_idx = self.d - tree.num_vertices()
            if tree_idx < int(self.vine_struct.trunc_lvl):
                edges = tree.num_vertices() - 1
                for v0 in range(edges):
                    v1 = int(self.vine_struct.min_at(tree_idx, v0)) - 1
                    e = tree.add_edge(v0, v1)
                    pc = _get_pc_data(v0, v1, tree)
                    wts = weights
                    if wts is not None and wts.numel() > 0 and wts.device != pc.device:
                        wts = wts.to(pc.device)
                    crit = _calculate_criterion(pc, crit_name, wts)
                    e.weight = 1.0
                    e.crit = float(crit)
            else:
                self.structure_unknown = True
                self._add_allowed_edges(tree)

    def _select_edges(self, tree: _Graph) -> None:
        if tree.num_vertices() <= 2:
            return
        algo = self.controls.tree_algorithm
        if algo == "mst_prim":
            keep = _mst_prim(tree)
        elif algo == "mst_kruskal":
            keep = _mst_kruskal(tree)
        elif algo == "random_unweighted":
            seed = int(self.controls.seeds[0]) if self.controls.seeds else 0
            keep = _random_spanning_tree_wilson(tree, weighted=False, seed=seed)
        elif algo == "random_weighted":
            seed = int(self.controls.seeds[0]) if self.controls.seeds else 0
            keep = _random_spanning_tree_wilson(tree, weighted=True, seed=seed)
        else:
            raise NotImplementedError(f"tree_algorithm={algo!r} not implemented in torch selector")
        # Remove all other edges.
        for e in list(tree.edges()):
            a, b = (e.u, e.v) if e.u < e.v else (e.v, e.u)
            if (a, b) not in keep:
                tree.remove_edge(e.u, e.v)

    def _add_edge_info(self, tree: _Graph) -> None:
        for e in tree.edges():
            v0 = tree.vertices[e.u]
            v1 = tree.vertices[e.v]
            cn = _find_common_neighbor(v0, v1)
            if cn < 0:
                raise RuntimeError("proximity condition violated")
            pos0 = _find_position(cn, v0.prev_edge_indices)
            pos1 = _find_position(cn, v1.prev_edge_indices)

            # var types of the two conditioned variables (port of add_pc_info)
            e.var_types = (v0.var_types[1 - pos0], v1.var_types[1 - pos1])

            n = int(v0.hfunc1.numel()) if v0.hfunc1 is not None else 0
            pc = torch.empty((n, 2), dtype=v0.hfunc1.dtype, device=v0.hfunc1.device)
            pc[:, 0] = _get_hfunc(v0, pos0 == 0)
            pc[:, 1] = _get_hfunc(v1, pos1 == 0)
            if (e.var_types[0] == "d") or (e.var_types[1] == "d"):
                pc4 = torch.empty((n, 4), dtype=pc.dtype, device=pc.device)
                pc4[:, :2] = pc
                pc4[:, 2] = _get_hfunc_sub(v0, pos0 == 0)
                pc4[:, 3] = _get_hfunc_sub(v1, pos1 == 0)
                pc = pc4
            e.pc_data = pc

            e.conditioned = _sym_diff_ordered(v0.all_indices, v1.all_indices)
            e.conditioning = _intersect(v0.all_indices, v1.all_indices)
            e.all_indices = list(e.conditioned) + list(e.conditioning)

    def _select_pair_copulas(self, tree: _Graph, *, tree_level: int) -> None:
        # tree_level = 0 for first vine tree (trees_[1] in C++).
        thr = float(self.controls.threshold)

        def bicop_controls() -> FitControlsBicop:
            fams = list(self.controls.family_set)
            # Default: use all families.
            if not fams:
                fams = list(BicopFamily)
            selcrit = self.controls.selection_criterion
            psi0 = float(self.controls.psi0)
            if selcrit == "mbicv":
                selcrit = "mbic"
                psi0 = psi0 ** float(tree_level + 1)
            return FitControlsBicop(
                family_set=fams,
                parametric_method=self.controls.parametric_method,
                nonparametric_method=self.controls.nonparametric_method,
                nonparametric_mult=self.controls.nonparametric_mult,
                selection_criterion=selcrit,
                weights=self.controls.weights,
                psi0=float(psi0),
                preselect_families=self.controls.preselect_families,
                allow_rotations=self.controls.allow_rotations,
                num_threads=1,
            )

        bc = bicop_controls()
        wts_global = self.controls.weights
        edges = list(tree.edges())

        def _fit_edge(args):
            """Fit pair copula for a single edge. Runs in a worker thread."""
            edge_idx, e = args
            if e.pc_data is None:
                raise RuntimeError("edge missing pc_data")
            pc_full = e.pc_data

            mask = torch.isfinite(pc_full).all(dim=1)
            wts = wts_global
            if wts is not None and wts.numel() > 0 and wts.device != pc_full.device:
                wts = wts.to(pc_full.device)
            if wts is not None and wts.numel() > 0:
                mask = mask & torch.isfinite(wts)
            pc = pc_full[mask]
            wts_masked = wts[mask] if wts is not None and wts.numel() > 0 else None

            if pc.shape[0] < 2:
                cop = Bicop(var_types=e.var_types)
            else:
                if not self.controls.select_families:
                    if self.pair_copulas is None:
                        raise RuntimeError("select_families=False requires existing pair_copulas")
                    if tree_level >= len(self.pair_copulas) or edge_idx >= len(self.pair_copulas[tree_level]):
                        raise RuntimeError("pair_copulas store incompatible with selected tree")
                    cop = self.pair_copulas[tree_level][edge_idx]
                    cop.var_types = e.var_types
                    cop.fit(pc, bc)
                else:
                    if float(e.crit) < thr:
                        cop = Bicop(var_types=e.var_types)
                    else:
                        cop = Bicop(var_types=e.var_types)
                        cop.select(pc, bc)

            try:
                if hasattr(cop, "_fit_loglik") and cop._fit_loglik is not None:
                    loglik = float(cop._fit_loglik)
                else:
                    loglik = float(cop.loglik(pc, weights=wts_masked))
            except Exception:
                loglik = 0.0

            npars = float(cop.get_npars())

            # Compute h-functions for next tree on full data
            hfunc1 = cop.hfunc1(pc_full)
            hfunc2 = cop.hfunc2(pc_full)
            hfunc1_sub = None
            hfunc2_sub = None
            if e.var_types[1] == "d":
                sub = pc_full.clone()
                sub[:, 1] = sub[:, 3]
                hfunc1_sub = cop.hfunc1(sub)
            if e.var_types[0] == "d":
                sub = pc_full.clone()
                sub[:, 0] = sub[:, 2]
                hfunc2_sub = cop.hfunc2(sub)

            return edge_idx, cop, loglik, npars, hfunc1, hfunc2, hfunc1_sub, hfunc2_sub

        # Parallel fitting: each edge in the same tree level is independent.
        # PyTorch C++ backend releases the GIL, enabling true multi-threaded execution.
        num_workers = min(self.controls.num_threads, len(edges))
        if num_workers > 1 and len(edges) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(_fit_edge, enumerate(edges)))
        else:
            results = [_fit_edge(args) for args in enumerate(edges)]

        # Write results back to edges (must be sequential to preserve order).
        for result in results:
            edge_idx, cop, loglik, npars, hfunc1, hfunc2, hfunc1_sub, hfunc2_sub = result
            e = edges[edge_idx]
            e.pair_copula = cop
            e.loglik = loglik
            e.npars = npars
            e.hfunc1 = hfunc1
            e.hfunc2 = hfunc2
            if hfunc1_sub is not None:
                e.hfunc1_sub = hfunc1_sub
            if hfunc2_sub is not None:
                e.hfunc2_sub = hfunc2_sub

    def _finalize(self, trunc_lvl: int) -> tuple[RVineStructure, list[list[Bicop]]]:
        trunc_lvl = max(0, min(self.d - 1, int(trunc_lvl)))
        pcs: list[list[Bicop]] = [[Bicop() for _ in range(self.d - 1 - t)] for t in range(trunc_lvl)]
        mat: list[list[int]] = [[0 for _ in range(self.d - 1 - t)] for t in range(trunc_lvl)]
        order0: list[int] = [0] * self.d  # 0-based variable indices

        if trunc_lvl == 0:
            s = RVineStructure.from_order_and_struct_array(order=list(range(1, self.d + 1)), struct_array=[], trunc_lvl=0, check=False)
            return s, pcs

        # Copy trees because we remove edges while filling columns.
        trees = [t.copy_structure() for t in self.trees]

        ning_set: list[int] = []
        for col in range(self.d - 1):
            # which tree is the highest in this column
            t = max(min(trunc_lvl, self.d - 1 - col), 1)

            # find a leaf edge in trees[t]
            chosen: _Edge | None = None
            chosen_pos = 0
            for e in trees[t].edges():
                d0 = trees[t].degree(e.u)
                d1 = trees[t].degree(e.v)
                if min(d0, d1) > 1:
                    continue
                chosen = e
                chosen_pos = 1 if d1 == 1 else 0
                break
            if chosen is None:
                raise RuntimeError("failed to find leaf edge while finalizing structure")

            if chosen.pair_copula is None:
                raise RuntimeError("edge missing pair_copula while finalizing")

            if chosen_pos == 1:
                chosen.pair_copula.flip()

            order0[col] = int(chosen.conditioned[chosen_pos])
            mat[t - 1][col] = int(chosen.conditioned[1 - chosen_pos])
            pcs[t - 1][col] = chosen.pair_copula
            ning_set = list(chosen.conditioning)

            trees[t].remove_edge(chosen.u, chosen.v)

            # fill column bottom to top
            for k in range(1, t):
                check_set = [order0[col]] + list(ning_set)
                found: _Edge | None = None
                for e in trees[t - k].edges():
                    if _is_same_set(e.all_indices, check_set):
                        found = e
                        break
                if found is None or found.pair_copula is None:
                    raise RuntimeError("failed to match edge while finalizing")
                pos = 1 if order0[col] == int(found.conditioned[1]) else 0
                if pos == 1:
                    found.pair_copula.flip()
                mat[t - k - 1][col] = int(found.conditioned[1 - pos])
                pcs[t - 1 - k][col] = found.pair_copula
                ning_set = list(found.conditioning)
                trees[t - k].remove_edge(found.u, found.v)

        order0[self.d - 1] = mat[0][self.d - 2]

        # Convert to 1-based (variable labels).
        order = [x + 1 for x in order0]
        for i in range(min(self.d - 1, trunc_lvl)):
            for j in range(self.d - 1 - i):
                mat[i][j] = int(mat[i][j]) + 1

        # IMPORTANT: vinecopulib constructs an RVineStructure with natural_order=false
        # and then relabels the struct array to natural order using the diagonal order.
        # Our RVineStructure stores the structure array in natural order (i.e., labels
        # refer to positions in `order`), so we apply the same relabeling here.
        # Port of RVineStructure::to_natural_order():
        #   new_label = argsort(order)[old_label - 1] + 1
        idx_sorted = sorted(range(self.d), key=lambda i: int(order[i]))  # tools_stl::get_order(order)
        for i in range(min(self.d - 1, trunc_lvl)):
            for j in range(self.d - 1 - i):
                old = int(mat[i][j])
                mat[i][j] = int(idx_sorted[old - 1]) + 1

        struct = RVineStructure.from_order_and_struct_array(order=order, struct_array=mat, trunc_lvl=trunc_lvl, check=False)
        return struct, pcs

    def select_all_trees(self) -> tuple[RVineStructure, list[list[Bicop]]]:
        if self.data_raw.ndim != 2:
            raise ValueError("data must be 2D")
        if int(self.u_main.shape[1]) != int(self.d):
            raise ValueError("data has wrong dimension")
        if not self.controls.select_families and (self.vine_struct is None or int(self.vine_struct.trunc_lvl) == 0):
            raise ValueError("select_families=False requires a fixed structure (trunc_lvl > 0)")

        needs_sparse = (self.controls.selection_criterion == "mbicv") or bool(self.controls.select_threshold) or bool(self.controls.select_trunc_lvl)
        if needs_sparse:
            return self._sparse_select_all_trees()
        return self._select_all_trees()

    def _select_all_trees(self) -> tuple[RVineStructure, list[list[Bicop]]]:
        self.trees = [self._make_base_tree()]

        trunc = self.controls.trunc_lvl
        if trunc is None:
            trunc = self.d - 1
        trunc = max(0, min(self.d - 1, int(trunc)))

        for t in range(self.d - 1):
            new_tree = self._edges_as_vertices(self.trees[t])
            if self.vine_struct is not None and t >= int(self.vine_struct.trunc_lvl):
                self.structure_unknown = True
            self._add_allowed_edges(new_tree)
            self._select_edges(new_tree)
            if new_tree.num_vertices() > 0:
                self._add_edge_info(new_tree)
                self._select_pair_copulas(new_tree, tree_level=t)
            self.trees.append(new_tree)
            if trunc == t + 1:
                break
        return self._finalize(trunc)

    def _thresholded_crits(self) -> list[float]:
        out: list[float] = []
        thr = float(self.controls.threshold)
        for t in range(1, len(self.trees)):
            for e in self.trees[t].edges():
                if float(e.crit) < thr:
                    out.append(float(e.crit))
        return out

    def _get_next_threshold(self, thresholded: list[float]) -> float:
        if not thresholded:
            return 1.0
        xs = sorted(thresholded, reverse=True)
        alpha = 0.05
        m = len(xs)
        new_index = int(math.ceil(float(m) * alpha) - 1.0)
        new_index = max(0, min(m - 1, new_index))
        return float(xs[new_index])

    def _n_eff(self) -> float:
        w = self.controls.weights
        n = float(self.u_main.shape[0])
        if w is None or w.numel() == 0:
            return n
        ws = float(w.sum().item())
        w2s = float((w * w).sum().item())
        if w2s <= 0.0:
            return n
        return (ws * ws) / w2s

    def _get_mbicv_of_tree(self, t: int, loglik_tree: float) -> float:
        # Port of VinecopSelector::get_mbicv_of_tree
        npars = 0.0
        non_indeps = 0
        for e in self.trees[t + 1].edges():
            npars += float(e.npars)
            non_indeps += int(e.pair_copula is not None and e.pair_copula.family != BicopFamily.indep)
        indeps = (self.d - t - 1) - non_indeps
        psi0 = float(self.controls.psi0) ** float(t + 1)
        log_prior = float(non_indeps) * math.log(psi0) + float(indeps) * math.log(1.0 - psi0)
        n_eff = self._n_eff()
        return -2.0 * float(loglik_tree) + math.log(n_eff) * float(npars) - 2.0 * float(log_prior)

    def _sparse_select_all_trees(self) -> tuple[RVineStructure, list[list[Bicop]]]:
        # Port of VinecopSelector::sparse_select_all_trees (continuous+discrete).
        family_set0 = list(self.controls.family_set)
        thresholded: list[float] = []
        mbicv_opt = float("inf")
        trees_opt: list[_Graph] = []
        loglik_opt = 0.0
        trunc_opt = self.d - 1
        thr_opt = float(self.controls.threshold)

        needs_break = False
        while not needs_break:
            # restore
            self.controls.family_set = list(family_set0)
            self.controls.trunc_lvl = self.d - 1
            self.trees = [self._make_base_tree()]

            if self.controls.select_threshold:
                self.controls.threshold = self._get_next_threshold(thresholded)

            mbicv = 0.0
            mbicv_trunc = 0.0
            loglik = 0.0
            select_trunc_lvl = bool(self.controls.select_trunc_lvl)
            select_threshold = bool(self.controls.select_threshold)
            num_changed = 0.0
            num_total = float(self.d) * float(self.d - 1) / 2.0

            t = 0
            while t < self.d - 1:
                if self.controls.trunc_lvl is not None and int(self.controls.trunc_lvl) < t:
                    break
                new_tree = self._edges_as_vertices(self.trees[t])
                self._add_allowed_edges(new_tree)
                self._select_edges(new_tree)
                if new_tree.num_vertices() > 0:
                    self._add_edge_info(new_tree)
                    self._select_pair_copulas(new_tree, tree_level=t)
                self.trees.append(new_tree)

                num_changed += float(self.d - 1 - t)
                loglik_tree = sum(float(e.loglik) for e in self.trees[t + 1].edges())
                loglik += float(loglik_tree)
                mbicv_tree = self._get_mbicv_of_tree(t, loglik_tree)
                mbicv_trunc += float(mbicv_tree)

                if (num_changed / num_total > 0.1):
                    num_changed = 0.0
                    if select_trunc_lvl and (mbicv_trunc >= mbicv) and (t > 0):
                        # mbicv didn't improve; remove trees if needed
                        loglik -= float(loglik_tree)
                        mbicv_trunc -= float(mbicv_tree)
                        while t > 1:
                            tprev = t - 1
                            ll_prev = sum(float(e.loglik) for e in self.trees[tprev + 1].edges())
                            mb_prev = self._get_mbicv_of_tree(tprev, ll_prev)
                            if mb_prev <= 0.0:
                                break
                            loglik -= float(ll_prev)
                            mbicv_trunc -= float(mb_prev)
                            t -= 1
                        trees_opt = [tr.copy_structure() for tr in self.trees]
                        loglik_opt = float(loglik)
                        trunc_opt = int(t)
                        thr_opt = float(self.controls.threshold)
                        if not select_threshold:
                            needs_break = True
                    else:
                        mbicv = float(mbicv_trunc)

                t += 1

            # check optimum
            if mbicv == 0.0:
                trees_opt = [tr.copy_structure() for tr in self.trees]
                loglik_opt = float(loglik)
                trunc_opt = int(self.controls.trunc_lvl) if self.controls.trunc_lvl is not None else (self.d - 1)
                thr_opt = float(self.controls.threshold)
                if not select_threshold:
                    needs_break = True
            elif mbicv >= mbicv_opt:
                needs_break = True
            else:
                trees_opt = [tr.copy_structure() for tr in self.trees]
                loglik_opt = float(loglik)
                trunc_opt = int(self.controls.trunc_lvl) if self.controls.trunc_lvl is not None else (self.d - 1)
                thr_opt = float(self.controls.threshold)
                mbicv_opt = float(mbicv)
                needs_break = needs_break or (not select_threshold)
                needs_break = needs_break or (float(self.controls.threshold) < 0.01)
                thresholded = self._thresholded_crits()

        # set final model
        self.trees = trees_opt if trees_opt else self.trees
        self.controls.threshold = float(thr_opt)
        trunc = max(0, min(self.d - 1, int(trunc_opt)))
        return self._finalize(trunc)
