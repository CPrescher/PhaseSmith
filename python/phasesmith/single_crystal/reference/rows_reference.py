"""Small-case independent NumPy mathematical reference (not a DAFi source port).
Algorithm credit: Aslandukov et al. (2022), doi:10.1107/S1600576722008081.
Deliberately readable/exhaustive; not used to time real datasets.
"""

from collections import defaultdict
from itertools import pairwise

import numpy as np

ROOT = np.sqrt(0.5)


def key(v, eps):
    norm = np.linalg.norm(v)
    if norm <= 1e-15:
        return None
    k = int(np.argmax(np.abs(v)))
    v = v / norm * (1 if v[k] >= 0 else -1)
    i, j = [i for i in range(3) if i != k]
    side = int(np.floor(2 * ROOT / eps)) + 1
    a, b = np.floor((v[[i, j]] + ROOT) / eps).astype(int)
    return (k * side + max(a, 0)) * side + max(b, 0)


def decode(code, eps):
    side = int(np.floor(2 * ROOT / eps)) + 1
    k = code // side**2
    a = (code // side) % side
    b = code % side
    i, j = [i for i in range(3) if i != k]
    v = np.zeros(3)
    v[[i, j]] = (np.array([a, b]) + 0.5) * eps - ROOT
    v[k] = np.sqrt(max(0, 1 - v[i] ** 2 - v[j] ** 2))
    return v


def interval_support(ds, o):
    ranges = [
        [
            (d / (m + o["relative_epsilon"]), d / (m - o["relative_epsilon"]))
            for m in range(1, o["kmax"] + 1)
        ]
        for d in ds
    ]
    return max(
        (
            sum(any(lo <= x <= hi for lo, hi in rr) for rr in ranges)
            for x in [lo for rr in ranges for lo, hi in rr]
        ),
        default=0,
    )


def phase_subset(row, d, eps):
    # Independent exhaustive endpoint formulation; ties choose earliest end,
    # then earliest feasible start, matching the documented numerical convention.
    phase = sorted([(t % d, int(np.floor(t / d)), i) for t, i in row], key=lambda z: (z[0], z[2]))
    n = len(phase)
    phase += [(r + d, s - 1, i) for r, s, i in phase]
    best = []
    for r in range(2 * n):
        feasible = [
            left
            for left in range(min(r + 1, n))
            if r - left + 1 <= n and phase[r][0] - phase[left][0] <= d * eps
        ]
        if not feasible:
            continue
        left = min(feasible)
        chosen = {}
        for _, site, i in phase[left : r + 1]:
            chosen[site] = min(i, chosen.get(site, i))
        if len(chosen) > len(best):
            best = sorted(chosen.values())
    return best


def projected_grids(p, ids, v, o):
    k = int(np.argmax(abs(v)))
    ij = [i for i in range(3) if i != k]
    keep = min(
        o["min_row"],
        o.get("phase_min_row", 0) or o["min_row"],
        3 if o.get("extend_best", False) else o["min_row"],
        o.get("gap_min_row", 0) or o["min_row"],
    )
    if o.get("projection_grids", 2):
        grids = []
        for ox, oy in [(0, 0), (1, 1), (1, 0), (0, 1)][: o.get("projection_grids", 2)]:
            rows = defaultdict(list)
            for i in ids:
                t = p[i, k] / v[k]
                proj = p[i] - v * t
                cell = tuple(
                    np.floor(
                        (proj[ij] + np.array([ox, oy]) * o["projection_epsilon"])
                        / (2 * o["projection_epsilon"])
                    ).astype(int)
                )
                rows[cell].append((t, i))
            grids.append([sorted(rows[c]) for c in sorted(rows) if len(rows[c]) >= keep])
        return grids
    cells = defaultdict(list)
    for i in ids:
        t = p[i, k] / v[k]
        proj = p[i] - v * t
        cell = tuple(np.floor(proj[ij] / o["projection_epsilon"]).astype(int))
        cells[cell].append((t, i))
    remaining = set(cells)
    components = []
    while remaining:
        todo = [min(remaining)]
        remaining.remove(todo[0])
        component = []
        while todo:
            c = todo.pop()
            component.append(c)
            adjacent = [d for d in remaining if max(abs(c[0] - d[0]), abs(c[1] - d[1])) <= 1]
            for d in adjacent:
                remaining.remove(d)
                todo.append(d)
        components.append(sorted(component))
    grids = [[], []]
    for comp in components:
        if sum(len(cells[c]) for c in comp) < keep:
            continue
        if (
            max(c[0] for c in comp) - min(c[0] for c in comp) <= 1
            and max(c[1] for c in comp) - min(c[1] for c in comp) <= 1
        ):
            row = sorted(item for c in comp for item in cells[c])
            grids[0].append(row)
            grids[1].append(row)
            continue
        for pass_index in [0, 1]:
            local = {(c if pass_index == 0 else (c[1], c[0])): cells[c] for c in comp}
            unused = set(local)
            for x, y in sorted(local):
                if (x, y) not in unused:
                    continue
                chosen = []
                count = 0
                for low in [y - 1, y]:
                    block = [(a, b) for a in [x, x + 1] for b in [low, low + 1] if (a, b) in unused]
                    n = sum(len(local[c]) for c in block)
                    if n > count:
                        chosen = block
                        count = n
                if count < keep:
                    unused.remove((x, y))
                    continue
                grids[pass_index].append(sorted(item for c in chosen for item in local[c]))
                unused.difference_update(chosen)
    return grids


def direction_details(p, ids, v, o):
    best = []
    details = ([], None, 0, [])
    for gi, allrows in enumerate(projected_grids(p, ids, v, o)):
        rows = allrows
        gaps = sorted(
            t2 - t1
            for row in rows
            if len(row) >= (o.get("gap_min_row", 0) or o["min_row"])
            for (t1, _), (t2, _) in pairwise(row)
            if t2 - t1 > 1e-12
        )
        if not gaps:
            continue
        chosen = []
        for r in range(len(gaps)):
            window = [
                x
                for x in gaps[: r + 1]
                if gaps[r] - x
                <= (
                    o["gap_epsilon"] * (gaps[r] + x)
                    if o.get("relative_gap", False)
                    else 2 * o["gap_epsilon"]
                )
            ]
            if len(window) > len(chosen):
                chosen = window
        d = (chosen[0] + chosen[-1]) / 2
        if d <= 1e-12:
            continue
        group = []
        for row in rows:
            sub = phase_subset(
                row, d, o["relative_epsilon"] * (2 if o.get("phase_radius", False) else 1)
            )
            if len(sub) >= (o.get("phase_min_row", 0) or o["min_row"]):
                group += sub
        if len(group) > len(best):
            best = sorted(group)
            details = (best, d, gi, allrows)
    return details


def direction_group(p, ids, v, o):
    return direction_details(p, ids, v, o)[0]


def extend_direction(p, ids, v, o):
    keep = {**o, "extend_best": True}
    base, d, _grid, rows = direction_details(p, ids, v, keep)
    if d is None:
        return base
    extended = []
    for row in rows:
        sub = phase_subset(
            row, d, o["relative_epsilon"] * (2 if o.get("phase_radius", False) else 1)
        )
        if len(sub) >= 3:
            extended += sub
    return sorted(extended) if len(extended) > len(base) else base


def run(q, o):
    q = np.asarray(q)
    order = sorted(range(len(q)), key=lambda i: (*q[i], i))
    unique = []
    for i in order:
        if not unique or not np.array_equal(q[i], q[unique[-1]]):
            unique.append(i)
    if not unique:
        return np.zeros(len(q), int)
    o = dict(o)
    if o.get("naive_max_points", 0) > 0 and len(unique) <= o["naive_max_points"]:
        o["smart"] = False
    raw = q[unique]
    mean = np.array([sum(x / len(raw) for x in raw[:, k]) for k in range(3)])
    scale = np.max(abs(raw - mean), axis=0)
    scale[scale == 0] = 1
    p = (raw - mean) / scale
    hist = defaultdict(int)
    for c in range(min(len(p), o.get("max_centers", len(p)))):
        rows = defaultdict(list)
        for j in range(len(p)):
            if c == j:
                continue
            v = p[j] - p[c]
            code = key(v, o["direction_epsilon"])
            if code is None:
                continue
            if not o["smart"]:
                if j > c:
                    hist[code] += 1
            else:
                rows[code].append(np.linalg.norm(v))
        if o["smart"]:
            for code, ds in rows.items():
                if len(ds) + 1 < o["min_row"]:
                    continue
                support = interval_support(ds, o) + 1
                if support >= o["min_row"]:
                    hist[code] += support
    ranked = sorted(hist, key=lambda k: (-hist[k], k))[: o["directions"]]
    ids = list(range(len(p)))
    groups = []
    while len(groups) < o["groups"] and len(ids) >= o["min_row"]:
        candidates = [
            (rank, direction_group(p, ids, decode(code, o["direction_epsilon"]), o))
            for rank, code in enumerate(ranked)
        ]
        candidates = [(rank, g) for rank, g in candidates if len(g) >= o["min_row"]]
        if not candidates:
            break
        removed = set()
        for batch_index in range(min(o["batch"], o["groups"] - len(groups))):
            candidates = [(rank, [i for i in g if i not in removed]) for rank, g in candidates]
            candidates.sort(key=lambda z: (-len(z[1]), z[0]))
            if len(candidates[0][1]) < o["min_row"]:
                break
            group = candidates[0][1]
            if o.get("extend_best", False) and batch_index == 0:
                extra = extend_direction(
                    p, ids, decode(ranked[candidates[0][0]], o["direction_epsilon"]), o
                )
                if len(extra) > len(group):
                    group = extra
            groups.append(group)
            removed.update(group)
        ids = [i for i in ids if i not in removed]
    labels = np.zeros(len(q), int)
    for g, ids in enumerate(groups, 1):
        labels[np.array(unique)[ids]] = g
    return labels
