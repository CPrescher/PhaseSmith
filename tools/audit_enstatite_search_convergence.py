"""Post-search common-objective and feasible-convergence diagnostics.

Reads only saved experimental reductions and PhaseSmith candidates, never the
source HKL/CIF or atomic reference. SciPy is an optional diagnostic optimizer;
all production equations/controls and frozen raw-workflow results are unchanged.
"""

import json
from pathlib import Path

import numpy as np
from phasesmith import _core
from phasesmith.crystallography import UnitCell
from phasesmith.io.space_groups import space_group_by_number

CELL = UnitCell(8.8101, 5.1824, 18.2288, 90, 90, 90)
GROUP = space_group_by_number(61).space_group


def merge(root):
    z = np.load(root / "reduced-observations.npz", allow_pickle=False)
    use = z["use"]
    h = z["hkl"][use]
    y = z["intensity"][use]
    s = z["sigma"][use]
    valid = ~GROUP.systematic_absences(h)
    # Pbca point rotations only change signs; abs(hkl) identifies its Friedel families.
    family = {}
    for hv, yi, si, keep in zip(h, y, s, valid, strict=True):
        if keep:
            family.setdefault(tuple(abs(hv)), []).append((float(yi), float(si)))
    amplitudes = {}
    for key, rows in family.items():
        minimum = min(v for _, v in rows)
        weights = [(minimum / v) ** 2 for _, v in rows]
        total = 0.0
        for w in weights:
            total += w
        intensity = 0.0
        for (i, _), w in zip(rows, weights, strict=True):
            intensity += i * (w / total)
        amplitudes[key] = np.sqrt(max(0.0, intensity))
    plan = _core._PhaseQualityPlan(
        list(CELL.as_tuple()),
        GROUP._native,
        np.ascontiguousarray(h).ravel(),
        np.ascontiguousarray(y),
        np.ascontiguousarray(s),
        ["Mg", "Si", "O"],
    )
    hk = np.asarray(plan.metadata()["hkl"], np.int64)
    return hk, np.array([amplitudes[tuple(abs(hv))] for hv in hk])


def model(candidate, hkl, fo):
    c = candidate["refinement"]
    points = []
    species = []
    tensors = []
    direct = CELL.geometry().direct_basis
    for element, point, u in zip(c["species"], c["fractional_xyz"], c["u_cartesian"], strict=True):
        for op in GROUP.operations:
            points.append(op.apply_fractional(point))
            species.append(element)
            rotation = direct @ op.rotation @ np.linalg.inv(direct)
            tensors.append(rotation @ np.array(u) @ rotation.T)
    native = _core._SymmetricSolutionModel(
        list(CELL.as_tuple()),
        GROUP._native,
        species,
        np.ascontiguousarray(points).ravel(),
        np.ascontiguousarray(hkl).ravel(),
        np.ascontiguousarray(fo),
        "anisotropic",
        1e-5,
        True,
        True,
    )
    meta = native.metadata()
    seed = []
    for site in meta["sites"]:
        image = site["images"][0]
        r = np.array(image["cartesian_rotation"])
        seed.append(r.T @ tensors[image["source_index"]] @ r)
    p = native.initial_tensors(np.ascontiguousarray(seed).ravel())
    return native, meta, p


def evaluate(native, p):
    e = native.evaluate(p)
    j = e["jacobian"].reshape(len(p), -1)
    g = 2 * j @ e["residual"]
    return e, dict(
        r1=e["r1"],
        objective=e["objective"],
        amplitude_scale=e["amplitude_scale"],
        gradient_inf=float(abs(g).max()),
        min_u_eigenvalue=float(np.linalg.eigvalsh(e["u_cartesian"]).min()),
    )


def replay(root, output):
    import hashlib
    import time
    from dataclasses import asdict

    from phasesmith.experimental.solution import (
        SolutionOptions,
        solve_structure_elements,
        write_solution,
    )
    from phasesmith.io.hkl import HklData

    config = json.loads((root / "search-input.json").read_text())
    z = np.load(root / "reduced-observations.npz", allow_pickle=False)
    use = z["use"]
    data = HklData(z["hkl"][use], z["intensity"][use], z["sigma"][use])
    options = SolutionOptions(**config["options"])
    start = time.perf_counter()
    result = solve_structure_elements(
        data,
        cell=CELL,
        space_group=GROUP,
        elements=config["priors"]["elements"],
        maximum_atoms=config["resource"]["maximum_atoms"],
        options=options,
        **config["controls"],
    )
    seconds = time.perf_counter() - start
    write_solution(result, output / "candidate")
    saved = json.loads((output / "candidate/result.json").read_text())
    original = json.loads((root / "candidate/result.json").read_text())
    payload = {
        key: saved[key]
        for key in ("candidates", "phasing_histories", "termination", "r1_target_met")
    }
    record = dict(
        input=str(root),
        seconds=seconds,
        options=asdict(options),
        numerical_result_sha256=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        candidates_identical_to_original=payload["candidates"] == original["candidates"],
        phasing_histories_identical_to_original=payload["phasing_histories"]
        == original["phasing_histories"],
        r1=saved["candidates"][0]["r1"],
        native_module_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
    )
    (output / "replay.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def cross_fit(baseline, current, output):
    from scipy.optimize import least_squares

    roots = dict(previous=baseline, native=current)
    candidates = {
        key: json.loads((root / "candidate/result.json").read_text())["candidates"][0]
        for key, root in roots.items()
    }
    records = []
    for data, root in roots.items():
        hkl, fo = merge(root)
        for start, candidate in candidates.items():
            native, meta, p = model(candidate, hkl, fo)
            _, initial = evaluate(native, p)
            if data == start:
                np.testing.assert_allclose(initial["r1"], candidate["r1"], atol=2e-13, rtol=0)
                np.testing.assert_allclose(
                    initial["objective"], candidate["objective"], atol=2e-13, rtol=0
                )
            entry = dict(data=data, start=start, original=initial, continued={})
            for solver, cached in [("cached", True), ("dense", False)]:
                fit = native.refine(p, 200, 60.0, None, cached, cached, cached, cached, False, 1)
                q = fit["parameters"]
                e, stopped = evaluate(native, q)
                ids = np.array(
                    [
                        meta["offsets"][i][0] + j
                        for i, site in enumerate(meta["sites"])
                        for j in range(len(site["coordinate_basis"]))
                    ]
                )
                jacobian = e["jacobian"].reshape(len(q), -1)
                gradient = 2 * jacobian @ e["residual"]
                stopped.update(
                    termination=fit["termination"],
                    accepted_steps=fit["accepted_steps"],
                    coordinate_gradient_inf=float(abs(gradient[ids]).max()),
                )
                if solver == "cached":
                    checked = []
                    for i in ids[::6]:
                        d = np.zeros(len(q))
                        d[i] = 1e-6
                        fd = (
                            native.evaluate(q + d)["objective"]
                            - native.evaluate(q - d)["objective"]
                        ) / 2e-6
                        checked.append(
                            dict(
                                parameter=int(i),
                                analytical=float(gradient[i]),
                                finite_difference=float(fd),
                            )
                        )
                        np.testing.assert_allclose(fd, gradient[i], atol=1e-8, rtol=1e-4)
                    template = q.copy()
                    last = None
                    last_e = None

                    def coordinate_evaluation(x, template=template, ids=ids, native=native):
                        nonlocal last, last_e
                        if last is None or not np.array_equal(x, last):
                            value = template.copy()
                            value[ids] = x
                            last = x.copy()
                            last_e = native.evaluate(value)
                        return last_e

                    count = len(q)
                    control = least_squares(
                        lambda x: coordinate_evaluation(x)["residual"],
                        q[ids],
                        jac=lambda x, evaluate=coordinate_evaluation, count=count, ids=ids: (
                            evaluate(x)["jacobian"].reshape(count, -1)[ids].T
                        ),
                        x_scale="jac",
                        ftol=1e-13,
                        gtol=1e-10,
                        xtol=1e-13,
                        max_nfev=300,
                    )
                    value = template.copy()
                    value[ids] = control.x
                    _, coordinate_only = evaluate(native, value)
                    coordinate_only.update(
                        success=bool(control.success),
                        evaluations=control.nfev,
                        coordinate_gradient_inf=float(control.optimality * 2),
                        termination=str(control.message),
                        tensors_fixed=True,
                        interpretation=(
                            "Independent SciPy optimizer on the same native objective; all ADPs "
                            "frozen at the feasible stopped values. This checks optimizer "
                            "convergence, not an independent physics implementation."
                        ),
                    )
                    assert coordinate_only["objective"] < stopped["objective"]
                    stopped["coordinate_only_control"] = coordinate_only
                    stopped["finite_difference_coordinate_gradient_checks"] = checked
                entry["continued"][solver] = stopped
            records.append(entry)
            print(json.dumps(entry), flush=True)
    record = dict(
        schema="phasesmith.enstatite-search-common-objective.v1",
        raw_frames_read=False,
        atomic_reference_read=False,
        production_controls_changed=False,
        prepared_merging=(
            "Pbca Friedel families identified by absolute HKLs; native systematic "
            "absences/expanded order; source-order normalized inverse-variance means. "
            "Original candidate scores reproduced within 2e-13."
        ),
        continuation_iterations=200,
        continuation_max_seconds=60,
        coordinate_only_max_evaluations=300,
        cases=records,
    )
    with (output / "cross-fit.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.replay:
        replay(args.baseline, args.output)
    else:
        if args.current is None:
            parser.error("--current required for cross fitting")
        cross_fit(args.baseline, args.current, args.output)


if __name__ == "__main__":
    main()
