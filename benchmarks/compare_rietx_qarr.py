#!/usr/bin/env python3
"""Compare bounded real QARR 1g workflows, disclosing failed equivalence gates.

rietx is an optional, version-pinned benchmark dependency. The unchanged
PhaseSmith Python-facade validation is the primary baseline. Both workflows pass its
profile/QPA quality thresholds; a stricter forward-equivalence check determines
whether an identical-model speed claim is allowed (currently it is not).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import phasesmith as ps
import rietx as rx
from compare_xrd_rust import measure_interleaved
from phasesmith.refinement import rietveld as rv
from phasesmith.validation import run_qarr_1g_validation, verify_validation_dataset
from phasesmith.validation.real_data import (
    _QARR_QPA_METADATA,
    QARR_1G_CUKA_FIXED_DISPERSION,
    QARR_1G_WEIGHED_WEIGHT_FRACTIONS,
    _qarr_displacement_defaults,
    _qarr_initial_scales,
    _qarr_instrument_values,
    _qarr_physics,
)
from pydantic import BaseModel
from rietx.schemas.instrument import (
    BackgroundChebyshev,
    BackgroundFixedPlusChebyshev,
    Dispersion,
)
from rietx.schemas.structure import AnisoU, PreferredOrientation

ROOT = Path(__file__).resolve().parents[1]
# A diagnostic gate, not a tolerance adjusted to admit this comparison.
FORWARD_RELATIVE_L2_LIMIT = 1e-3
RWP_DIFFERENCE_LIMIT = 0.005  # existing compare_gsasii_qarr.py criterion


def prepare(root: Path):
    """Map the existing validation's initial physical model without changing it."""
    p = rx.Parameter
    v = _qarr_instrument_values(root / "cuka.instprm")
    d = ps.read_powder_data(root / "cpd-1g.prn", format="columns")
    bg = ps.SmoothBrucknerBackground(smooth_width=1, iterations=50, chebyshev_order=None).estimate(
        d.x, d.observed_y
    )
    pattern = ps.PowderPattern(
        d.x,
        observed_y=d.observed_y,
        uncertainty=np.sqrt(np.maximum(d.observed_y, 1)),
        background=bg,
    )
    ins = ps.ConstantWavelengthInstrument(
        v["Lam1"],
        v["U"] * 1e-4,
        v["V"] * 1e-4,
        v["W"] * 1e-4,
        v["X"] * 0.01,
        v["Y"] * 0.01,
    )
    exp = ps.ConstantWavelengthExperiment.x_ray_components(
        ins,
        ps.WavelengthComponents.doublet(v["Lam1"], v["Lam2"], v["I(L2)/I(L1)"]),
        axial_geometry=ps.FcjGeometry(v["SH/L"] / 2, v["SH/L"] / 2),
    )
    phases, rph = [], []
    for name in QARR_1G_WEIGHED_WEIGHT_FRACTIONS:
        phase = rv.RietveldInput.from_cif(
            pattern,
            exp,
            root / (name + ".cif"),
            phase_id=name,
            selection=rv.RietveldParameterSelection(phase_scale=False, lattice=False),
            scattering=ps.XrayFixedDispersion(QARR_1G_CUKA_FIXED_DISPERSION),
            intensity_correction=ps.BraggBrentanoPolarizedLp(v["Lam1"], v["Polariz."]),
            coordinate_tolerance=1e-4,
        ).phases[0]
        st = _qarr_displacement_defaults(phase.structure)
        phases.append(replace(phase, structure=st, physics=_qarr_physics(name, st)))
        rp = rx.Structure.from_cif(str(root / (name + ".cif"))).phases[0]
        rp.name = name
        for atom, site in zip(rp.atoms, st.sites, strict=True):
            if atom.label != site.source_label:
                raise RuntimeError("CIF site identities differ")
            # rietx's default CIF import uses equivalent isotropic displacement.
            # Explicitly retain the fixed tensors of the PhaseSmith real case.
            if site.anisotropic_displacement is not None:
                u = site.anisotropic_displacement.u_cif_angstrom2
                atom.aniso = AnisoU(
                    **{
                        k: p(value=z)
                        for k, z in zip(("u11", "u22", "u33", "u23", "u13", "u12"), u, strict=True)
                    }
                )
            else:
                atom.biso.value = 8 * np.pi**2 * site.u_iso_angstrom2
        rp.lor_size.value = 180 / np.pi * 0.9 * v["Lam1"] / 1000
        rp.gauss_strain.value = 8 * np.log(2) * (2 * 180 / np.pi * 8e-4) ** 2
        if name != "CaF2":
            rp.preferred_orientation = PreferredOrientation(axis=(0, 0, 1))
        rph.append(rp)
    scales = _qarr_initial_scales(pattern, exp, phases)
    phases = tuple(replace(phase, scale=s) for phase, s in zip(phases, scales, strict=True))
    for rp, scale in zip(rph, scales, strict=True):
        rp.scale.value = scale
    rs = rx.Structure(phases=rph)
    ri = rx.Instrument.bragg_brentano()
    ri.source.lines[0].wavelength.value = v["Lam1"]
    ri.source.lines[1].wavelength.value = v["Lam2"]
    ratio = v["I(L2)/I(L1)"]
    ri.source.lines[0].weight.value = 1 / (1 + ratio)
    ri.source.lines[1].weight.value = ratio / (1 + ratio)
    ri.source.polarization.value = v["Polariz."]
    ri.source.dispersion = Dispersion(
        overrides={k: (z.real, z.imag) for k, z in QARR_1G_CUKA_FIXED_DISPERSION.items()}
    )
    ri.geometry.axial_sl.value = ri.geometry.axial_hl.value = v["SH/L"] / 2
    for k, z in {
        "u": ins.u_deg2 * 8 * np.log(2),
        "v": ins.v_deg2 * 8 * np.log(2),
        "w": ins.w_deg2 * 8 * np.log(2),
        "x": ins.x_deg,
        "y": ins.y_deg,
    }.items():
        getattr(ri.profile, k).value = z
    ri.background = BackgroundFixedPlusChebyshev(
        fixed_two_theta=d.x.tolist(),
        fixed_intensity=bg.tolist(),
        chebyshev=BackgroundChebyshev(coefficients=[p(value=0) for _ in range(3)]),
    )
    return pattern, exp, phases, rs, ri


def freeze(model):
    """Reset vary flags between explicitly noncumulative public fit stages."""
    if isinstance(model, rx.Parameter):
        model.vary = False
    elif isinstance(model, BaseModel):
        for key in type(model).model_fields:
            freeze(getattr(model, key))
    elif isinstance(model, list):
        for item in model:
            freeze(item)


def run_rietx(root: Path) -> dict:
    # Rebuild each run, including shared scale/background initialization and CIF
    # mapping. This conservatively includes PhaseSmith preparation in rietx time.
    pattern, _, _, structure, instrument = prepare(root)
    data = rx.PatternData(
        two_theta=pattern.x.tolist(),
        intensity=pattern.observed_y.tolist(),
        sigma=pattern.uncertainty.tolist(),
    )
    widths = [
        "instrument.profile.u",
        "instrument.profile.v",
        "instrument.profile.w",
        "instrument.zero_shift",
    ]
    stages = [
        (["phases.*.scale", *widths, "instrument.background.*"], 8),
        (
            [
                "phases.*.scale",
                *widths,
                "phases.*.atoms.*.biso",
                "phases.*.lor_size",
                "phases.*.gauss_strain",
                "phases.*.preferred_orientation.r",
            ],
            28,
        ),
        (["phases.*.scale"], 10),
    ]
    records = []
    for index, (free, budget) in enumerate(stages):
        freeze(structure)
        freeze(instrument)
        session = rx.Refinement(structure, instrument, history=False)
        result = session.fit(
            data, plan=rx.RefinementPlan([rx.Stage(f"stage{index + 1}", free, max_iter=budget)])
        )
        structure, instrument = session.fitted_structure, session.fitted_instrument
        records.append(
            {
                "termination": result.status,
                "rwp": result.statistics.rwp,
                "parameters": [item.path for item in result.parameters if item.vary],
                "stages": [stage.model_dump(mode="json") for stage in result.stages],
            }
        )
    observed = pattern.observed_y
    calc = np.asarray(result.y_calc)
    residual = calc - observed
    cheb = np.polynomial.chebyshev.chebval(
        2 * (pattern.x - pattern.x[0]) / np.ptp(pattern.x) - 1,
        [item.value for item in instrument.background.chebyshev.coefficients],
    )
    # Same fixed Z and molar masses as the existing PhaseSmith acceptance case.
    quantities = [
        ps.QuantitativePhase(
            phase.name,
            phase.scale.value,
            *_QARR_QPA_METADATA[phase.name],
            next(q.cell_volume for q in result.qpa.phases if q.name == phase.name),
        )
        for phase in structure.phases
    ]
    fractions = {q.phase_id: q.weight_fraction for q in ps.quantitative_phase_analysis(quantities)}
    record = {
        "poisson_rwp": float(
            np.linalg.norm(residual / pattern.uncertainty)
            / np.linalg.norm(observed / pattern.uncertainty)
        ),
        "unit_weight_rwp": float(np.linalg.norm(residual) / np.linalg.norm(observed)),
        "profile_correlation": float(
            np.corrcoef(
                observed - pattern.background,
                calc - pattern.background - cheb,
            )[0, 1]
        ),
        "weight_fractions": fractions,
        "maximum_weight_fraction_error": max(
            abs(fractions[k] - v) for k, v in QARR_1G_WEIGHED_WEIGHT_FRACTIONS.items()
        ),
        "qpa_covariance_available": all(
            q.weight_fraction_stderr is not None
            and np.isfinite(q.weight_fraction_stderr)
            and q.weight_fraction_stderr >= 0
            for q in result.qpa.phases
        ),
        "stages": records,
    }
    json.dumps(record, allow_nan=False)
    if not (
        record["poisson_rwp"] <= 0.20
        and record["unit_weight_rwp"] <= 0.15
        and record["profile_correlation"] >= 0.98
        and record["maximum_weight_fraction_error"] <= 0.02
        and record["qpa_covariance_available"]
        and all(s["termination"] == "converged" for s in records)
    ):
        raise RuntimeError(f"rietx real-data quality gate failed: {record}")
    return record


def run(
    root: Path, threads: int, repetitions: int, profile_accuracy: ps.ProfileAccuracy | None = None
) -> dict:
    from rietx._about import COMPILED_THREADS_ENV
    from rietx.model import compiled

    if rx.__version__ != "1.4.0" or ps._core.BUILD_MODE != "release":
        raise RuntimeError("requires rietx 1.4.0 and a release PhaseSmith build")
    paths = verify_validation_dataset("iucr-qarr-1g", root)
    os.environ[COMPILED_THREADS_ENV] = str(threads)
    compiled.warm(block=True)
    if not compiled.enabled():
        raise RuntimeError("rietx compiled kernels unavailable")
    execution = ps.ExecutionPolicy(threads=threads)
    original_calculate = rv.calculate

    def controlled_calculate(*args, **kwargs):
        # The validation's initial-scale helper otherwise uses the default two
        # workers, including inside the rietx workflow's shared preparation.
        kwargs.setdefault("execution", execution)
        return original_calculate(*args, **kwargs)

    with patch.object(rv, "calculate", controlled_calculate):
        pattern, experiment, phases, structure, instrument = prepare(root)
    accuracy = profile_accuracy or ps.ProfileAccuracy()
    py = rv.calculate(
        pattern,
        experiment,
        phases,
        support_fwhm=30,
        profile_accuracy=accuracy,
        execution=execution,
    ).profile_y
    ry = rx.Refinement(structure, instrument, history=False).predict(pattern.x) - pattern.background
    forward_delta = float(np.linalg.norm(py - ry) / np.linalg.norm(py))
    records = {"phasesmith": [], "rietx": []}

    def native():
        original = rv.refine

        def refine(request, options, **kwargs):
            return original(request, replace(options, profile_accuracy=accuracy), **kwargs)

        with (
            patch.object(rv, "refine", refine),
            patch.object(rv, "calculate", controlled_calculate),
        ):
            result = run_qarr_1g_validation(root, execution=execution)
        if result.status != "passed":
            raise RuntimeError("PhaseSmith real-data validation failed")
        record = result.to_record()
        record.pop("elapsed_seconds")
        records["phasesmith"].append(record)
        return record

    def external():
        with patch.object(rv, "calculate", controlled_calculate):
            record = run_rietx(root)
        records["rietx"].append(record)
        return record

    last, timings = measure_interleaved({"phasesmith": native, "rietx": external}, 1, repetitions)
    for name, runs in records.items():
        if any(record != runs[0] for record in runs[1:]):
            raise RuntimeError(f"{name} scientific results are not deterministic")
    ps_rwp = next(
        c["measured"] for c in last["phasesmith"]["checks"] if c["check_id"] == "poisson_rwp"
    )
    rwp_delta = abs(ps_rwp - last["rietx"]["poisson_rwp"])
    return {
        "schema": "phasesmith.rietx-qarr-benchmark.v1",
        "scope": "real_qarr_1g_bounded_workflows_with_quality_gates",
        "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "samples": pattern.x.size,
        "phases": 3,
        "reflections": sum(p.reflections.reflection_count for p in phases),
        "threads": threads,
        "shared_preparation_threads": threads,
        "phasesmith_refinement_backend": "native_fixed_spectrum",
        "phasesmith_profile_accuracy": asdict(accuracy),
        "phasesmith_powder_intensity_convention": "friedel_pair_average",
        "warmups": 1,
        "repetitions": repetitions,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "phasesmith_build": ps._core.BUILD_MODE,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("phasesmith", "rietx", "numpy", "scipy", "numba")
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    COMPILED_THREADS_ENV,
                )
            },
        },
        "quality_gates_passed": True,
        "equivalence": {
            "forward_relative_l2": forward_delta,
            "forward_relative_l2_limit": FORWARD_RELATIVE_L2_LIMIT,
            "poisson_rwp_difference": rwp_delta,
            "poisson_rwp_difference_limit": RWP_DIFFERENCE_LIMIT,
            "passed": forward_delta <= FORWARD_RELATIVE_L2_LIMIT
            and rwp_delta <= RWP_DIFFERENCE_LIMIT,
        },
        "notes": [
            "Thread counts are requested worker budgets, not measured active cores; "
            "rietx may execute serially below its parallel row threshold.",
            "Shared PhaseSmith preparation honors the requested worker count. "
            "Earlier records without shared_preparation_threads used the default two workers.",
            "Timings describe bounded workflows at accepted real-data quality, not identical math.",
            "No speed ratio is emitted when forward/fit equivalence fails.",
            "Python validation orchestrates three fits; each compatible fixed-spectrum "
            "fit dispatches to the native Rust solver.",
            "PhaseSmith validation recipe: fixed Bruckner + 3 Chebyshev, doublet, FCJ, "
            "fixed CIF anisotropic displacement, 3 phases, size/strain/texture and isotropic ADPs.",
            "rietx receives explicit fixed anisotropic tensors, normalized line weights, "
            "Gaussian FWHM squared = 8 ln(2) times PhaseSmith variance, Biso = 8 pi^2 Uiso.",
            "rietx size coefficient uses the primary wavelength; "
            "PhaseSmith size is component-aware.",
            "Default support differs: PhaseSmith 30 FWHM, "
            "rietx area-tail windows frozen per stage.",
            "Explicit PhaseSmith profile_accuracy overrides apply to refinement and the "
            "forward comparison; shared input/initial-scale preparation stays unchanged.",
            "Budgets 8/28/10 and parameter roles match; stopping rules and bounds do not.",
            "rietx public fits calculate covariance each stage; PhaseSmith only at scale polish.",
            "Both rebuild inputs each run; rietx time includes shared PhaseSmith initialization "
            "and model mapping. Imports/JIT warmup/checksums excluded. History/stage reports off.",
        ],
        "scientific_results": last,
        "timings": timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory", type=Path, default=ROOT / "validation/data/iucr-qarr-1g"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--fast-fcj", action="store_true")
    parser.add_argument("--tail-area-tolerance", type=float)
    args = parser.parse_args()
    if args.threads <= 0 or args.repetitions <= 0:
        parser.error("threads and repetitions must be positive")
    accuracy = ps.ProfileAccuracy(
        fast_fcj=args.fast_fcj, tail_area_tolerance=args.tail_area_tolerance
    )
    record = run(args.data_directory, args.threads, args.repetitions, accuracy)
    args.json_output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: record[k] for k in ("quality_gates_passed", "equivalence", "timings")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
