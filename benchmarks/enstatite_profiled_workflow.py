"""Observe the unchanged frozen raw workflow with nested wall-clock timing.

All data and products use the ordinary driver's external paths. Instrumentation
wraps existing call boundaries without changing arguments, arrays or numerical
controls. Nested inclusive times must not be added; exclusive seconds partition
the observed main-thread wall time. Frame I/O/native worker time is included in
its owning stage. Search subprocess startup is distinguished from solver time.
"""

import functools
import importlib
import json
import resource
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import enstatite_raw_workflow as workflow


class Timings:
    def __init__(self):
        self.started = time.perf_counter()
        self.stack = []
        self.events = []

    def wrap(self, name, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            started = time.perf_counter()
            state = dict(name=name, start_seconds=started - self.started, child_seconds=0.0)
            self.stack.append(state)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                self.stack.pop()
                state.update(
                    inclusive_seconds=elapsed,
                    exclusive_seconds=elapsed - state.pop("child_seconds"),
                )
                self.events.append(state)
                if self.stack:
                    self.stack[-1]["child_seconds"] += elapsed

        return call

    def patch(self, module, attribute, name):
        setattr(module, attribute, self.wrap(name, getattr(module, attribute)))

    def record(self):
        elapsed = time.perf_counter() - self.started
        categories = {}
        for event in self.events:
            categories[event["name"]] = (
                categories.get(event["name"], 0.0) + event["exclusive_seconds"]
            )
        return dict(
            schema="phasesmith.enstatite-wall-timings.v1",
            wall_seconds=elapsed,
            maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            accounted_exclusive_seconds=sum(categories.values()),
            exclusive_by_name=categories,
            events=self.events,
            interpretation=(
                "Nested inclusive times overlap; exclusive seconds partition each "
                "process's recorded wall time. Native parallel tasks and original-count "
                "I/O are included in the owning function. Timing observer changes "
                "no numerical arguments."
            ),
        )


def main():
    timer = Timings()
    worker = "--worker" in sys.argv
    search = "--search-only" in sys.argv
    if worker or search:
        option = "--worker" if worker else "--search-only"
        packet = json.loads(Path(sys.argv[sys.argv.index(option) + 1]).read_text())
        output = Path(packet["output"])
    else:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
    for name in (
        "stage_validity",
        "stage_positions",
        "stage_indexing",
        "stage_integrate",
        "run_search",
    ):
        timer.patch(workflow, name, name)
    if worker:
        targets = [
            ("frame_validity", "estimate_seat_shadows", "shadow_estimation"),
            ("rotation", "locate_series", "detection_and_linking"),
            ("grains", "find_grains", "grain_indexing"),
            ("geometry_refinement", "refine_geometry", "geometry_fit"),
            ("profiles", "prepare_profiles", "pixel_collection_and_background"),
            ("profiles", "learn_profiles", "profile_learning"),
            ("reduction_io", "save_case", "observation_archive"),
            ("reduction", "reduce_observations", "reduction_other"),
            ("reduction", "refine_profiles", "profile_fitting"),
            ("reduction", "physical_factors", "physical_corrections"),
            ("reduction", "empirical_design", "empirical_design"),
            ("empirical_scaling", "fit_empirical_scale", "empirical_scaling"),
            ("intensity_uncertainty", "estimate_equivalent_error", "equivalent_error_model"),
        ]
        for module, attribute, name in targets:
            timer.patch(
                importlib.import_module("phasesmith.experimental." + module), attribute, name
            )
        import numpy as np

        timer.patch(np, "savez_compressed", "compressed_archives")
        try:
            archive = importlib.import_module("phasesmith.io.numeric_archive")
        except ModuleNotFoundError:
            pass  # Retained baseline wheels predate the native writer.
        else:
            timer.patch(archive, "save_npz", "compressed_archives")
    if not worker and not search:
        timer.patch(workflow, "verify_dataset", "source_verification")
    original_run = subprocess.run

    @functools.wraps(original_run)
    def run(command, *args, **kwargs):
        # Ensure each numerical child also records its own nested timing.
        if len(command) > 2 and Path(command[1]).name == "enstatite_raw_workflow.py":
            command = list(command)
            command[1] = str(Path(__file__).resolve())
        return timer.wrap("search_process" if worker else "worker_process", original_run)(
            command, *args, **kwargs
        )

    subprocess.run = run
    from detection_timing import observe_detection

    detection_stats = None
    try:
        with observe_detection(workflow) if worker else nullcontext() as detection_stats:
            return timer.wrap("process_main", workflow.main)()
    finally:
        if detection_stats is not None:
            with (output / "timings-detection.json").open("x") as stream:
                json.dump(detection_stats, stream, indent=2, allow_nan=False)
                stream.write("\n")
        with (
            output
            / (
                "timings-worker.json"
                if worker
                else "timings-search.json"
                if search
                else "timings-parent.json"
            )
        ).open("x") as stream:
            json.dump(timer.record(), stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
