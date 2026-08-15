"""Finite JSON command-line boundary for PhaseSmith automation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO, cast

from .automation import (
    AutomationError,
    inspect_cif_file,
    inspect_powder_file,
    load_recipe_proposal,
    load_workflow_spec,
    plan_workflow,
    recipe_proposal_schema,
    resume_workflow,
    run_workflow,
    workflow_spec_schema,
    write_workflow_plan,
)
from .io.powder import PowderFormat


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise AutomationError("cli.arguments", message)


def _parser() -> _JsonArgumentParser:
    parser = _JsonArgumentParser(
        prog="phasesmith",
        description="Inspect, plan, approve, run, and resume bounded PhaseSmith tasks.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    schema = commands.add_parser("schema", help="print an automation JSON Schema")
    schema.add_argument("contract", choices=("workflow", "recipe"))

    pattern = commands.add_parser(
        "inspect-pattern",
        help="inspect a powder file without inferring experiment physics",
    )
    pattern.add_argument("path")
    pattern.add_argument(
        "--format",
        choices=("auto", "columns", "gsas_fxye", "gsas_std"),
        default="auto",
    )
    pattern.add_argument("--bank", type=int, default=1)

    cif = commands.add_parser(
        "inspect-cif",
        help="inspect a CIF without constructing or running a refinement",
    )
    cif.add_argument("path")
    cif.add_argument("--block")
    cif.add_argument("--permissive", action="store_true")

    plan = commands.add_parser("plan", help="create a read-only byte-bound workflow plan")
    plan.add_argument("spec")
    plan.add_argument("--output")
    plan.add_argument("--overwrite", action="store_true")

    run = commands.add_parser("run", help="execute an explicitly approved workflow plan")
    run.add_argument("spec")
    run.add_argument("--approve", required=True, metavar="PLAN_ID")
    run.add_argument("--proposal", help="validated human- or AI-authored recipe JSON")
    run.add_argument("--overwrite", action="store_true")

    resume = commands.add_parser("resume", help="continue an explicitly approved checkpoint")
    resume.add_argument("spec")
    resume.add_argument("--approve", required=True, metavar="PLAN_ID")
    resume.add_argument("--overwrite", action="store_true")

    report = commands.add_parser("report", help="print a completed automation audit record")
    report.add_argument("path", help="an output directory or terminal result JSON file")
    return parser


def _emit(record: object, stream: TextIO) -> None:
    stream.write(json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _report_record(path: str | Path) -> dict[str, object]:
    source = Path(path).resolve()
    if source.is_dir():
        candidates = [
            item
            for item in (source / "workflow-result.json", source / "resume-result.json")
            if item.is_file()
        ]
        if len(candidates) != 1:
            raise AutomationError(
                "report.ambiguous",
                "output directory must contain exactly one terminal automation result",
                path=str(source),
            )
        source = candidates[0]
    try:
        if source.stat().st_size > 16 * 1024 * 1024:
            raise AutomationError(
                "report.too_large",
                "automation report exceeds the 16 MiB read limit",
                path=str(source),
            )
        value = json.loads(source.read_text(encoding="utf-8"))
    except AutomationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AutomationError(
            "report.invalid",
            "could not read the automation result",
            path=str(source),
            reason=str(error),
        ) from error
    if not isinstance(value, dict) or value.get("schema") not in {
        "phasesmith.workflow-result.v1",
        "phasesmith.resume-result.v1",
    }:
        raise AutomationError(
            "report.schema",
            "file is not a supported terminal automation result",
            path=str(source),
        )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""

    try:
        exit_status = 0
        args = _parser().parse_args(argv)
        if args.command == "schema":
            result = (
                workflow_spec_schema() if args.contract == "workflow" else recipe_proposal_schema()
            )
        elif args.command == "inspect-pattern":
            result = inspect_powder_file(
                args.path,
                format=cast(PowderFormat, args.format),
                bank=args.bank,
            )
        elif args.command == "inspect-cif":
            result = inspect_cif_file(
                args.path,
                block=args.block,
                strict=not args.permissive,
            )
        elif args.command == "plan":
            result = plan_workflow(load_workflow_spec(args.spec))
            if args.output is not None:
                write_workflow_plan(result, args.output, overwrite=args.overwrite)
            result = result.to_record()
        elif args.command == "run":
            plan = plan_workflow(load_workflow_spec(args.spec))
            proposal = None if args.proposal is None else load_recipe_proposal(args.proposal, plan)
            run_result = run_workflow(
                plan,
                approval_plan_id=args.approve,
                proposal=proposal,
                overwrite=args.overwrite,
            )
            result = run_result.to_record()
            exit_status = 0 if run_result.workflow.completed else 3
        elif args.command == "resume":
            plan = plan_workflow(load_workflow_spec(args.spec))
            resume_result = resume_workflow(
                plan,
                approval_plan_id=args.approve,
                overwrite=args.overwrite,
            )
            result = resume_result.to_record()
            exit_status = (
                0
                if resume_result.result.termination_reason.value in {"converged", "stagnated"}
                else 3
            )
        else:
            result = _report_record(args.path)
    except AutomationError as error:
        _emit(error.to_record(), sys.stderr)
        return 2
    except Exception as error:
        failure = AutomationError(
            "cli.failure",
            "PhaseSmith could not complete the requested command",
            exception_type=type(error).__name__,
            reason=str(error),
        )
        _emit(failure.to_record(), sys.stderr)
        return 2
    _emit(result, sys.stdout)
    return exit_status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
