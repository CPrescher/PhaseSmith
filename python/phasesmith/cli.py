"""Finite JSON command-line boundary for PhaseSmith automation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO, cast

from ._skill import REFERENCES, skill_path, skill_text
from .automation import (
    AutomationError,
    advisor_packet,
    automation_schema,
    automation_schema_names,
    inspect_cif_file,
    inspect_powder_file,
    lint_recipe_proposal_file,
    load_recipe_proposal,
    load_workflow_plan,
    load_workflow_spec,
    plan_workflow,
    prepare_review_packet,
    replan_workflow,
    resume_workflow,
    review_workflow_output,
    run_workflow,
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

    skill = commands.add_parser("skill", help="locate or read the bundled agent skill")
    skill_action = skill.add_mutually_exclusive_group(required=True)
    skill_action.add_argument("--path", action="store_true", help="print the skill directory")
    skill_action.add_argument(
        "--print",
        dest="skill_reference",
        nargs="?",
        const="skill",
        choices=("skill", *REFERENCES, "all"),
        help="print the entrypoint (default), a named reference, or all instructions as text",
    )

    schema = commands.add_parser("schema", help="print an automation JSON Schema")
    schema.add_argument(
        "contract",
        choices=(*automation_schema_names(), "workflow", "recipe"),
    )

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

    advisor = commands.add_parser(
        "advisor-packet",
        help="emit a sanitized prompt-ready packet for an external recipe advisor",
    )
    advisor.add_argument("workflow", help="a workflow specification or stored plan")

    run = commands.add_parser("run", help="execute an explicitly approved workflow plan")
    run.add_argument("workflow", help="a workflow specification or stored plan")
    run.add_argument("--approve", required=True, metavar="PLAN_ID")
    run.add_argument("--proposal", help="validated human- or AI-authored recipe JSON")
    run.add_argument("--overwrite", action="store_true")

    resume = commands.add_parser("resume", help="continue an explicitly approved checkpoint")
    resume.add_argument("workflow", help="a workflow specification or stored plan")
    resume.add_argument("--approve", required=True, metavar="PLAN_ID")
    resume.add_argument("--overwrite", action="store_true")

    lint = commands.add_parser(
        "lint-recipe",
        help="validate and scientifically lint an external recipe proposal",
    )
    lint.add_argument("workflow", help="a workflow specification or stored plan")
    lint.add_argument("proposal")

    review = commands.add_parser(
        "review",
        help="derive deterministic scientific diagnostics from a workflow output",
    )
    review.add_argument("path", help="a completed workflow output directory")

    replan = commands.add_parser(
        "replan",
        help="create a new digest-linked plan from a saved accepted workflow state",
    )
    replan.add_argument("path", help="a completed workflow output directory")
    replan.add_argument("--output-directory", required=True)
    replan.add_argument("--workflow-id")
    replan.add_argument("--plan-output")
    replan.add_argument("--overwrite", action="store_true")

    review_packet = commands.add_parser(
        "review-packet",
        help="create a child plan and sanitized review packet for an external advisor",
    )
    review_packet.add_argument("path", help="a workflow output directory")
    review_packet.add_argument("--output-directory", required=True)
    review_packet.add_argument("--workflow-id")
    review_packet.add_argument("--plan-output", required=True)
    review_packet.add_argument("--overwrite", action="store_true")

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
        if args.command == "skill":
            if args.path:
                sys.stdout.write(str(skill_path()) + "\n")
            else:
                sys.stdout.write(skill_text(args.skill_reference))
            return 0
        elif args.command == "schema":
            result = automation_schema(args.contract)
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
        elif args.command == "advisor-packet":
            result = advisor_packet(load_workflow_plan(args.workflow))
        elif args.command == "run":
            plan = load_workflow_plan(args.workflow)
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
            plan = load_workflow_plan(args.workflow)
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
        elif args.command == "lint-recipe":
            plan = load_workflow_plan(args.workflow)
            result = lint_recipe_proposal_file(args.proposal, plan)
            exit_status = 0 if result["valid_contract"] else 3
        elif args.command == "review":
            result = review_workflow_output(args.path)
        elif args.command == "replan":
            replanned = replan_workflow(
                args.path,
                output_directory=args.output_directory,
                workflow_id=args.workflow_id,
            )
            if args.plan_output is not None:
                write_workflow_plan(replanned, args.plan_output, overwrite=args.overwrite)
            result = replanned.to_record()
        elif args.command == "review-packet":
            replanned, result = prepare_review_packet(
                args.path,
                output_directory=args.output_directory,
                workflow_id=args.workflow_id,
            )
            write_workflow_plan(replanned, args.plan_output, overwrite=args.overwrite)
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
