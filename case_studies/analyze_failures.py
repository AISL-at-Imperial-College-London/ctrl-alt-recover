"""Post-hoc semantic failure analysis for case-study traces.

The control loop itself already tags every LLM propose/action call with a
*mechanical* `failure_type` at record time (connection_error, timeout,
json_parse_error, schema_validation_error, ...) - see
`common/llm_failures.py`. That tells you a call failed and roughly how, but
not *why the model actually did that* - which matters a lot once you start
comparing local models (llama3:8b, qwen3:8b, ...) against each other or
against hosted models, since they fail in more varied and more interesting
ways.

This script adds that semantic layer as a separate, offline pass: for every
run under a trace directory that either crashed or completed without
succeeding, it feeds the relevant prompt(s)/response(s)/mechanical error
context to a stronger "judge" model and asks it to explain the root cause.
The judge model is never part of the control loop - it only reads finished
traces - so it doesn't affect what's being benchmarked and doesn't need to
run for every step of every run, only the failed ones.

Output: one `failure_analysis.json` written next to each analyzed run's
`final_result.json`, plus an aggregate CSV across all analyzed runs so
failure modes can be pivoted by model/fault/cause.

Usage:
    python analyze_failures.py --case cstr
    python analyze_failures.py --case mixer --model llama3:8b qwen3:8b
    python analyze_failures.py --dry-run                     # see what would run, no API calls
    python analyze_failures.py --force                       # re-analyze runs that already have a diagnosis
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = REPO_ROOT / "traces"

MAX_CALLS_PER_RUN = 5
MAX_FIELD_CHARS = 4000


# =============================================================================
# Diagnosis schema
# =============================================================================


class FailureDiagnosis(BaseModel):
    primary_cause: Literal[
        "format_non_compliance",  # ignored the "return JSON only" instruction
        "schema_mismatch",  # returned JSON but wrong fields/types
        "reasoning_error",  # valid output, but wrong/unsafe decision vs. the rules it was given
        "capability_limitation",  # model appears unable to follow the multi-step instructions at all
        "infra_connection",  # local server (e.g. Ollama) unreachable
        "infra_timeout",  # call timed out, likely model too slow for the configured timeout
        "prompt_ambiguity",  # the prompt itself plausibly allows the model's (wrong) reading
        "unsafe_persistent_state",  # kept producing valid, on-format actions but never reached a safe/successful state
        "other",
    ] = Field(..., description="Single best-fit root cause category")
    explanation: str = Field(
        ...,
        description="2-4 sentences on what the model actually did and why that caused the failure",
    )
    suggested_fix: str = Field(
        ...,
        description="Short, concrete suggestion: prompt change, timeout change, schema change, etc.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ..., description="Confidence in this diagnosis given the context available"
    )


SYSTEM_PROMPT = dedent("""
    You are a failure-analysis assistant for LLM-controlled process-safety
    case studies (a CSTR reactor, and a tank-mixer sequencing system). A
    smaller model - often a local model such as llama3:8b or qwen3:8b - was
    used to propose control actions in a closed loop with a digital-twin
    safety evaluator. The run you are given either crashed or completed
    without reaching a successful/safe outcome.

    You will receive:
    - run metadata (case study, model under test, injected fault, run index)
    - the run's final status/metrics (from final_result.json)
    - a small window of the LLM calls made during the run: each call's
      prompt, raw response, and any mechanical `failure_type` already
      detected by simple exception classification (e.g. json_parse_error,
      schema_validation_error, connection_error, timeout) - or null if that
      particular call parsed fine.

    A mechanical failure_type tells you THAT parsing/validation failed, not
    WHY the model produced that output. Diagnose the root cause in terms of
    what the model actually did: ignored the output-format instruction,
    returned valid JSON with the wrong shape, followed the format but broke
    a stated safety/control rule, seems categorically unable to follow the
    multi-step instructions, or the failure looks like infrastructure
    (server unreachable / too slow) rather than a model-quality issue.

    Return STRICT JSON matching the given schema only.
    """).strip()


# =============================================================================
# Trace discovery
# =============================================================================


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_run_dirs(
    trace_dir: Path,
    *,
    case: Optional[str] = None,
    experiment_id: Optional[str] = None,
    model: Optional[List[str]] = None,
    fault: Optional[str] = None,
) -> List[Path]:
    """traces/<case_study>/<experiment_id>/<model>/<fault>/run_XXX/"""
    case_glob = case if case else "*"
    candidates = sorted(trace_dir.glob(f"{case_glob}/*/*/*/run_*"))

    def keep(run_dir: Path) -> bool:
        parts = run_dir.relative_to(
            trace_dir
        ).parts  # (case, exp_id, model, fault, run_XXX)
        if len(parts) != 5:
            return False
        _, exp_id, model_dir, fault_dir, _ = parts
        if experiment_id and experiment_id not in exp_id:
            return False
        if model and not any(_norm(m) == _norm(model_dir) for m in model):
            return False
        if fault and _norm(fault) != _norm(fault_dir):
            return False
        return True

    return [d for d in candidates if keep(d)]


def load_run_context(run_dir: Path) -> Optional[Dict[str, Any]]:
    final_result_path = run_dir / "final_result.json"
    if not final_result_path.exists():
        return None
    final_result = json.loads(final_result_path.read_text(encoding="utf-8"))

    metadata = {}
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    events: List[Dict[str, Any]] = []
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))

    return {"metadata": metadata, "final_result": final_result, "events": events}


def classify_run_outcome(final_result: Dict[str, Any]) -> Optional[str]:
    """'hard' = crashed, 'soft' = completed but never succeeded, None = fine."""
    status = final_result.get("status")
    result = final_result.get("result") or {}
    if status == "error":
        return "hard"
    if status == "completed" and "success" in result and not result.get("success"):
        return "soft"
    return None


def _truncate(obj: Any, max_chars: int = MAX_FIELD_CHARS) -> Any:
    if isinstance(obj, str):
        return obj if len(obj) <= max_chars else obj[:max_chars] + "...[truncated]"
    if isinstance(obj, dict):
        return {k: _truncate(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(v, max_chars) for v in obj]
    return obj


def collect_llm_calls(
    run_dir: Path, events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    calls = []
    for ev in events:
        if ev.get("event") != "llm_call":
            continue
        record: Dict[str, Any] = {
            "iteration": ev.get("iteration"),
            "agent": ev.get("agent"),
            "failure_type": ev.get("failure_type"),
            "latency_seconds": ev.get("latency_seconds"),
        }
        response_path = (ev.get("response_artifact") or {}).get("path")
        prompt_path = (ev.get("prompt_artifact") or {}).get("path")
        if response_path and (run_dir / response_path).exists():
            record["response"] = json.loads(
                (run_dir / response_path).read_text(encoding="utf-8")
            )
        if prompt_path and (run_dir / prompt_path).exists():
            record["prompt"] = json.loads(
                (run_dir / prompt_path).read_text(encoding="utf-8")
            )
        calls.append(record)
    return calls


def select_calls(
    calls: List[Dict[str, Any]], max_calls: int = MAX_CALLS_PER_RUN
) -> List[Dict[str, Any]]:
    """Prefer the failing call plus a little trailing context, capped for cost."""
    failing_idx = [i for i, c in enumerate(calls) if c.get("failure_type")]
    if failing_idx:
        idx = failing_idx[-1]
        window = calls[max(0, idx - max_calls + 1) : idx + 1]
    else:
        window = calls[-max_calls:]
    return [_truncate(c) for c in window]


# =============================================================================
# Diagnosis
# =============================================================================


def build_context_payload(
    run_dir: Path, context: Dict[str, Any], outcome_kind: str
) -> Dict[str, Any]:
    calls = collect_llm_calls(run_dir, context["events"])
    return {
        "outcome_kind": outcome_kind,
        "metadata": context["metadata"],
        "final_result": _truncate(context["final_result"]),
        "llm_calls": select_calls(calls),
    }


def diagnose(payload: Dict[str, Any], analyzer_model: str) -> FailureDiagnosis:
    llm = ChatOpenAI(model=analyzer_model, temperature=0.0, timeout=60)
    structured = llm.with_structured_output(FailureDiagnosis, include_raw=True)
    user_payload = json.dumps(payload, ensure_ascii=False, default=str)
    raw = structured.invoke([("system", SYSTEM_PROMPT), ("user", user_payload)])
    diagnosis = raw.get("parsed") if isinstance(raw, dict) else None
    if diagnosis is None:
        raw_text = getattr(
            raw.get("raw") if isinstance(raw, dict) else raw, "content", raw
        )
        raise RuntimeError(
            f"Analyzer model returned no parseable diagnosis. Raw: {raw_text!r}"
        )
    return diagnosis


def write_diagnosis(
    run_dir: Path,
    diagnosis: FailureDiagnosis,
    *,
    analyzer_model: str,
    outcome_kind: str,
) -> Path:
    out = {
        "analyzer_model": analyzer_model,
        "analyzed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "outcome_kind": outcome_kind,
        "primary_cause": diagnosis.primary_cause,
        "explanation": diagnosis.explanation,
        "suggested_fix": diagnosis.suggested_fix,
        "confidence": diagnosis.confidence,
    }
    out_path = run_dir / "failure_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trace-dir", type=str, default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--case", type=str, choices=["cstr", "mixer"], default=None)
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Substring match against experiment id",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=None,
        help="Filter to these model dir names, e.g. llama3:8b",
    )
    parser.add_argument("--fault", type=str, default=None)
    parser.add_argument(
        "--analyzer-model",
        type=str,
        default="gpt-5.1",
        help="LLM to use for post-hoc failure analysis",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze runs that already have failure_analysis.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of runs analyzed (0 = no cap)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List runs that would be analyzed; make no API calls",
    )
    parser.add_argument(
        "--summary-csv", type=str, default=None, help="Where to write the aggregate CSV"
    )
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).expanduser().resolve()
    run_dirs = find_run_dirs(
        trace_dir,
        case=args.case,
        experiment_id=args.experiment_id,
        model=args.model,
        fault=args.fault,
    )

    # summary_rows always reflects EVERY failed/incomplete run under the filters,
    # not just the ones touched in this invocation - a run already carrying a
    # failure_analysis.json contributes its saved diagnosis instead of being
    # silently dropped. Losing already-analyzed rows from the aggregate CSV just
    # because this run didn't re-touch them is the bug that bit us before.
    to_analyze: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        context = load_run_context(run_dir)
        if context is None:
            continue
        outcome_kind = classify_run_outcome(context["final_result"])
        if outcome_kind is None:
            continue

        meta = context["metadata"]
        rel = run_dir.relative_to(trace_dir)
        base_row = {
            "case_study": meta.get("case_study"),
            "experiment_id": meta.get("experiment_id"),
            "model": meta.get("model"),
            "fault": meta.get("fault"),
            "run": meta.get("run"),
            "outcome_kind": outcome_kind,
            "run_dir": str(rel),
        }

        existing_path = run_dir / "failure_analysis.json"
        if existing_path.exists() and not args.force:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            base_row.update(
                primary_cause=existing.get("primary_cause", ""),
                confidence=existing.get("confidence", ""),
                explanation=existing.get("explanation", ""),
                suggested_fix=existing.get("suggested_fix", ""),
                analyzer_error="",
            )
            summary_rows.append(base_row)
            continue

        to_analyze.append(
            {
                "run_dir": run_dir,
                "context": context,
                "outcome_kind": outcome_kind,
                "row": base_row,
            }
        )

    print(
        f"Found {len(run_dirs)} run(s) under {trace_dir}; {len(to_analyze)} need analysis."
    )
    if args.limit and len(to_analyze) > args.limit:
        print(f"Capping to --limit {args.limit}")
        to_analyze = to_analyze[: args.limit]

    for i, item in enumerate(to_analyze, start=1):
        run_dir = item["run_dir"]
        rel = run_dir.relative_to(trace_dir)
        row = item["row"]
        print(f"[{i}/{len(to_analyze)}] {rel} (outcome={item['outcome_kind']})")

        if args.dry_run:
            summary_rows.append(row)
            continue

        try:
            payload = build_context_payload(
                run_dir, item["context"], item["outcome_kind"]
            )
            diagnosis = diagnose(payload, args.analyzer_model)
            write_diagnosis(
                run_dir,
                diagnosis,
                analyzer_model=args.analyzer_model,
                outcome_kind=item["outcome_kind"],
            )
            row.update(
                primary_cause=diagnosis.primary_cause,
                confidence=diagnosis.confidence,
                explanation=diagnosis.explanation,
                suggested_fix=diagnosis.suggested_fix,
                analyzer_error="",
            )
        except Exception as exc:
            print(f"    ANALYZER ERROR: {exc}")
            row.update(
                primary_cause="",
                confidence="",
                explanation="",
                suggested_fix="",
                analyzer_error=str(exc)[:500],
            )
        summary_rows.append(row)

    if not summary_rows:
        print("Nothing to summarize.")
        return

    df = pd.DataFrame(summary_rows)
    summary_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else trace_dir / "failure_analysis_summary.csv"
    )
    df.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {summary_path}")

    if not args.dry_run and "primary_cause" in df.columns:
        pivot = (
            df[df["primary_cause"] != ""]
            .groupby(["model", "primary_cause"])
            .size()
            .unstack(fill_value=0)
        )
        if not pivot.empty:
            print("\nPrimary cause counts by model:")
            print(pivot.to_string())


if __name__ == "__main__":
    main()
