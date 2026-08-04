"""Structured, append-only experiment tracing for both case studies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import numbers
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return name or "unknown"


def _jsonable(value: Any) -> Any:
    """Convert common scientific/LangChain/Pydantic values to JSON-safe data."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


class TraceRecorder:
    """Writes one self-contained directory per experiment run."""

    TRACE_VERSION = "1.0"

    def __init__(
        self,
        *,
        base_dir: str | Path,
        case_study: str,
        model: str,
        fault: str,
        run: int,
        experiment_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.experiment_id = _safe_name(
            experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.run_dir = (
            self.base_dir
            / _safe_name(case_study)
            / self.experiment_id
            / _safe_name(model)
            / _safe_name(fault)
            / f"run_{int(run):03d}"
        )
        for child in (
            "queries",
            "subgraphs",
            "prompts",
            "responses",
            "trajectories",
            "live",
        ):
            (self.run_dir / child).mkdir(parents=True, exist_ok=True)

        self.events_path = self.run_dir / "events.jsonl"
        self._sequence = 0
        self._llm_call = 0
        self._artifacts: list[Dict[str, Any]] = []
        self._live_paths: set[Path] = set()
        self._live_streams: Dict[Path, TextIO] = {}
        self.metadata: Dict[str, Any] = {
            "trace_version": self.TRACE_VERSION,
            "case_study": case_study,
            "model": model,
            "fault": fault,
            "run": int(run),
            "experiment_id": self.experiment_id,
            "repository_commit": _git_commit(self.repo_root),
            "started_at_utc": _utc_now(),
            **dict(metadata or {}),
        }
        self._write_json("metadata.json", self.metadata)
        self.record_event("run_started", metadata=self.metadata)

    def _artifact_record(self, path: Path, content: bytes) -> Dict[str, Any]:
        record = {
            "path": path.relative_to(self.run_dir).as_posix(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        self._artifacts.append(record)
        return record

    def _write_text(self, relative_path: str, content: str) -> Dict[str, Any]:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")
        path.write_bytes(encoded)
        return self._artifact_record(path, encoded)

    def _write_json(self, relative_path: str, value: Any) -> Dict[str, Any]:
        content = json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n"
        return self._write_text(relative_path, content)

    def record_event(self, event: str, **payload: Any) -> None:
        self._sequence += 1
        record = {
            "sequence": self._sequence,
            "timestamp_utc": _utc_now(),
            "event": event,
            **_jsonable(payload),
        }
        with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_live_row(self, *, stream_name: str, row: Mapping[str, Any]) -> Path:
        """Append one immediately visible JSONL row for live dashboards."""
        path = self.run_dir / "live" / f"{_safe_name(stream_name)}.jsonl"
        if path not in self._live_paths:
            self._live_paths.add(path)
            self._live_streams[path] = path.open(
                "w", encoding="utf-8", newline="\n", buffering=1
            )
            self.record_event(
                "live_stream_started",
                stream=stream_name,
                path=path.relative_to(self.run_dir).as_posix(),
            )

        record = {
            "written_at_utc": _utc_now(),
            **_jsonable(dict(row)),
        }
        output = self._live_streams[path]
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        return path

    def record_sparql(
        self,
        *,
        name: str,
        query: str,
        subgraph: str,
        endpoint: Optional[str],
        loaded: bool,
    ) -> None:
        safe = _safe_name(name)
        query_artifact = self._write_text(f"queries/{safe}.sparql", query or "")
        graph_artifact = self._write_text(f"subgraphs/{safe}.ttl", subgraph or "")
        self.record_event(
            "sparql_retrieval",
            name=name,
            endpoint=endpoint,
            loaded=bool(loaded),
            query_artifact=query_artifact,
            subgraph_artifact=graph_artifact,
        )

    def record_llm_call(
        self,
        *,
        agent: str,
        iteration: int,
        messages: Sequence[Mapping[str, Any]],
        raw_response: Any,
        parsed_response: Any,
        rationale: Optional[str],
        usage: Optional[Mapping[str, Any]],
        latency_seconds: float,
    ) -> None:
        self._llm_call += 1
        stem = f"call_{self._llm_call:04d}_{_safe_name(agent)}_iteration_{int(iteration):04d}"
        prompt_artifact = self._write_json(
            f"prompts/{stem}.json",
            {"agent": agent, "iteration": int(iteration), "messages": messages},
        )
        response_artifact = self._write_json(
            f"responses/{stem}.json",
            {
                "agent": agent,
                "iteration": int(iteration),
                "raw_response": raw_response,
                "parsed_response": parsed_response,
                "agent_provided_rationale": rationale,
                "usage": usage or {},
                "latency_seconds": latency_seconds,
            },
        )
        self.record_event(
            "llm_call",
            agent=agent,
            iteration=int(iteration),
            rationale=rationale,
            usage=usage or {},
            latency_seconds=latency_seconds,
            prompt_artifact=prompt_artifact,
            response_artifact=response_artifact,
        )

    def write_trajectory(
        self,
        *,
        name: str,
        rows: Iterable[Mapping[str, Any]],
        accepted: bool,
        condition: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_rows = [_jsonable(dict(row)) for row in rows]
        if not normalized_rows:
            self.record_event(
                "trajectory_empty", name=name, accepted=accepted, condition=condition
            )
            return None

        fieldnames: list[str] = []
        for row in normalized_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        path = self.run_dir / "trajectories" / f"{_safe_name(name)}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in normalized_rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        for key, value in row.items()
                    }
                )
        content = path.read_bytes()
        artifact = self._artifact_record(path, content)
        sidecar = self._write_json(
            f"trajectories/{_safe_name(name)}.metadata.json",
            {
                "name": name,
                "accepted": bool(accepted),
                "condition": condition,
                "rows": len(normalized_rows),
                **dict(metadata or {}),
            },
        )
        self.record_event(
            "trajectory_written",
            name=name,
            accepted=bool(accepted),
            condition=condition,
            rows=len(normalized_rows),
            trajectory_artifact=artifact,
            metadata_artifact=sidecar,
        )
        return artifact

    def finalize(self, *, result: Any, status: str = "completed") -> None:
        self.record_event("run_finished", status=status, result=result)
        final = {
            "status": status,
            "finished_at_utc": _utc_now(),
            "result": _jsonable(result),
        }
        self._write_json("final_result.json", final)
        for stream in self._live_streams.values():
            stream.close()
        for live_path in sorted(self._live_paths):
            if live_path.exists():
                self._artifact_record(live_path, live_path.read_bytes())
        events_content = self.events_path.read_bytes()
        self._artifact_record(self.events_path, events_content)
        manifest = {
            "trace_version": self.TRACE_VERSION,
            "run_directory": self.run_dir.as_posix(),
            "artifact_count": len(self._artifacts),
            "artifacts": self._artifacts,
        }
        self._write_json("manifest.json", manifest)
