"""Live dashboard for a running CSTR trace.

The viewer is read-only. It follows JSONL streams written by cstr_case.py and
automatically switches to the newest run unless --run-dir is supplied.
"""

from __future__ import annotations

import argparse
import json
import time
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_ROOT = REPO_ROOT / "traces" / "cstr"


class JsonlTail:
    """Incrementally read complete JSON objects appended to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0

    def read_new(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0

        records: List[Dict[str, Any]] = []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    self.offset = start
                    break
                self.offset = stream.tell()
                try:
                    value = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    records.append(value)
        return records


def _number(row: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            value = float(row.get(key))
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    return float("nan")


def _series(rows: Iterable[Dict[str, Any]], *keys: str) -> np.ndarray:
    return np.asarray([_number(row, *keys) for row in rows], dtype=float)


def _latest_run(trace_root: Path) -> Optional[Path]:
    if not trace_root.exists():
        return None
    metadata_files = sorted(
        trace_root.rglob("metadata.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for metadata_path in metadata_files:
        run_dir = metadata_path.parent
        live_dir = run_dir / "live"
        if (live_dir / "cstr_plant.jsonl").exists() or (
            live_dir / "cstr_validation.jsonl"
        ).exists():
            return run_dir
    return None


def _read_metadata(run_dir: Path) -> Dict[str, Any]:
    try:
        value = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _event_text(event: Dict[str, Any]) -> str:
    kind = str(event.get("event", "event"))
    if kind == "sparql_retrieval":
        return f"KG read: {event.get('name', '?')} (loaded={event.get('loaded')})"
    if kind == "llm_call":
        return (
            f"LLM read/write: {event.get('agent', '?')} call, "
            f"iteration {event.get('iteration', '?')}"
        )
    if kind == "validation_decision":
        verdict = "ACCEPTED" if event.get("accepted") else "REJECTED"
        return f"Validation {verdict}: {event.get('summary', event.get('fail_reason', ''))}"
    if kind == "trajectory_written":
        return f"Trajectory written: {event.get('name', '?')} ({event.get('rows', '?')} rows)"
    if kind == "live_stream_started":
        return f"Live stream opened: {event.get('stream', '?')}"
    if kind == "run_finished":
        return f"Run finished: {event.get('status', '?')}"
    return kind.replace("_", " ")


class CstrDashboard:
    def __init__(
        self,
        *,
        trace_root: Path,
        run_dir: Optional[Path],
        window: int,
        refresh_ms: int,
    ) -> None:
        self.trace_root = trace_root
        self.fixed_run_dir = run_dir.resolve() if run_dir else None
        self.window = max(50, int(window))
        self.refresh_ms = max(100, int(refresh_ms))

        self.run_dir: Optional[Path] = None
        self.metadata: Dict[str, Any] = {}
        self.plant_tail: Optional[JsonlTail] = None
        self.validation_tail: Optional[JsonlTail] = None
        self.events_tail: Optional[JsonlTail] = None
        self.plant_rows: List[Dict[str, Any]] = []
        self.validation_rows: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.last_read = {"plant": 0, "validation": 0, "events": 0}
        self._next_discovery = 0.0

        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            pass
        self.fig, grid = plt.subplots(
            3, 2, figsize=(15, 10), constrained_layout=True
        )
        self.temp_ax = grid[0, 0]
        self.level_ax = grid[0, 1]
        self.flow_ax = grid[1, 0]
        self.actuator_ax = grid[1, 1]
        self.validation_ax = grid[2, 0]
        self.activity_ax = grid[2, 1]
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("CSTR live trace")

    def _select_run(self) -> None:
        if self.fixed_run_dir is not None:
            candidate = self.fixed_run_dir
        else:
            now = time.monotonic()
            if now < self._next_discovery:
                return
            self._next_discovery = now + 2.0
            candidate = _latest_run(self.trace_root)
        if candidate is None or candidate == self.run_dir:
            return
        self.run_dir = candidate
        self.metadata = _read_metadata(candidate)
        self.plant_tail = JsonlTail(candidate / "live" / "cstr_plant.jsonl")
        self.validation_tail = JsonlTail(
            candidate / "live" / "cstr_validation.jsonl"
        )
        self.events_tail = JsonlTail(candidate / "events.jsonl")
        self.plant_rows = []
        self.validation_rows = []
        self.events = []

    def _read(self) -> None:
        self._select_run()
        if self.run_dir is None:
            return
        new_plant = self.plant_tail.read_new() if self.plant_tail else []
        new_validation = (
            self.validation_tail.read_new() if self.validation_tail else []
        )
        new_events = self.events_tail.read_new() if self.events_tail else []
        self.plant_rows.extend(new_plant)
        self.validation_rows.extend(new_validation)
        self.events.extend(new_events)
        row_limit = max(10_000, self.window * 4)
        self.plant_rows = self.plant_rows[-row_limit:]
        self.validation_rows = self.validation_rows[-row_limit:]
        self.events = self.events[-100:]
        self.last_read = {
            "plant": len(new_plant),
            "validation": len(new_validation),
            "events": len(new_events),
        }

    @staticmethod
    def _finish_axis(axis: Any, *, title: str, ylabel: str) -> None:
        axis.set_title(title)
        axis.set_xlabel("Simulation time [s]")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize=8)

    def _draw_plant(self) -> None:
        rows = self.plant_rows[-self.window :]
        if not rows:
            panels = (
                (self.temp_ax, "Plant temperature"),
                (self.level_ax, "Plant level"),
                (self.flow_ax, "Plant flows"),
                (self.actuator_ax, "Actuator commands"),
            )
            for axis, title in panels:
                axis.clear()
                axis.set_title(title)
                axis.axis("off")
                axis.text(
                    0.5,
                    0.5,
                    "Waiting for live plant data...\n"
                    "Start a new CSTR experiment in another terminal.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            return

        t = _series(rows, "t", "time")

        self.temp_ax.clear()
        self.temp_ax.plot(t, _series(rows, "T_meas"), label="T measured")
        self.temp_ax.plot(t, _series(rows, "T_true"), label="T true", alpha=0.7)
        self.temp_ax.plot(t, _series(rows, "T_sp"), "--", label="T setpoint")
        unsafe_t = [
            _number(row, "t", "time")
            for row in rows
            if str(row.get("control_zone", "")) == "UNSAFE"
        ]
        unsafe_y = [
            _number(row, "T_meas")
            for row in rows
            if str(row.get("control_zone", "")) == "UNSAFE"
        ]
        if unsafe_t:
            self.temp_ax.scatter(
                unsafe_t, unsafe_y, s=12, color="crimson", label="UNSAFE"
            )
        self._finish_axis(
            self.temp_ax, title="Plant temperature", ylabel="Temperature [K]"
        )

        self.level_ax.clear()
        self.level_ax.plot(t, _series(rows, "L_true", "V"), label="Level true")
        self.level_ax.plot(
            t, _series(rows, "L_meas"), label="Level measured", alpha=0.7
        )
        self.level_ax.plot(t, _series(rows, "L_sp"), "--", label="Level setpoint")
        self._finish_axis(self.level_ax, title="Plant level", ylabel="Level")

        self.flow_ax.clear()
        self.flow_ax.plot(t, _series(rows, "Fin"), label="Fin")
        self.flow_ax.plot(t, _series(rows, "Fout"), label="Fout")
        self.flow_ax.plot(t, _series(rows, "Fin_sp"), "--", label="Fin setpoint")
        self._finish_axis(self.flow_ax, title="Plant flows", ylabel="Flow")

        self.actuator_ax.clear()
        self.actuator_ax.plot(t, _series(rows, "u_valve"), label="Inlet valve")
        self.actuator_ax.plot(t, _series(rows, "u_pump"), label="Outlet pump")
        self.actuator_ax.plot(t, _series(rows, "u_cool"), label="Cooling")
        self.actuator_ax.set_ylim(-0.05, 1.05)
        self._finish_axis(
            self.actuator_ax,
            title="Actuator commands",
            ylabel="Normalized command",
        )

    def _draw_validation(self) -> None:
        self.validation_ax.clear()
        if not self.validation_rows:
            self.validation_ax.text(
                0.5,
                0.5,
                "No validation rollout yet",
                ha="center",
                va="center",
                transform=self.validation_ax.transAxes,
            )
            self.validation_ax.set_title("Latest digital-twin validation")
            self.validation_ax.axis("off")
            return

        action_calls = [
            int(value)
            for row in self.validation_rows
            if np.isfinite(value := _number(row, "action_call"))
        ]
        if not action_calls:
            self.validation_ax.text(
                0.5,
                0.5,
                "Validation stream has no action-call identifier",
                ha="center",
                va="center",
                transform=self.validation_ax.transAxes,
            )
            self.validation_ax.axis("off")
            return
        latest_call = max(action_calls)
        rows = [
            row
            for row in self.validation_rows
            if int(_number(row, "action_call")) == latest_call
        ][-self.window :]
        t = _series(rows, "t", "time")
        self.validation_ax.plot(t, _series(rows, "T_meas"), label="T measured")
        self.validation_ax.plot(
            t, _series(rows, "proposed_T_sp"), "--", label="Proposed T setpoint"
        )
        unsafe = np.asarray(
            [str(row.get("control_zone", "")) == "UNSAFE" for row in rows]
        )
        if unsafe.any():
            self.validation_ax.scatter(
                t[unsafe],
                _series(rows, "T_meas")[unsafe],
                s=12,
                color="crimson",
                label="UNSAFE",
            )
        self._finish_axis(
            self.validation_ax,
            title=f"Latest digital-twin validation (action call {latest_call})",
            ylabel="Temperature [K]",
        )

    def _draw_activity(self) -> None:
        self.activity_ax.clear()
        self.activity_ax.axis("off")
        if self.run_dir is None:
            message = (
                "Waiting for a CSTR trace...\n\n"
                f"Watching: {self.trace_root}\n"
                "Start cstr_case.py in another terminal."
            )
            self.activity_ax.text(0.02, 0.95, message, va="top", family="monospace")
            return

        latest = self.plant_rows[-1] if self.plant_rows else {}
        completed = (self.run_dir / "final_result.json").exists()
        status = "COMPLETE" if completed else "RUNNING"
        header = [
            f"Status: {status}",
            f"Experiment: {self.metadata.get('experiment_id', '?')}",
            f"Model: {self.metadata.get('model', '?')}",
            f"Fault: {self.metadata.get('fault', '?')}",
            f"Run: {self.metadata.get('run', '?')}",
            f"Time: {_number(latest, 't', 'time'):.1f} s",
            f"Phase: {latest.get('phase', '?')}",
            f"Zone: {latest.get('control_zone', '?')}",
            f"Fault tag: {latest.get('fault_tags', latest.get('fault_tag', '?'))}",
            f"New reads: plant={self.last_read['plant']}, "
            f"validation={self.last_read['validation']}, events={self.last_read['events']}",
            "",
            "Recent trace activity:",
        ]
        activity = [
            textwrap.shorten(_event_text(event), width=92, placeholder="...")
            for event in self.events[-9:]
        ]
        text = "\n".join(header + [f"• {item}" for item in activity])
        self.activity_ax.text(
            0.01,
            0.98,
            text,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
            transform=self.activity_ax.transAxes,
        )

    def update(self, _frame: int) -> None:
        self._read()
        self._draw_plant()
        self._draw_validation()
        self._draw_activity()
        if self.run_dir is not None:
            title = (
                f"{self.metadata.get('experiment_id', '?')} / "
                f"{self.metadata.get('model', '?')} / "
                f"{self.metadata.get('fault', '?')} / "
                f"run {self.metadata.get('run', '?')}"
            )
            self.fig.suptitle(f"CSTR live trace — {title}", fontsize=11)

    def show(self) -> None:
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=self.refresh_ms,
            cache_frame_data=False,
        )
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Follow a CSTR experiment trace and plot it in real time."
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help="CSTR trace root used when automatically following the newest run",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Follow one specific run directory instead of the newest run",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1200,
        help="Maximum recent points displayed in each plot",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=500,
        help="Dashboard refresh interval in milliseconds",
    )
    args = parser.parse_args()

    dashboard = CstrDashboard(
        trace_root=args.trace_root.resolve(),
        run_dir=args.run_dir,
        window=args.window,
        refresh_ms=args.refresh_ms,
    )
    dashboard.show()


if __name__ == "__main__":
    main()
