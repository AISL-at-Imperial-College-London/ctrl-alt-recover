# # =========================
# # Online anomaly detector (CSV-free)
# # Put this in your MAIN file (or in online_monitor.py and import it)
# # =========================

# from __future__ import annotations

# from dataclasses import dataclass
# from collections import deque
# from typing import Any, Deque, Dict, List, Optional, Sequence, Set
# import math
# import numpy as np


# @dataclass
# class OnlineDetectorConfig:
#     metrics: Sequence[str]
#     allowed_phases: Optional[Set[int]] = None     # e.g. {2} for NORMAL
#     min_baseline_samples: int = 200               # learn baseline before flagging
#     k_sigma: float = 3.0                          # z-score threshold
#     min_violations: int = 1                       # metrics violating to count raw anomaly
#     persistence_window: int = 30                  # samples
#     persistence_ratio: float = 0.7                # fraction of window raw=True to confirm
#     require_full_window: bool = True              # avoid early ratio spikes
#     freeze_baseline_on_anomaly: bool = True       # don't learn during faults


# class _RunningMeanVar:
#     __slots__ = ("n", "mean", "m2")

#     def __init__(self) -> None:
#         self.n = 0
#         self.mean = 0.0
#         self.m2 = 0.0

#     def update(self, x: float) -> None:
#         self.n += 1
#         delta = x - self.mean
#         self.mean += delta / self.n
#         delta2 = x - self.mean
#         self.m2 += delta * delta2

#     @property
#     def std(self) -> float:
#         if self.n < 2:
#             return 0.0
#         return math.sqrt(max(self.m2 / (self.n - 1), 0.0))


# class OnlineAdaptiveDetector:
#     """
#     Online detector that learns baseline online and returns a boolean fault flag.
#     Works directly on your per-step 'row' dict (from row_from_sim()).
#     """

#     def __init__(self, cfg: OnlineDetectorConfig):
#         self.cfg = cfg
#         self.stats: Dict[str, _RunningMeanVar] = {m: _RunningMeanVar() for m in cfg.metrics}
#         self.raw_window: Deque[bool] = deque(maxlen=int(cfg.persistence_window))

#     def _phase_ok(self, row: Dict[str, Any]) -> bool:
#         if self.cfg.allowed_phases is None:
#             return True
#         ph = row.get("phase", None)
#         try:
#             ph_i = int(ph)
#         except Exception:
#             return False
#         return ph_i in self.cfg.allowed_phases

#     def _get_float(self, row: Dict[str, Any], k: str) -> Optional[float]:
#         if k not in row:
#             return None
#         try:
#             v = float(row[k])
#         except Exception:
#             return None
#         if not math.isfinite(v):
#             return None
#         return v

#     def baseline_ready(self) -> bool:
#         return all(self.stats[m].n >= self.cfg.min_baseline_samples for m in self.cfg.metrics)

#     def update(self, row: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Returns dict with:
#           fault_flag: bool  (persistence-filtered)
#           is_anomaly_raw: bool
#           anomaly_ratio: float
#           n_violations: int
#           violated_params: str
#           baseline_ready: bool
#           window_ready: bool
#         """
#         if not self._phase_ok(row):
#             return {
#                 "fault_flag": False,
#                 "is_anomaly_raw": False,
#                 "anomaly_ratio": float(np.mean(self.raw_window)) if self.raw_window else 0.0,
#                 "n_violations": 0,
#                 "violated_params": "",
#                 "baseline_ready": self.baseline_ready(),
#                 "window_ready": len(self.raw_window) >= self.cfg.persistence_window,
#             }

#         ready = self.baseline_ready()
#         violated: List[str] = []

#         if ready:
#             for m in self.cfg.metrics:
#                 x = self._get_float(row, m)
#                 if x is None:
#                     continue
#                 st = self.stats[m]
#                 s = st.std
#                 if s <= 1e-9:
#                     continue
#                 z = abs((x - st.mean) / s)
#                 if z > self.cfg.k_sigma:
#                     violated.append(m)

#         n_viol = len(violated)
#         raw = bool(ready and (n_viol >= int(self.cfg.min_violations)))

#         # persistence
#         self.raw_window.append(raw if ready else False)
#         ratio = float(np.mean(self.raw_window)) if self.raw_window else 0.0
#         window_ready = len(self.raw_window) >= int(self.cfg.persistence_window)

#         fault_flag = bool(ratio >= float(self.cfg.persistence_ratio))
#         if self.cfg.require_full_window and not window_ready:
#             fault_flag = False

#         # baseline update (learn) — optionally freeze during anomaly
#         if not (self.cfg.freeze_baseline_on_anomaly and raw):
#             for m in self.cfg.metrics:
#                 x = self._get_float(row, m)
#                 if x is None:
#                     continue
#                 self.stats[m].update(x)

#         return {
#             "fault_flag": fault_flag,
#             "is_anomaly_raw": raw,
#             "anomaly_ratio": ratio,
#             "n_violations": n_viol,
#             "violated_params": ",".join(violated),
#             "baseline_ready": self.baseline_ready(),
#             "window_ready": window_ready,
#         }


# def default_metrics() -> List[str]:
#     """
#     Keep this short at first to avoid false positives.
#     These keys must exist in your row_from_sim() output.
#     """
#     return [
#         "T_meas",
#         "L_meas",
#         "Fin",
#         "Fout",
#         "Fc",
#         "u_valve",
#         "u_pump",
#         "u_cool",
#         "UA_eff",
#         "mass_balance",
#         "F_leak",
#         "F_over",
#     ]


# # =========================
# # Integration points in YOUR existing graph code
# # =========================

# # In initializing(state): replace state["_detector"] creation with:
# # state["_detector"] = OnlineAdaptiveDetector(
# #     OnlineDetectorConfig(
# #         metrics=default_metrics(),
# #         allowed_phases={2},          # NORMAL only (prevents startup triggering)
# #         min_baseline_samples=200,    # learn ~200s baseline
# #         k_sigma=3.0,
# #         min_violations=1,
# #         persistence_window=30,
# #         persistence_ratio=0.7,
# #         require_full_window=True,
# #         freeze_baseline_on_anomaly=True,
# #     )
# # )

# # In monitoring(state): use returned boolean:
# # det_out = state["_detector"].update(row)
# # state["fault_flag_prev"] = bool(state.get("fault_flag", False))
# # state["fault_flag"] = bool(det_out["fault_flag"])
# # state["anomaly_ratio"] = float(det_out["anomaly_ratio"])
# # state["violated_params"] = str(det_out["violated_params"])
# # state["n_violations"] = int(det_out["n_violations"])
# #
# # rising_edge = state["fault_flag"] and (not state["fault_flag_prev"])
# # state["should_act"] = (not state["sp_change_done"]) and rising_edge


from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, Optional, Set
import numpy as np


@dataclass
class TempThresholdConfig:
    allowed_phases: Optional[Set[int]] = None  # {2} for NORMAL
    delta_above_sp: float = 2.0  # trigger if T_meas > T_sp + 2 (312 if sp=310)
    persistence_window: int = 30
    persistence_ratio: float = 0.7
    require_full_window: bool = True

    # optional "settling" gate so you only start looking once you're near setpoint
    enable_settle_gate: bool = True
    settle_band: float = 1.0  # |T_meas_error| <= 1.0 counts as settled
    settle_required_samples: int = 60  # consecutive samples to declare settled


class OnlineTempThresholdDetector:
    """
    CSV-free online temperature trigger:
      - NORMAL-only (phase gating)
      - threshold trigger relative to current setpoint
      - persistence filter
    Returns boolean fault_flag each step.
    """

    def __init__(self, cfg: TempThresholdConfig):
        self.cfg = cfg
        self._raw_flags: Deque[bool] = deque(maxlen=int(cfg.persistence_window))
        self._settle_count = 0
        self._settled = False

    def _phase_ok(self, row: Dict[str, Any]) -> bool:
        if self.cfg.allowed_phases is None:
            return True
        try:
            ph = int(row.get("phase", -1))
        except Exception:
            return False
        return ph in self.cfg.allowed_phases

    def update(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not self._phase_ok(row):
            return {
                "fault_flag": False,
                "is_anomaly_raw": False,
                "anomaly_ratio": (
                    float(np.mean(self._raw_flags)) if self._raw_flags else 0.0
                ),
                "threshold": None,
                "settled": self._settled,
            }

        # settle gate (optional but recommended)
        if self.cfg.enable_settle_gate:
            T_err = float(row.get("T_meas_error", np.nan))
            if np.isfinite(T_err) and abs(T_err) <= self.cfg.settle_band:
                self._settle_count += 1
            else:
                self._settle_count = 0
            if (not self._settled) and self._settle_count >= int(
                self.cfg.settle_required_samples
            ):
                self._settled = True
        else:
            self._settled = True

        T_meas = float(row.get("T_meas", np.nan))
        # Use current setpoint (robust even if you later change T_sp)
        # row_from_sim already provides T_meas_error = T_meas - T_sp
        T_err = float(row.get("T_meas_error", np.nan))
        threshold = self.cfg.delta_above_sp

        # Trigger rule: T_meas > T_sp + delta  <=>  T_meas_error > delta
        raw = bool(self._settled and np.isfinite(T_err) and (T_err > threshold))

        # persistence filter
        self._raw_flags.append(raw)
        ratio = float(np.mean(self._raw_flags)) if self._raw_flags else 0.0
        window_ready = len(self._raw_flags) >= int(self.cfg.persistence_window)

        fault_flag = bool(ratio >= float(self.cfg.persistence_ratio))
        if self.cfg.require_full_window and not window_ready:
            fault_flag = False

        return {
            "fault_flag": fault_flag,
            "is_anomaly_raw": raw,
            "anomaly_ratio": ratio,
            "threshold": threshold,
            "settled": self._settled,
            "window_ready": window_ready,
        }


# =========================
# WIRING INTO YOUR GRAPH
# =========================

# In initializing(state), replace OnlineAdaptiveDetector(...) with:
#
# state["_detector"] = OnlineTempThresholdDetector(
#     TempThresholdConfig(
#         allowed_phases={2},             # NORMAL only
#         delta_above_sp=2.0,             # 310 -> trigger above 312
#         persistence_window=10,          # faster response than 30 (tune)
#         persistence_ratio=0.6,          # e.g. 6/10 samples
#         require_full_window=True,
#         enable_settle_gate=True,
#         settle_band=1.0,
#         settle_required_samples=60,
#     )
# )
#
# In monitoring(state), leave the existing logic, just read fault_flag from detector:
#
# det_out = state["_detector"].update(row)
# state["fault_flag_prev"] = bool(state.get("fault_flag", False))
# state["fault_flag"] = bool(det_out["fault_flag"])
# state["anomaly_ratio"] = float(det_out.get("anomaly_ratio",_]()
