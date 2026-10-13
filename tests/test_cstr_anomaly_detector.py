"""Tests for the online temperature threshold detector.

The module is named cstr_anomaly_threshold_test.py but is source, not a test.
"""

import pytest

from cstr_anomaly_threshold_test import OnlineTempThresholdDetector, TempThresholdConfig


def cfg(**kwargs):
    defaults = dict(
        persistence_window=4,
        persistence_ratio=0.75,
        enable_settle_gate=False,
    )
    defaults.update(kwargs)
    return TempThresholdConfig(**defaults)


def feed(det, errors, **extra):
    return [
        det.update({"T_meas_error": e, "T_meas": 310.0 + e, **extra}) for e in errors
    ]


def test_config_defaults():
    c = TempThresholdConfig()
    assert c.allowed_phases is None
    assert (c.delta_above_sp, c.persistence_window, c.persistence_ratio) == (
        2.0,
        30,
        0.7,
    )
    assert c.require_full_window and c.enable_settle_gate
    assert (c.settle_band, c.settle_required_samples) == (1.0, 60)


def test_threshold_is_strictly_above_delta():
    det = OnlineTempThresholdDetector(cfg())
    out = feed(det, [2.0, 2.01, float("nan")])
    assert [o["is_anomaly_raw"] for o in out] == [False, True, False]
    assert all(o["threshold"] == 2.0 for o in out)
    assert out[0]["settled"] is True  # settle gate disabled


def test_full_window_required_before_flag():
    det = OnlineTempThresholdDetector(cfg())
    out = feed(det, [5.0] * 4)
    assert [o["fault_flag"] for o in out] == [False, False, False, True]
    assert [o["window_ready"] for o in out] == [False, False, False, True]
    assert out[2]["anomaly_ratio"] == 1.0


def test_partial_window_allowed_when_not_required():
    det = OnlineTempThresholdDetector(cfg(require_full_window=False))
    out = feed(det, [5.0, 0.0])
    assert out[0]["fault_flag"] is True
    assert out[1]["fault_flag"] is False and out[1]["anomaly_ratio"] == 0.5


def test_persistence_ratio_over_rolling_window():
    det = OnlineTempThresholdDetector(cfg())
    out = feed(det, [5.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.0])
    # ratios once window full: 3/4, 3/4, 2/4, 2/4
    assert [o["anomaly_ratio"] for o in out[3:]] == [0.75, 0.75, 0.5, 0.5]
    assert [o["fault_flag"] for o in out[3:]] == [True, True, False, False]


def test_settle_gate_blocks_until_consecutive_settled_samples():
    det = OnlineTempThresholdDetector(
        cfg(enable_settle_gate=True, settle_required_samples=3, persistence_window=1)
    )
    # 0.5 in band, 3.0 out of band resets the count, nan resets it too
    out = feed(det, [0.5, 0.5, 3.0, 0.5, float("nan"), 0.5, -1.0, 1.0])
    assert [o["settled"] for o in out] == [False] * 7 + [True]
    assert not any(o["is_anomaly_raw"] for o in out)
    # Once settled it latches, even when far out of band.
    later = feed(det, [5.0])[0]
    assert later["settled"] and later["is_anomaly_raw"] and later["fault_flag"]


def test_phase_gating():
    det = OnlineTempThresholdDetector(cfg(allowed_phases={2}, persistence_window=2))
    out_bad = det.update({"phase": 1, "T_meas_error": 5.0})
    assert out_bad == {
        "fault_flag": False,
        "is_anomaly_raw": False,
        "anomaly_ratio": 0.0,
        "threshold": None,
        "settled": False,
    }
    feed(det, [5.0, 0.0], phase="2")  # string phase is coerced
    gated = det.update({"T_meas_error": 5.0})  # missing phase -> -1
    assert gated["threshold"] is None and gated["anomaly_ratio"] == 0.5
    assert det.update({"phase": None, "T_meas_error": 5.0})["fault_flag"] is False
    assert det.update({"phase": "NORMAL", "T_meas_error": 5.0})["threshold"] is None


@pytest.mark.parametrize("phases", [None, {1, 2, 3}])
def test_phase_ok_variants(phases):
    det = OnlineTempThresholdDetector(cfg(allowed_phases=phases))
    assert det._phase_ok({"phase": 3}) is True


def test_fresh_detector_is_reset_state():
    c = cfg()
    det = OnlineTempThresholdDetector(c)
    feed(det, [5.0] * 4)
    fresh = OnlineTempThresholdDetector(c)
    assert fresh.update({"T_meas_error": 0.0})["anomaly_ratio"] == 0.0
    assert det._raw_flags.maxlen == 4
