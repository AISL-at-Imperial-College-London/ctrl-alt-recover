"""Offline tests for the mixer digital twin (case_studies/mixer_case/simulation.py)."""

import random

import pytest

from simulation import Simulation


@pytest.fixture(autouse=True)
def _seed():
    random.seed(1234)


def make_state(**overrides):
    state = {
        "level_B201": 0.022,
        "level_B202": 0.022,
        "level_B203": 0.022,
        "level_B204": 0.022,
        "max_height_B201": 0.033,
        "min_height_B201": 0.022,
        "max_height_B202": 0.033,
        "min_height_B202": 0.022,
        "max_height_B203": 0.033,
        "min_height_B203": 0.022,
        "max_height_B204": 0.055,
        "min_height_B204": 0.022,
        "time": 1e6,
        "tank_B201_state": "Empty",
        "tank_B202_state": "Empty",
        "tank_B203_state": "Empty",
        "tank_B204_state": "Empty",
        "condition": "normal",
        "pump_main": 150,
        "pump_alt": 150,
        "pump_power": 0.0,
        "pump_power_alt": 0.0,
    }
    for tank in ("B201", "B202", "B203", "B204"):
        state[f"valve_in_{tank}"] = 0
        state[f"valve_out_{tank}"] = 0
    state.update(overrides)
    return state


def assert_log_aligned(log):
    lengths = {len(v) for v in log.values()}
    assert len(lengths) == 1, {k: len(v) for k, v in log.items()}
    return lengths.pop()


def test_init_copies_state_and_starts_empty_log():
    sim = Simulation(make_state(level_B203=0.03, condition="leak", pump_power=0.5))
    assert sim.level_B203 == 0.03
    assert sim.condition == "leak"
    assert sim.pump_power == 0.5
    assert sim.max_time == 1e6
    assert sim.time == 0
    assert len(sim.timer) == 19  # "%Y-%m-%d %H:%M:%S"
    assert all(v == [] for v in sim.log.values())
    assert "sensor_discrete_tank_B204_high" in sim.log


def test_init_requires_all_keys():
    state = make_state()
    del state["pump_power_alt"]
    with pytest.raises(KeyError):
        Simulation(state)


def test_volumetric_flow_rate():
    sim = Simulation(make_state())
    sim.get_volumetric_flow_rate()  # pumps off
    sim.pump_power = 1.0
    sim.get_volumetric_flow_rate()  # fewer than two B204 samples
    sim.log["level_B204"] += [0.02, 0.025]
    sim.get_volumetric_flow_rate()
    sim.pump_power, sim.pump_power_alt = 0.0, 1.0
    sim.log["level_B204"].append(0.026)
    sim.get_volumetric_flow_rate()
    assert sim.log["volumetric_flow_rate"][:2] == [0, 0]
    assert sim.log["volumetric_flow_rate"][2] == pytest.approx(0.005)
    assert sim.log["volumetric_flow_rate"][3] == pytest.approx(0.001)


@pytest.mark.parametrize(
    "flag",
    [
        f"state_{op}_tank_{t}"
        for op in ("filling", "emptying")
        for t in ("B201", "B202", "B203", "B204")
    ],
)
def test_get_state_sets_exactly_one_flag(flag):
    sim = Simulation(make_state())
    sim.get_state(flag)
    flags = {k: v[-1] for k, v in sim.log.items() if k.startswith("state_")}
    assert len(flags) == 8
    assert flags.pop(flag) == 1
    assert set(flags.values()) == {0}


@pytest.mark.parametrize(
    "timer, step, expected",
    [
        ("2024-01-01 10:20:30", 3, "2024-01-01 10:20:33"),
        ("2024-01-01 10:20:58", 5, "2024-01-01 10:21:03"),
        ("2024-01-01 10:59:58", 5, "2024-01-01 11:00:03"),
        ("2024-01-01 23:59:59", 1, "2024-01-01 00:00:00"),
    ],
)
def test_get_time_rolls_over(monkeypatch, timer, step, expected):
    monkeypatch.setattr(random, "randint", lambda a, b: step)
    sim = Simulation(make_state())
    sim.timer = timer
    assert sim.get_time() == expected
    assert sim.timer == expected


def test_check_level_thresholds():
    sim = Simulation(
        make_state(level_B201=0.0, level_B202=0.01, level_B203=0.025, level_B204=0.04)
    )
    sim.check_level()
    sim.level_B201, sim.level_B204 = 0.031, 0.051
    sim.check_level()
    log = sim.log
    first = {
        k[len("sensor_discrete_tank_") :]: v[0]
        for k, v in log.items()
        if "discrete" in k
    }
    assert first == {
        "B201_low": 0, "B201_medium": 0, "B201_high": 0,
        "B202_low": 1, "B202_medium": 0, "B202_high": 0,
        "B203_low": 1, "B203_medium": 1, "B203_high": 0,
        "B204_low": 1, "B204_medium": 1, "B204_high": 0,
    }  # fmt: skip
    assert log["sensor_discrete_tank_B201_high"][1] == 1
    # B204 uses a higher "high" threshold (0.05) than the other tanks.
    assert log["sensor_discrete_tank_B204_high"][1] == 1


@pytest.mark.parametrize("tank", ["B201", "B202", "B203"])
def test_filling_until_full_then_closes_inlet(tank):
    sim = Simulation(make_state(**{f"valve_in_{tank}": 1}))
    log = getattr(sim, f"state_filling_tank_{tank}")()
    assert log is sim.log
    n = assert_log_aligned(log)
    assert 8 <= n <= 15  # ~0.011 m at ~0.001 m/step
    assert getattr(sim, f"level_{tank}") >= 0.033
    assert log[f"level_{tank}"][-2] < 0.033
    assert getattr(sim, f"valve_in_{tank}") == 0
    assert log[f"valve_in_{tank}"][-1] == 0
    assert log[f"valve_in_{tank}"][0] == 1
    assert set(log[f"state_filling_tank_{tank}"]) == {1}
    assert sim.time == pytest.approx(n / 10)
    assert set(log["sensor_continuous_tank"]) == {0}
    others = {"B201", "B202", "B203"} - {tank}
    assert all(set(log[f"level_{o}"]) == {0.022} for o in others)


@pytest.mark.parametrize("tank", ["B201", "B202", "B203"])
def test_filling_with_closed_inlet_raises(tank):
    sim = Simulation(make_state())
    with pytest.raises(ValueError, match=f"Valve in {tank} is closed"):
        getattr(sim, f"state_filling_tank_{tank}")()
    assert sim.log["time"] == []


def test_filling_without_time_budget_does_nothing():
    sim = Simulation(make_state(valve_in_B201=1, time=0))
    assert sim.state_filling_tank_B201() is None
    assert sim.level_B201 == 0.022


def emptying_state(tank, *, alt=False, **overrides):
    values = {f"level_{tank}": 0.033, f"valve_out_{tank}": 1, "valve_in_B204": 1}
    values["pump_power_alt" if alt else "pump_power"] = 1.0
    values.update(overrides)
    return make_state(**values)


@pytest.mark.parametrize("alt", [False, True], ids=["P101", "P102"])
@pytest.mark.parametrize("tank", ["B201", "B202", "B203"])
def test_emptying_moves_water_into_b204(tank, alt):
    sim = Simulation(emptying_state(tank, alt=alt))
    log = getattr(sim, f"state_emptying_tank_{tank}")()
    n = assert_log_aligned(log)
    assert n > 1
    level = getattr(sim, f"level_{tank}")
    assert level <= 0.022
    moved = 0.033 - level
    assert sim.level_B204 == pytest.approx(0.022 + moved)
    assert sum(log["sensor_continuous_tank"]) == pytest.approx(moved)
    assert set(log[f"state_emptying_tank_{tank}"]) == {1}
    assert log["pump_main" if not alt else "pump_alt"][0] == 150


@pytest.mark.parametrize("tank", ["B201", "B202"])
def test_emptying_closes_outlet_when_empty(tank):
    sim = Simulation(emptying_state(tank))
    log = getattr(sim, f"state_emptying_tank_{tank}")()
    assert getattr(sim, f"valve_out_{tank}") == 0
    assert log[f"valve_out_{tank}"][-1] == 0


def test_emptying_b203_keeps_outlet_open():
    # Unlike B201/B202, the B203 close-out is commented out in the source.
    sim = Simulation(emptying_state("B203"))
    log = sim.state_emptying_tank_B203()
    assert sim.valve_out_B203 == 1
    assert log["valve_out_B203"][-1] == 1


@pytest.mark.xfail(
    strict=True,
    reason="simulation.py:601 writes the B203 outlet into log['valve_out_B202']",
)
def test_emptying_b203_bypass_does_not_touch_b202_log():
    sim = Simulation(emptying_state("B203", alt=True))
    log = sim.state_emptying_tank_B203()
    assert set(log["valve_out_B202"]) == {0}


@pytest.mark.parametrize("alt", [False, True], ids=["P101", "P102"])
@pytest.mark.parametrize("tank", ["B201", "B202", "B203"])
def test_leak_loses_water_between_tank_and_b204(tank, alt):
    sim = Simulation(emptying_state(tank, alt=alt, condition="leak"))
    getattr(sim, f"state_emptying_tank_{tank}")()
    moved = 0.033 - getattr(sim, f"level_{tank}")
    gained = sim.level_B204 - 0.022
    assert 0.7 * moved <= gained <= 0.9 * moved
    assert set(sim.log["condition"]) == {"leak"}


@pytest.mark.parametrize("alt", [False, True], ids=["P101", "P102"])
@pytest.mark.parametrize("tank", ["B201", "B202", "B203"])
def test_sensor_fault_under_reports_flow(tank, alt):
    sim = Simulation(emptying_state(tank, alt=alt, condition="sensor_fault"))
    getattr(sim, f"state_emptying_tank_{tank}")()
    moved = 0.033 - getattr(sim, f"level_{tank}")
    reported = sum(sim.log["sensor_continuous_tank"])
    assert 0.8 * moved <= reported <= 0.9 * moved
    assert sim.level_B204 == pytest.approx(0.022 + moved)


@pytest.mark.parametrize("tank", ["B201", "B202", "B203"])
def test_emptying_without_pump_raises(tank):
    sim = Simulation(make_state(**{f"level_{tank}": 0.033, f"valve_out_{tank}": 1}))
    with pytest.raises(ValueError, match=f"cannot empty the tank{tank}") as err:
        getattr(sim, f"state_emptying_tank_{tank}")()
    assert "valve_in_B204" in str(err.value)
    # pump_main/pump_alt are speeds (150), never `True`, so they are always listed.
    assert "pump_main" in str(err.value)


@pytest.mark.xfail(
    strict=True,
    reason="simulation.py:505 reports valve_out_B201 when emptying B202 fails",
)
def test_emptying_b202_error_names_b202_outlet():
    sim = Simulation(make_state(level_B202=0.033, valve_out_B201=1, pump_power=1.0))
    with pytest.raises(ValueError) as err:
        sim.state_emptying_tank_B202()
    assert "valve_out_B202" in str(err.value)


@pytest.mark.parametrize("tank", ["B201", "B202", "B203", "B204"])
def test_emptying_already_empty_tank_is_noop(tank):
    sim = Simulation(emptying_state(tank, **{f"level_{tank}": 0.022}))
    assert getattr(sim, f"state_emptying_tank_{tank}")() is None
    assert sim.log["time"] == []


def test_emptying_b204_drains_and_reopens_b201_inlet():
    sim = Simulation(make_state(level_B204=0.04, valve_out_B204=1))
    log = sim.state_emptying_tank_B204()
    n = assert_log_aligned(log)
    assert 12 <= n <= 25
    assert sim.level_B204 <= 0.022
    assert sim.valve_out_B204 == 0 and log["valve_out_B204"][-1] == 0
    assert sim.valve_in_B201 == 1
    assert set(log["state_emptying_tank_B204"]) == {1}


def test_emptying_b204_with_closed_outlet_raises():
    sim = Simulation(make_state(level_B204=0.04))
    with pytest.raises(ValueError, match=r"\['valve_out_B204'\] is closed"):
        sim.state_emptying_tank_B204()


def test_seeded_runs_are_reproducible():
    def run():
        random.seed(7)
        sim = Simulation(emptying_state("B201", condition="leak"))
        sim.state_emptying_tank_B201()
        return sim.log["level_B201"], sim.log["level_B204"]

    assert run() == run()
