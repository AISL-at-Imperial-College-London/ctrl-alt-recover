"""Fault modes, controllers and phase logic of the CSTR digital twin.

RNG behaviour is covered separately in case_studies/cstr_case/test_cstr_twin_rng.py.
"""

import numpy as np
import pytest

from cstr_digital_twin import (
    PID,
    CSTRSimulation,
    FaultConfig,
    FaultManager,
    Phase,
    Pump,
    level_pid_command,
)


def fm(**kwargs):
    return FaultManager(FaultConfig(**kwargs), UA_nominal=500.0)


def test_ua_eff_fouling_curve():
    m = fm(fouling_t_start=100.0, fouling_tau=50.0, fouling_max=0.7)
    assert m.UA_eff(99.0) == 500.0
    assert m.UA_eff(100.0) == pytest.approx(500.0)
    assert m.UA_eff(150.0) == pytest.approx(500.0 * (1 - 0.7 * (1 - np.exp(-1))))
    assert m.UA_eff(1e6) == pytest.approx(500.0 * 0.3)
    assert fm(enable=False).UA_eff(1e6) == 500.0
    assert fm(fouling_on=False).UA_eff(1e6) == 500.0


def test_ua_eff_fraction_clipped_and_zero_tau():
    m = fm(fouling_t_start=0.0, fouling_tau=0.0, fouling_max=5.0)
    assert m.UA_eff(1.0) == pytest.approx(500.0 * 0.05)


def test_sensor_temperature_bias():
    m = fm(temp_bias_on=True, temp_bias_t=10.0, temp_bias_K=5.0)
    assert m.sensor_T(9.0, T_true=300.0, noise_std=0.0) == 300.0
    assert m.sensor_T(10.0, T_true=300.0, noise_std=0.0) == 305.0
    off = fm(enable=False, temp_bias_on=True, temp_bias_t=0.0)
    assert off.sensor_T(10.0, T_true=300.0, noise_std=0.0) == 300.0


def test_sensor_temperature_noise_uses_rng():
    m = FaultManager(FaultConfig(), 500.0, rng=np.random.RandomState(3))
    expected = 300.0 + 2.0 * np.random.RandomState(3).randn()
    assert m.sensor_T(0.0, T_true=300.0, noise_std=2.0) == expected


def test_sensor_level_bias():
    m = fm(level_bias_on=True, level_bias_t=5.0, level_bias_L=-1.0)
    assert m.sensor_L(4.0, 10.0) == 10.0
    assert m.sensor_L(5.0, 10.0) == 9.0
    assert fm(enable=False, level_bias_on=True).sensor_L(1e6, 10.0) == 10.0


def test_cooling_actuator_faults():
    assert fm(enable=False, cool_stuck_open=True).actuator_cool_opening(1e6, 0.2) == 0.2
    stuck_open = fm(cool_stuck_open=True, cool_stuck_open_t=10.0)
    assert stuck_open.actuator_cool_opening(9.0, 0.2) == 0.2
    assert stuck_open.actuator_cool_opening(10.0, 0.2) == 1.0
    stuck_closed = fm(cool_stuck_closed=True, cool_stuck_closed_t=10.0)
    assert stuck_closed.actuator_cool_opening(10.0, 0.9) == 0.3
    # Stuck-open is checked first when both are active.
    both = fm(cool_stuck_open=True, cool_stuck_closed=True)
    assert both.actuator_cool_opening(1e6, 0.5) == 1.0
    assert fm().actuator_cool_opening(1e6, 0.5) == 0.5


def test_inlet_actuator_fault():
    m = fm(inlet_stuck_open=True, inlet_stuck_open_t=10.0, inlet_stuck_open_value=0.8)
    assert m.actuator_inlet_opening(9.0, 0.1) == 0.1
    assert m.actuator_inlet_opening(10.0, 0.1) == 0.8
    assert (
        fm(enable=False, inlet_stuck_open=True).actuator_inlet_opening(1e6, 0.1) == 0.1
    )


def test_pump_flow_factor():
    m = fm(
        pump_degrade_t=10.0,
        pump_degrade_factor=0.5,
        outlet_block_on=True,
        outlet_block_t=20.0,
        outlet_block_factor=0.1,
    )
    assert m.pump_flow_factor(5.0) == 1.0
    assert m.pump_flow_factor(10.0) == 0.5
    assert m.pump_flow_factor(20.0) == pytest.approx(0.05)
    assert fm(enable=False).pump_flow_factor(1e6) == 1.0
    assert fm(pump_degrade_on=False).pump_flow_factor(1e6) == 1.0


def test_leak_flow():
    m = fm(leak_on=True, leak_t=10.0, leak_k=0.01)
    assert m.leak_flow(9.0, 5.0) == 0.0
    assert m.leak_flow(10.0, 5.0) == pytest.approx(0.05)
    assert m.leak_flow(10.0, -5.0) == 0.0
    assert fm(enable=False, leak_on=True).leak_flow(1e6, 5.0) == 0.0


def test_overflow_flow():
    m = fm(V_max=11.0, k_over=0.05)
    assert m.overflow_flow(11.0) == 0.0
    assert m.overflow_flow(13.0) == pytest.approx(0.1)
    assert fm(overflow_protection=False).overflow_flow(20.0) == 0.0
    assert fm(enable=False).overflow_flow(20.0) == 0.0


def test_level_pid_command_feedforward_plus_pi():
    pid = PID(Kp=0.2, Ki=0.01)
    pump = Pump(Kq=0.08)
    # Fin/Kq = 0.5 feedforward, e = 11 - 10 = 1 -> Kp*e = 0.2
    u = level_pid_command(pid, L_sp=10.0, L=11.0, dt=2.0, Fin=0.04, pump=pump)
    assert u == pytest.approx(0.7)
    assert pid.integ == pytest.approx(0.02) and pid.prev_err == 1.0


def test_level_pid_command_saturation_and_zero_dt():
    pid = PID(Kp=10.0, Ki=1.0, Kd=1.0)
    u = level_pid_command(pid, L_sp=0.0, L=5.0, dt=0.0, Fin=1.0, pump=Pump(Kq=0.0))
    assert u == 1.0
    assert pid.integ == 0.0  # saturated high with e > 0: anti-windup holds
    u = level_pid_command(pid, L_sp=10.0, L=0.0, dt=1.0, Fin=0.0, pump=Pump())
    assert u == 0.0
    assert pid.integ == 0.0  # saturated low with e < 0


def test_phase_latches_startup_on_level():
    sim = CSTRSimulation(T_startup=100.0, T_normal=100.0, L_sp=10.0)
    assert sim._phase(L=5.0, t=0.0) is Phase.STARTUP
    assert sim._phase(L=10.0, t=1.0) is Phase.NORMAL
    assert sim._phase(L=0.0, t=2.0) is Phase.NORMAL  # latched
    assert sim._phase(L=0.0, t=200.0) is Phase.SHUTDOWN


def test_phase_startup_ends_on_time():
    sim = CSTRSimulation(T_startup=100.0, T_normal=100.0)
    assert sim._phase(L=0.0, t=99.0) is Phase.STARTUP
    assert sim._phase(L=0.0, t=100.0) is Phase.NORMAL


def test_constructor_overrides_and_totals():
    sim = CSTRSimulation(T_startup=1.0, T_normal=2.0, T_shutdown=3.0, seed=None)
    assert sim.T_total == 6.0
    assert sim.fault_cfg.enable is False
    assert sim.faults.UA_nominal == sim.UA == 500.0
    assert sim.rng is None


def test_setpoint_overrides_are_applied():
    sim = CSTRSimulation(isolated_rng=True)
    sim.simulate_step(T_sp=305, L_sp=8, Fin_sp_normal=0.01, Fin_sp_startup=0.02)
    assert (sim.T_sp, sim.L_sp, sim.Fin_sp_normal, sim.Fin_sp_startup) == (
        305.0,
        8.0,
        0.01,
        0.02,
    )


def test_normal_phase_step_runs_level_controller():
    sim = CSTRSimulation(T_startup=0.0, T_normal=100.0, isolated_rng=True)
    sim.x[0] = 12.0  # above L_sp -> pump should drain
    out = sim.simulate_step()
    assert out["phase"] == "NORMAL"
    assert out["pump_cmd"] > 0.0 and out["u_pump"] > 0.0
    assert out["t"] == 1.0 and out["fault_tags"] == "OK"


def test_normal_phase_stop_mode_closes_inlet():
    sim = CSTRSimulation(T_startup=0.0, T_normal=100.0, isolated_rng=True)
    sim.valve_in.opening = 0.5
    out = sim.simulate_step(mode="stop")
    assert out["phase"] == "NORMAL"
    assert out["valve_cmd"] == 0.0  # PID drives Fin_sp=0, clipped at umin


def test_startup_stop_mode_zero_feed():
    sim = CSTRSimulation(isolated_rng=True)
    out = sim.simulate_step(mode="STOP")
    assert out["phase"] == "STARTUP"
    assert out["valve_cmd"] == 0.0 and out["pump_cmd"] == 0.0


def test_shutdown_empty_tank_idles_pump_and_cooling():
    sim = CSTRSimulation(T_startup=0.0, T_normal=0.0, isolated_rng=True)
    out = sim.simulate_step()
    assert out["phase"] == "SHUTDOWN"
    assert (out["valve_cmd"], out["pump_cmd"], out["cool_cmd"]) == (0.0, 0.0, 0.0)


def test_shutdown_full_tank_drains_with_minimum_pump():
    sim = CSTRSimulation(T_startup=0.0, T_normal=0.0, isolated_rng=True)
    sim.x[0] = 1.0  # below shutdown level 2.0 -> PI wants 0, floor is 0.05
    out = sim.simulate_step()
    assert out["phase"] == "SHUTDOWN"
    assert out["pump_cmd"] == 0.05
    assert out["cool_cmd"] > 0.0  # T (350K) above T_sp, V > 0.1 -> cooling on


def test_fault_tags_and_overflow():
    cfg = FaultConfig(
        fouling_t_start=0.0,
        pump_degrade_t=0.0,
        V_max=5.0,
    )
    sim = CSTRSimulation(fault_cfg=cfg, T_startup=0.0, isolated_rng=True)
    sim.x[0] = 8.0
    out = sim.simulate_step()
    assert out["fault_tags"] == "FOUL|PUMP_DEG|OVERFLOW"
    assert out["F_over"] == pytest.approx(0.05 * 3.0)
    assert out["UA_eff"] == pytest.approx(500.0)  # fouling starts at t=0


def test_fault_tags_ignore_enable_flag():
    # Tags read the individual switches only, so a disabled FaultConfig still
    # reports FOUL/PUMP_DEG once the scheduled times pass.
    cfg = FaultConfig(enable=False, fouling_t_start=0.0, pump_degrade_t=0.0)
    out = CSTRSimulation(fault_cfg=cfg, isolated_rng=True).simulate_step()
    assert out["fault_tags"] == "FOUL|PUMP_DEG"
    assert out["UA_eff"] == 500.0
