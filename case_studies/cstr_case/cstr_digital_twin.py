import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import random


# -----------------------------
# Event/Phase Definition
# -----------------------------
class Phase(Enum):
    STARTUP = 1
    NORMAL = 2
    SHUTDOWN = 3


# -----------------------------
# Fault Manager
# -----------------------------
@dataclass
class FaultConfig:
    enable: bool = True
    fouling_on: bool = True
    fouling_t_start: float = 2000.0
    fouling_tau: float = 2000.0
    fouling_max: float = 0.7

    cool_stuck_open: bool = False
    cool_stuck_open_t: float = 3000.0

    cool_stuck_closed: bool = False
    cool_stuck_closed_t: float = 3200.0

    pump_degrade_on: bool = True
    pump_degrade_t: float = 3500.0
    pump_degrade_factor: float = 0.5

    outlet_block_on: bool = False
    outlet_block_t: float = 3600.0
    outlet_block_factor: float = 0.1

    inlet_stuck_open: bool = False
    inlet_stuck_open_t: float = 2500.0
    inlet_stuck_open_value: float = 1.0

    temp_bias_on: bool = False
    temp_bias_t: float = 2200.0
    temp_bias_K: float = +5.0

    level_bias_on: bool = False
    level_bias_t: float = 2400.0
    level_bias_L: float = -1.0

    leak_on: bool = False
    leak_t: float = 2600.0
    leak_k: float = 0.005

    overflow_protection: bool = True
    V_max: float = 11.0
    k_over: float = 0.05


class FaultManager:
    def __init__(self, cfg: FaultConfig, UA_nominal: float):
        self.cfg = cfg
        self.UA_nominal = UA_nominal

    def UA_eff(self, t: float) -> float:
        c = self.cfg
        if (not c.enable) or (not c.fouling_on) or (t < c.fouling_t_start):
            return self.UA_nominal
        dt = t - c.fouling_t_start
        frac = c.fouling_max * (1.0 - np.exp(-dt / max(1e-6, c.fouling_tau)))
        return self.UA_nominal * (1.0 - np.clip(frac, 0.0, 0.95))

    def sensor_T(self, t: float, T_true: float, noise_std: float) -> float:
        T_meas = T_true + noise_std * np.random.randn()
        c = self.cfg
        if c.enable and c.temp_bias_on and t >= c.temp_bias_t:
            T_meas += c.temp_bias_K
        return T_meas

    def sensor_L(self, t: float, L_true: float) -> float:
        c = self.cfg
        if c.enable and c.level_bias_on and t >= c.level_bias_t:
            return L_true + c.level_bias_L
        return L_true

    def actuator_cool_opening(self, t: float, u_cmd: float) -> float:
        c = self.cfg
        if not c.enable:
            return u_cmd
        if c.cool_stuck_open and t >= c.cool_stuck_open_t:
            return 1.0
        if c.cool_stuck_closed and t >= c.cool_stuck_closed_t:
            return 0.3
        return u_cmd

    def actuator_inlet_opening(self, t: float, u_cmd: float) -> float:
        c = self.cfg
        if c.enable and c.inlet_stuck_open and t >= c.inlet_stuck_open_t:
            return c.inlet_stuck_open_value
        return u_cmd

    def pump_flow_factor(self, t: float) -> float:
        c = self.cfg
        if not c.enable:
            return 1.0
        f = 1.0
        if c.pump_degrade_on and t >= c.pump_degrade_t:
            f *= c.pump_degrade_factor
        if c.outlet_block_on and t >= c.outlet_block_t:
            f *= c.outlet_block_factor
        return f

    def leak_flow(self, t: float, V: float) -> float:
        c = self.cfg
        if c.enable and c.leak_on and t >= c.leak_t:
            return max(0.0, c.leak_k * V)
        return 0.0

    def overflow_flow(self, V: float) -> float:
        c = self.cfg
        if c.enable and c.overflow_protection and V > c.V_max:
            return c.k_over * (V - c.V_max)
        return 0.0


# -----------------------------
# Simple actuators + PID
# -----------------------------
@dataclass
class Valve:
    Cv_max: float = 0.06
    tau: float = 0.3
    opening: float = 0.0
    cmd: float = 0.0

    def update(self, dt: float):
        a = 1.0 - np.exp(-dt / max(1e-6, self.tau))
        self.opening = float(
            np.clip(self.opening + a * (self.cmd - self.opening), 0.0, 1.0)
        )

    def flow(self) -> float:
        return max(0.0, self.Cv_max * self.opening)


@dataclass
class Pump:
    Kq: float = 0.08
    tau: float = 0.2
    speed: float = 0.0

    def update(self, cmd: float, dt: float):
        cmd = float(np.clip(cmd, 0.0, 1.0))
        a = 1.0 - np.exp(-dt / max(1e-6, self.tau))
        self.speed = float(np.clip(self.speed + a * (cmd - self.speed), 0.0, 1.0))

    def flow(self) -> float:
        return max(0.0, self.Kq * self.speed)


@dataclass
class PID:
    Kp: float
    Ki: float
    Kd: float = 0.0
    umin: float = 0.0
    umax: float = 1.0
    integ: float = 0.0
    prev_err: float = 0.0

    def step(self, sp: float, pv: float, dt: float, reverse: bool = False) -> float:
        e = (pv - sp) if reverse else (sp - pv)
        d = (e - self.prev_err) / dt if dt > 0 else 0.0
        u_unsat = self.Kp * e + self.integ + self.Kd * d
        u = float(np.clip(u_unsat, self.umin, self.umax))
        saturated = u != u_unsat
        if (not saturated) or (u == self.umax and e < 0) or (u == self.umin and e > 0):
            self.integ += self.Ki * e * dt
        self.prev_err = e
        return u


def level_pid_command(
    pid: PID, L_sp: float, L: float, dt: float, Fin: float, pump: Pump
) -> float:
    u_ff = float(np.clip(Fin / max(1e-9, pump.Kq), 0.0, 1.0))
    e = L - L_sp
    d = (e - pid.prev_err) / dt if dt > 0 else 0.0
    u_unsat = pid.Kp * e + pid.integ + pid.Kd * d
    u_pi = float(np.clip(u_unsat, pid.umin, pid.umax))
    sat = u_pi != u_unsat
    if (not sat) or (u_pi == pid.umax and e < 0) or (u_pi == pid.umin and e > 0):
        pid.integ += pid.Ki * e * dt
    pid.prev_err = e
    return float(np.clip(u_ff + u_pi, 0.0, 1.0))


# -----------------------------
# CSTR model
# -----------------------------
def rate_constants(T: float):
    kA = 0.1 * np.exp(-5000.0 / (8.314 * T))
    kAD = 0.05 * np.exp(-7000.0 / (8.314 * T))
    return kA, kAD


def cstr_rhs(
    t, x, Fin, Fout, CA_in, CD_in, Tin, Tc, Fc, UA_eff, Cp, DH1, DH2, Fc_in_base
):
    V, CA, CB, CC, CD, T = x
    V = max(V, 1e-1)
    V_energy = max(V, 1.0)
    T_amb = 300
    UA_amb = 50

    kA, kAD = rate_constants(T)
    R1 = kAD * CA * CD
    R2 = kA * CA

    dV = Fin - Fout
    dCA = (Fin / V) * CA_in - (Fout / V) * CA - R1 - R2
    dCB = -(Fout / V) * CB + R1
    dCC = -(Fout / V) * CC + R2
    dCD = (Fin / V) * CD_in - (Fout / V) * CD - R1

    UA_scaled = UA_eff * (Fc / max(1e-6, Fc_in_base))
    dT = (
        (Fin / V) * (Tin - T)
        + ((-DH1) * R1 + (-DH2) * R2) / Cp
        - (UA_scaled / (V * Cp)) * (T - Tc)
        - (UA_amb / (V_energy * Cp)) * (T - T_amb)
    )

    return np.array([dV, dCA, dCB, dCC, dCD, dT], dtype=float)


# =============================================================================
# SIM CLASS
# =============================================================================
class CSTRSimulation:
    """
    Call simulate_step(...) to advance the simulation by ONE dt.
    Keeps internal state: x, t, actuator states, PID integrators, etc.
    """

    def __init__(
        self,
        *,
        dt: float = 1.0,
        A_cs: float = 1.0,
        T_startup: float = 1000.0,
        T_normal: float = 6000.0,
        T_shutdown: float = 1000.0,
        # setpoints defaults
        T_sp: float = 310.0,
        L_sp: float = 10.0,
        Fin_sp_startup: float = 2 / 60.0,
        Fin_sp_normal: float = 2 / 60.0,
        # process constants
        Cp: float = 4180.0,
        DH1: float = -1000.0,
        DH2: float = -1000.0,
        UA: float = 500.0,
        Fc_in_base: float = 0.5,
        Fc_max: float = 0.5,
        CA_in_base: float = 0.10,
        CD_in_base: float = 0.05,
        Tin_base: float = 350.0,
        Tc_base: float = 290.0,
        conc_noise_pct: float = 0.02,
        flow_meas_noise_pct: float = 0.02,
        sensor_T_noise_std: float = 0.5,
        seed: Optional[int] = 42,
        fault_cfg: Optional[FaultConfig] = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        # time config
        self.dt = float(dt)
        self.T_startup = float(T_startup)
        self.T_normal = float(T_normal)
        self.T_shutdown = float(T_shutdown)
        self.T_total = self.T_startup + self.T_normal + self.T_shutdown

        self.A_cs = float(A_cs)

        # setpoints (can be overwritten each step)
        self.T_sp = float(T_sp)
        self.L_sp = float(L_sp)
        self.Fin_sp_startup = float(Fin_sp_startup)
        self.Fin_sp_normal = float(Fin_sp_normal)

        # constants/noise
        self.Cp = float(Cp)
        self.DH1 = float(DH1)
        self.DH2 = float(DH2)
        self.UA = float(UA)
        self.Fc_in_base = float(Fc_in_base)
        self.Fc_max = float(Fc_max)
        self.CA_in_base = float(CA_in_base)
        self.CD_in_base = float(CD_in_base)
        self.Tin_base = float(Tin_base)
        self.Tc_base = float(Tc_base)
        self.conc_noise_pct = float(conc_noise_pct)
        self.flow_meas_noise_pct = float(flow_meas_noise_pct)
        self.sensor_T_noise_std = float(sensor_T_noise_std)

        # faults
        self.fault_cfg = (
            fault_cfg if fault_cfg is not None else FaultConfig(enable=False)
        )
        self.faults = FaultManager(self.fault_cfg, UA_nominal=self.UA)

        # actuators + controllers (internal stateful)
        self.valve_in = Valve()
        self.cool_valve_in = Valve(Cv_max=self.Fc_max)
        self.pump_out = Pump()

        self.pid_flow = PID(Kp=10.0, Ki=5.0)
        self.pid_level = PID(Kp=0.2, Ki=0.002)
        self.pid_temp = PID(Kp=0.002, Ki=0.0002)

        # plant state
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.Tin_base], dtype=float)
        self.t = 0.0
        self.current_phase = Phase.STARTUP
        self._startup_done = False

    def _phase(self, L: float, t: float) -> Phase:
        # latch startup completion
        if not self._startup_done:
            if (L >= self.L_sp) or (t >= self.T_startup):
                self._startup_done = True

        if not self._startup_done:
            return Phase.STARTUP

        if t < (self.T_startup + self.T_normal):
            return Phase.NORMAL

        return Phase.SHUTDOWN

    def simulate_step(
        self,
        *,
        T_sp: Optional[float] = None,
        L_sp: Optional[float] = None,
        Fin_sp_normal: Optional[float] = None,
        Fin_sp_startup: Optional[float] = None,
        mode: str = "RUN",
    ) -> Dict[str, Any]:
        """
        Advance by exactly one dt.

        mode:
          - "RUN": normal operation (use your phase logic)
          - "STOP": force inlet closed (Fin_sp=0), keep cooling to protect temperature, allow pump to drain
        """
        # allow external supervisory setpoint changes
        if T_sp is not None:
            self.T_sp = float(T_sp)
        if L_sp is not None:
            self.L_sp = float(L_sp)
        if Fin_sp_normal is not None:
            self.Fin_sp_normal = float(Fin_sp_normal)
        if Fin_sp_startup is not None:
            self.Fin_sp_startup = float(Fin_sp_startup)

        # unpack
        V, CA, CB, CC, CD, T = self.x
        L_true = V / self.A_cs

        # phase update
        self.current_phase = self._phase(L_true, self.t)

        # sensors (faulted)
        T_meas = self.faults.sensor_T(
            self.t, T_true=T, noise_std=self.sensor_T_noise_std
        )
        L_meas = self.faults.sensor_L(self.t, L_true=L_true)

        # supervisory STOP mode overrides feed
        stop_mode = mode.upper() == "STOP"

        # --- phase control logic (unchanged, with STOP override)
        if self.current_phase == Phase.STARTUP:
            Fin_sp = 0.0 if stop_mode else self.Fin_sp_startup
            pump_cmd = 0.0

            Fin_meas = self.valve_in.flow()
            Fin_meas_noisy = Fin_meas * (
                1 + self.flow_meas_noise_pct * np.random.randn()
            )
            valve_cmd = self.pid_flow.step(Fin_sp, Fin_meas_noisy, self.dt)

            cool_valve_cmd = self.pid_temp.step(
                self.T_sp, T_meas, self.dt, reverse=True
            )

        elif self.current_phase == Phase.NORMAL:
            Fin_sp = 0.0 if stop_mode else self.Fin_sp_normal

            Fin_meas = self.valve_in.flow()
            Fin_meas_noisy = Fin_meas * (
                1 + self.flow_meas_noise_pct * np.random.randn()
            )
            valve_cmd = self.pid_flow.step(Fin_sp, Fin_meas_noisy, self.dt)

            Fin_tmp = self.valve_in.flow()
            # in STOP mode we typically want to drain down a bit, keep level safe
            L_sp_used = self.L_sp
            pump_cmd = level_pid_command(
                self.pid_level, L_sp_used, L_meas, self.dt, Fin_tmp, self.pump_out
            )

            cool_valve_cmd = self.pid_temp.step(
                self.T_sp, T_meas, self.dt, reverse=True
            )

        else:  # SHUTDOWN
            Fin_sp = 0.0
            valve_cmd = 0.0

            L_sp_shutdown = 2.0
            if V > 0.01:
                Fin_tmp = self.valve_in.flow()
                pump_cmd = level_pid_command(
                    self.pid_level,
                    L_sp_shutdown,
                    L_meas,
                    self.dt,
                    Fin_tmp,
                    self.pump_out,
                )
                pump_cmd = max(pump_cmd, 0.05)
            else:
                pump_cmd = 0.0

            cool_valve_cmd = (
                self.pid_temp.step(self.T_sp, T_meas, self.dt, reverse=True)
                if V > 0.1
                else 0.0
            )

        # --- apply actuator faults
        valve_cmd_act = self.faults.actuator_inlet_opening(self.t, valve_cmd)
        cool_cmd_act = self.faults.actuator_cool_opening(self.t, cool_valve_cmd)

        # --- move actuators
        self.valve_in.cmd = valve_cmd_act
        self.valve_in.update(self.dt)

        self.pump_out.update(pump_cmd, self.dt)

        self.cool_valve_in.cmd = cool_cmd_act
        self.cool_valve_in.update(self.dt)

        # --- flows
        Fin = self.valve_in.flow()
        Fout_nom = self.pump_out.flow()
        Fout = Fout_nom * self.faults.pump_flow_factor(self.t)

        Fc = self.cool_valve_in.flow()

        # leaks + overflow
        F_leak = self.faults.leak_flow(self.t, V)
        F_over = self.faults.overflow_flow(V)

        # inlet conditions (noisy)
        CA_in = self.CA_in_base * (1 + self.conc_noise_pct * np.random.randn())
        CD_in = self.CD_in_base * (1 + self.conc_noise_pct * np.random.randn())
        Tin = self.Tin_base * (1 + self.conc_noise_pct * np.random.randn())
        Tc = self.Tc_base

        UA_eff = self.faults.UA_eff(self.t)

        # integrate one step
        sol = solve_ivp(
            lambda ts, X: cstr_rhs(
                ts,
                X,
                Fin,
                Fout + F_leak + F_over,
                CA_in,
                CD_in,
                Tin,
                Tc,
                Fc,
                UA_eff,
                self.Cp,
                self.DH1,
                self.DH2,
                self.Fc_in_base,
            ),
            (self.t, self.t + self.dt),
            self.x,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            max_step=0.05,
        )
        self.x = sol.y[:, -1]
        self.t += self.dt

        # update true values for returned measurements
        V2, CA2, CB2, CC2, CD2, T2 = self.x
        L_true2 = V2 / self.A_cs

        # fault tag
        tags = []
        if self.fault_cfg.fouling_on and self.t >= self.fault_cfg.fouling_t_start:
            tags.append("FOUL")
        if self.fault_cfg.pump_degrade_on and self.t >= self.fault_cfg.pump_degrade_t:
            tags.append("PUMP_DEG")
        if F_over > 0:
            tags.append("OVERFLOW")

        return {
            "t": self.t,
            "dt": self.dt,
            "phase": self.current_phase.name,
            "x": self.x.copy(),
            "V": float(V2),
            "L_true": float(L_true2),
            "T_true": float(T2),
            "T_meas": float(T_meas),
            "L_meas": float(L_meas),
            "Fin": float(Fin),
            "Fout": float(Fout),
            "Fc": float(Fc),
            "F_leak": float(F_leak),
            "F_over": float(F_over),
            "UA_eff": float(UA_eff),
            # controller outputs (useful for metrics)
            "valve_cmd": float(valve_cmd_act),
            "pump_cmd": float(pump_cmd),
            "cool_cmd": float(cool_cmd_act),
            # actuator positions
            "u_valve": float(self.valve_in.opening),
            "u_pump": float(self.pump_out.speed),
            "u_cool": float(self.cool_valve_in.opening),
            "fault_tags": "|".join(tags) if tags else "OK",
        }
