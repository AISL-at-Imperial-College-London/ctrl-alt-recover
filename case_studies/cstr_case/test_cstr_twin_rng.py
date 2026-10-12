"""RNG regression tests for the CSTR digital twin (issue #1).

Needs only numpy and scipy. From the repository root run either
    python -m pytest case_studies/cstr_case/test_cstr_twin_rng.py
    python case_studies/cstr_case/test_cstr_twin_rng.py
"""

import copy

import numpy as np

from cstr_digital_twin import CSTRSimulation


def steps(sim, n, **setpoints):
    return np.array([sim.simulate_step(**setpoints)["x"] for _ in range(n)])


def continuation(validate, **kwargs):
    """Plant steps 11-30, optionally after an unapplied 20-step twin rollout."""
    plant = CSTRSimulation(seed=42, **kwargs)
    steps(plant, 10)
    if validate:
        steps(copy.deepcopy(plant), 20)  # clone_plant_obj() is a deepcopy
    return steps(plant, 20)


def test_default_twin_rollout_shifts_plant_noise():
    # Default behaviour, kept on purpose: twin and plant share NumPy's global
    # stream, so an unapplied rollout changes the noise the plant sees next.
    assert not np.array_equal(continuation(False), continuation(True))


def test_isolated_twin_rollout_leaves_plant_unchanged():
    assert np.array_equal(
        continuation(False, isolated_rng=True), continuation(True, isolated_rng=True)
    )


def test_isolated_twin_replays_plant_future():
    plant = CSTRSimulation(seed=42, isolated_rng=True)
    steps(plant, 10)
    twin = copy.deepcopy(plant)
    proposal = dict(T_sp=309.0, L_sp=9.5, Fin_sp_normal=0.03, Fin_sp_startup=0.03)
    assert np.array_equal(steps(twin, 20, **proposal), steps(plant, 20, **proposal))


def test_isolated_seed_reproduces_across_instances():
    # A standalone default-mode run draws the same seeded stream.
    expected = steps(CSTRSimulation(seed=42), 20)
    a = CSTRSimulation(seed=42, isolated_rng=True)
    b = CSTRSimulation(seed=42, isolated_rng=True)
    run_a, run_b = [], []
    for _ in range(20):
        run_a.append(a.simulate_step()["x"])
        np.random.randn()  # unrelated draw from the global stream
        run_b.append(b.simulate_step()["x"])
    assert np.array_equal(run_a, expected)
    assert np.array_equal(run_b, expected)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
