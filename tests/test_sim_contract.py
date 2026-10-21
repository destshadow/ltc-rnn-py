"""Contract between the LTC controller and the vertical simulator (SIM).

The first group needs only this repository. The closed-loop test imports the
real SIM from ../Simulatore (or LTC_SIM_ROOT) and is skipped when absent.
"""

import math
import os
from pathlib import Path

import pytest
import torch

from inference.controller_stream import ControllerStream
from ltc.config import LTCConfig
from models.continuous_controller import ContinuousController


# Default dt of SIM/environment.py: LTC and SIM must share it.
SIM_DT = 0.01
EPISODE_STEPS = 2000  # 20 s simulated


def make_controller(input_size=2):
    torch.manual_seed(0)
    return ContinuousController(LTCConfig(input_size, 8), output_size=1)


def observations(steps, input_size):
    """Altitude error, vertical velocity and, optionally, motor thrust."""
    time = torch.arange(steps, dtype=torch.float32) * SIM_DT
    columns = [torch.sin(time), 0.5 * torch.cos(3 * time)]
    if input_size == 3:
        columns.append(0.5 + 0.1 * torch.sin(time))
    return torch.stack(columns, dim=1)


@pytest.mark.parametrize("input_size", [2, 3])
def test_long_episode_at_sim_dt_is_finite_and_bounded(input_size):
    stream = ControllerStream(make_controller(input_size), dt=SIM_DT)

    for sample in observations(EPISODE_STEPS, input_size):
        command = stream.push(sample)
        assert command.shape == (1,)
        # SIM accepts Python floats in [0, 1] and rejects anything else.
        value = float(command[0])
        assert math.isfinite(value) and 0.0 <= value <= 1.0

    assert torch.isfinite(stream.state_snapshot()).all()


def test_stream_matches_batch_rollout_at_sim_dt():
    model = make_controller().eval()
    samples = observations(200, 2)

    with torch.no_grad():
        state = model.initial_state(batch_size=1)
        expected = []
        for sample in samples:
            command, state = model(sample.unsqueeze(0), state, dt=SIM_DT)
            expected.append(command[0])

    stream = ControllerStream(model, dt=SIM_DT)
    actual = [stream.push(sample) for sample in samples]

    torch.testing.assert_close(torch.stack(actual), torch.stack(expected))
    torch.testing.assert_close(stream.state_snapshot(), state)


def test_dt_is_part_of_the_dynamics():
    # A controller tuned at dt=0.1 is not equivalent at SIM's dt=0.01.
    model = make_controller()
    samples = observations(50, 2)
    finals = []
    for dt in (SIM_DT, 0.1):
        stream = ControllerStream(model, dt=dt)
        for sample in samples:
            stream.push(sample)
        finals.append(stream.state_snapshot())

    assert not torch.allclose(finals[0], finals[1])


def test_reset_starts_an_independent_episode():
    stream = ControllerStream(make_controller(), dt=SIM_DT)
    samples = observations(100, 2)
    first = torch.stack([stream.push(sample) for sample in samples])

    stream.reset()
    second = torch.stack([stream.push(sample) for sample in samples])

    torch.testing.assert_close(second, first)


def sim_root():
    default = Path(__file__).resolve().parents[2] / "Simulatore"
    return Path(os.environ.get("LTC_SIM_ROOT", default))


@pytest.fixture
def sim(monkeypatch):
    root = sim_root()
    if not (root / "SIM" / "environment.py").is_file():
        pytest.skip(f"SIM non trovato in {root} (impostare LTC_SIM_ROOT)")

    monkeypatch.syspath_prepend(str(root))
    from SIM.drone import DroneState
    from SIM.environment import VerticalEnvironment
    from SIM.motor import MotorParameters, MotorState
    from SIM.parameters import Parameters
    from SIM.vertical import State

    def make_environment(altitude_m=1.0):
        parameters = Parameters()
        initial = DroneState(
            motion=State(altitude_m=altitude_m, velocity_m_s=0.0),
            motor=MotorState(thrust_fraction=parameters.hover_command),
        )
        return VerticalEnvironment(
            initial_state=initial,
            parameters=parameters,
            motor_parameters=MotorParameters(),
            dt=SIM_DT,
        )

    return make_environment


@pytest.mark.parametrize("input_size", [2, 3])
def test_closed_loop_with_real_sim(sim, input_size):
    # Untrained weights: this checks the interface, not flight quality.
    target_m, altitude_scale_m, velocity_scale_m_s = 2.0, 1.0, 1.0
    env = sim()
    stream = ControllerStream(make_controller(input_size), dt=SIM_DT)

    state = env.reset()
    for _ in range(500):
        features = [
            (target_m - state.motion.altitude_m) / altitude_scale_m,
            state.motion.velocity_m_s / velocity_scale_m_s,
        ]
        if input_size == 3:
            features.append(state.motor.thrust_fraction)

        command = float(stream.push(torch.tensor(features))[0])
        state = env.step(command)

        assert math.isfinite(state.motion.altitude_m)
        assert math.isfinite(state.motion.velocity_m_s)
        assert state.motion.altitude_m >= 0.0

    assert env.time_s == pytest.approx(500 * SIM_DT)

    # Episode boundary: SIM and LTC memory restart together.
    stream.reset()
    assert env.reset().motion.altitude_m == 1.0
    assert stream.state_snapshot() is None
