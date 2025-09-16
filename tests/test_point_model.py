import superposedpulses as sp
import numpy as np
import pytest


def test_abstract_model_initialization():
    """Test initialization of AbstractModel."""
    model = sp.AbstractModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    assert model.waiting_time == 10.0
    assert model.T == 100.0
    assert model.dt == 0.01
    assert isinstance(model._pulse_generator, sp.ExponentialShortPulseGenerator)
    assert model._last_used_forcing is None
    assert len(model._times) == int(100.0 / 0.01)


def test_point_model_initialization():
    """Test initialization of PointModel."""
    model = sp.PointModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    assert model.waiting_time == 10.0
    assert model.T == 100.0
    assert model.dt == 0.01
    assert isinstance(model._forcing_generator, sp.StandardForcingGenerator)
    assert model._noise is None


def test_point_model_set_custom_forcing_generator():
    """Test setting a custom forcing generator in PointModel."""
    model = sp.PointModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    custom_forcing_generator = sp.StandardForcingGenerator()
    model.set_custom_forcing_generator(custom_forcing_generator)
    assert model._forcing_generator == custom_forcing_generator


def test_point_model_make_realization():
    """Test the make_realization method of PointModel."""
    model = sp.PointModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    times, signal = model.make_realization()
    assert len(times) == len(signal)
    assert len(times) == int(100.0 / 0.01)
    assert np.all(signal >= 0)  # Signal should be non-negative


def test_point_model_set_amplitude_distribution():
    """Test setting amplitude distribution in PointModel."""
    model = sp.PointModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    model.set_amplitude_distribution("exp", average_amplitude=2.0)
    forcing = model._forcing_generator.get_forcing(model._times, waiting_time=10.0)
    assert np.all(forcing.amplitudes > 0)  # Exponential distribution is positive


def test_point_model_set_duration_distribution():
    """Test setting duration distribution in PointModel."""
    model = sp.PointModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    model.set_duration_distribution("deg", average_duration=1.0)
    forcing = model._forcing_generator.get_forcing(model._times, waiting_time=10.0)
    assert np.all(forcing.durations == 1.0)  # Degenerate distribution is constant


@pytest.mark.filterwarnings("ignore::Warning")
def test_point_model_add_noise():
    """Test adding noise to PointModel."""
    model = sp.PointModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    model.add_noise(noise_to_signal_ratio=0.5, seed=42, noise_type="additive")
    assert model._noise_type == "additive"
    assert model._sigma > 0
    assert model._noise is not None


def test_two_point_model_initialization():
    """Test initialization of TwoPointModel."""
    model = sp.TwoPointModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    assert model.waiting_time == 10.0
    assert model.T == 100.0
    assert model.dt == 0.01
    assert model._forcing_generator is not None


def test_two_point_model_make_realization():
    """Test the make_realization method of TwoPointModel."""
    model = sp.TwoPointModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    times, signal_a, signal_b = model.make_realization()
    assert len(times) == len(signal_a)
    assert len(times) == len(signal_b)
    assert len(times) == int(100.0 / 0.01)
    assert np.all(signal_a >= 0)  # Signal A should be non-negative
    assert np.all(signal_b >= 0)  # Signal B should be non-negative


def test_two_point_model_set_custom_forcing_generator():
    """Test setting a custom forcing generator in TwoPointModel."""
    model = sp.TwoPointModel(waiting_time=10.0, total_duration=100.0, dt=0.01)
    custom_forcing_generator = sp.StandardForcingGenerator()
    model.set_custom_forcing_generator(custom_forcing_generator)
    assert model._forcing_generator == custom_forcing_generator
