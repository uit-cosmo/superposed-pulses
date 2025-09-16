import superposedpulses.pulse_shape as ps
import pytest
import numpy as np


def test_standard_pulse_generator_initialization():
    """Test initialization with valid and invalid shape names."""

    gen = ps.StandardPulseGenerator(shape_name="1-exp")
    assert gen._shape_name == "1-exp"

    gen = ps.StandardPulseGenerator(shape_name="lorentz")
    assert gen._shape_name == "lorentz"

    with pytest.raises(AssertionError, match="Invalid shape_name"):
        ps.StandardPulseGenerator(shape_name="invalid-shape")


@pytest.fixture
def times():
    """Fixture for a time array."""
    return np.linspace(-100, 100, 1000)


@pytest.fixture
def duration():
    """Fixture for a default duration."""
    return 5.0


def test_standard_pulse_generator_get_pulse_1_exp(times, duration):
    """Test the '1-exp' pulse shape."""
    gen = ps.StandardPulseGenerator(shape_name="1-exp")
    pulse = gen.get_pulse(times, duration)
    assert len(pulse) == len(times)
    assert np.all(pulse >= 0)
    assert pulse[-1] < 1e-5


def test_standard_pulse_generator_get_pulse_lorentz(times, duration):
    """Test the 'lorentz' pulse shape."""
    gen = ps.StandardPulseGenerator(shape_name="lorentz")
    pulse = gen.get_pulse(times, duration)
    assert len(pulse) == len(times)
    assert np.all(pulse >= 0)
    assert pulse[np.argmin(np.abs(times))] == pytest.approx(1 / np.pi, rel=1e-3)
    assert pulse[-1] < 1e-3


def test_standard_pulse_generator_get_pulse_gaussian(times, duration):
    """Test the 'gaussian' pulse shape."""
    gen = ps.StandardPulseGenerator(shape_name="gaussian")
    pulse = gen.get_pulse(times, duration)
    assert len(pulse) == len(times)
    assert np.all(pulse >= 0)
    assert pulse[np.argmin(np.abs(times))] == pytest.approx(
        1 / np.sqrt(2 * np.pi), rel=1e-3
    )
    assert pulse[-1] < 1e-5


def test_standard_pulse_generator_get_pulse_2_exp(times, duration):
    """Test the '2-exp' pulse shape."""
    gen = ps.StandardPulseGenerator(shape_name="2-exp", lam=0.3)
    pulse = gen.get_pulse(times, duration)
    assert len(pulse) == len(times)
    assert np.all(pulse >= 0)
    assert pulse[-1] < 1e-5


def test_standard_pulse_generator_get_pulse_2_exp_invalid_lam(times, duration):
    """Test the '2-exp' pulse shape with invalid lam values."""
    with pytest.raises(AssertionError):
        gen = ps.StandardPulseGenerator(shape_name="2-exp", lam=1.5)
        gen.get_pulse(times, duration)


def test_standard_pulse_generator_tolerance_warning(times, duration):
    """Test that a warning is raised when the pulse exceeds the tolerance."""
    gen = ps.StandardPulseGenerator(shape_name="1-exp")
    with pytest.warns(UserWarning, match="Value at end point of kernel > tol"):
        gen.get_pulse(times, duration, tolerance=1e-10)


# Edge Case Tests
def test_standard_pulse_generator_empty_times(duration):
    """Test behavior with an empty times array."""
    gen = ps.StandardPulseGenerator(shape_name="1-exp")
    with pytest.raises(AssertionError):
        gen.get_pulse(np.array([]), duration)

def test_short_pulse_generator_empty_times(duration):
    """Test behavior with an empty times array."""
    gen = ps.ExponentialShortPulseGenerator()
    pulse = gen.get_pulse(np.array([]), duration)
    assert len(pulse) == 0
