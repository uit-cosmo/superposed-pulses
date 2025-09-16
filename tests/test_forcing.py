import superposedpulses as sp
import numpy as np


def test_forcing_initialization():
    """Test initialization of Forcing."""
    arrival_times = np.array([1.0, 2.0, 3.0])
    amplitudes = np.array([0.5, 1.0, 1.5])
    durations = np.array([0.1, 0.2, 0.3])
    forcing = sp.Forcing(3, arrival_times, amplitudes, durations)
    assert forcing.total_pulses == 3
    np.testing.assert_array_equal(forcing.arrival_times, arrival_times)
    np.testing.assert_array_equal(forcing.amplitudes, amplitudes)
    np.testing.assert_array_equal(forcing.durations, durations)


def test_forcing_get_pulse_parameters():
    """Test get_pulse_parameters method of Forcing."""
    arrival_times = np.array([1.0, 2.0, 3.0])
    amplitudes = np.array([0.5, 1.0, 1.5])
    durations = np.array([0.1, 0.2, 0.3])
    forcing = sp.Forcing(3, arrival_times, amplitudes, durations)
    pulse = forcing.get_pulse_parameters(1)
    assert pulse.arrival_time == 2.0
    assert pulse.amplitude == 1.0
    assert pulse.duration == 0.2


def test_standard_forcing_generator_default_amplitude_distribution():
    """Test default amplitude distribution in StandardForcingGenerator."""
    generator = sp.StandardForcingGenerator()
    total_pulses = 5
    amplitudes = generator._get_amplitudes(total_pulses)
    assert len(amplitudes) == total_pulses
    assert np.all(amplitudes > 0)  # Exponential distribution is positive


def test_standard_forcing_generator_custom_amplitude_distribution():
    """Test custom amplitude distribution in StandardForcingGenerator."""
    generator = sp.StandardForcingGenerator()

    def custom_distribution(n):
        return np.full(n, 2.0)

    generator.set_amplitude_distribution(custom_distribution)
    amplitudes = generator._get_amplitudes(3)
    np.testing.assert_array_equal(amplitudes, [2.0, 2.0, 2.0])


def test_standard_forcing_generator_default_duration_distribution():
    """Test default duration distribution in StandardForcingGenerator."""
    generator = sp.StandardForcingGenerator()
    total_pulses = 5
    durations = generator._get_durations(total_pulses)
    assert len(durations) == total_pulses
    np.testing.assert_array_equal(durations, np.ones(total_pulses))


def test_standard_forcing_generator_custom_duration_distribution():
    """Test custom duration distribution in StandardForcingGenerator."""
    generator = sp.StandardForcingGenerator()

    def custom_distribution(n):
        return np.full(n, 0.5)

    generator.set_duration_distribution(custom_distribution)
    durations = generator._get_durations(3)
    np.testing.assert_array_equal(durations, [0.5, 0.5, 0.5])


def test_standard_forcing_generator_get_forcing():
    """Test get_forcing method of StandardForcingGenerator."""
    generator = sp.StandardForcingGenerator()
    times = np.linspace(0, 10, 100)
    waiting_time = 1.0
    forcing = generator.get_forcing(times, waiting_time)
    assert forcing.total_pulses > 0
    assert len(forcing.arrival_times) == forcing.total_pulses
    assert len(forcing.amplitudes) == forcing.total_pulses
    assert len(forcing.durations) == forcing.total_pulses
    assert np.all(forcing.arrival_times >= times[0])
    assert np.all(forcing.arrival_times <= times[-1])


def test_standard_forcing_generator_with_custom_distributions():
    """Test StandardForcingGenerator with custom amplitude and duration distributions."""
    generator = sp.StandardForcingGenerator()
    # Custom amplitude distribution
    generator.set_amplitude_distribution(lambda n: np.random.randint(1, 5, size=n))
    # Custom duration distribution
    generator.set_duration_distribution(lambda n: np.random.uniform(0.1, 0.5, size=n))
    times = np.linspace(0, 10, 100)
    waiting_time = 1.0
    forcing = generator.get_forcing(times, waiting_time)
    assert forcing.total_pulses > 0
    assert len(forcing.arrival_times) == forcing.total_pulses
    assert len(forcing.amplitudes) == forcing.total_pulses
    assert len(forcing.durations) == forcing.total_pulses
    assert np.all(forcing.arrival_times >= times[0])
    assert np.all(forcing.arrival_times <= times[-1])
    assert np.all(forcing.amplitudes >= 1)
    assert np.all(forcing.amplitudes < 5)
    assert np.all(forcing.durations >= 0.1)
    assert np.all(forcing.durations <= 0.5)
