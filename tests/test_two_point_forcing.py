import superposedpulses as sp
import numpy as np


def test_two_point_forcing_initialization():
    """Test initialization of TwoPointForcing."""
    arrival_times = np.array([1.0, 2.0, 3.0])
    amplitudes_a = np.array([0.5, 1.0, 1.5])
    durations_a = np.array([0.1, 0.2, 0.3])
    amplitudes_b = np.array([0.6, 1.1, 1.6])
    durations_b = np.array([0.1, 0.2, 0.3])
    delays = np.array([0.05, 0.1, 0.15])
    forcing = sp.TwoPointForcing(
        total_pulses=3,
        arrival_times=arrival_times,
        amplitudes_a=amplitudes_a,
        durations_a=durations_a,
        amplitudes_b=amplitudes_b,
        durations_b=durations_b,
        delays=delays,
    )
    assert forcing.total_pulses == 3
    np.testing.assert_array_equal(forcing.arrival_times, arrival_times)
    np.testing.assert_array_equal(forcing.amplitudes_a, amplitudes_a)
    np.testing.assert_array_equal(forcing.durations_a, durations_a)
    np.testing.assert_array_equal(forcing.amplitudes_b, amplitudes_b)
    np.testing.assert_array_equal(forcing.durations_b, durations_b)
    np.testing.assert_array_equal(forcing.delays, delays)


def test_two_point_forcing_get_pulse_parameters_a():
    """Test get_pulse_parameters_a method of TwoPointForcing."""
    arrival_times = np.array([1.0, 2.0, 3.0])
    amplitudes_a = np.array([0.5, 1.0, 1.5])
    durations_a = np.array([0.1, 0.2, 0.3])
    amplitudes_b = np.array([0.6, 1.1, 1.6])
    durations_b = np.array([0.1, 0.2, 0.3])
    delays = np.array([0.05, 0.1, 0.15])
    forcing = sp.TwoPointForcing(
        total_pulses=3,
        arrival_times=arrival_times,
        amplitudes_a=amplitudes_a,
        durations_a=durations_a,
        amplitudes_b=amplitudes_b,
        durations_b=durations_b,
        delays=delays,
    )
    pulse = forcing.get_pulse_parameters_a(1)
    assert pulse.arrival_time == 2.0
    assert pulse.amplitude == 1.0
    assert pulse.duration == 0.2


def test_two_point_forcing_get_pulse_parameters_b():
    """Test get_pulse_parameters_b method of TwoPointForcing."""
    arrival_times = np.array([1.0, 2.0, 3.0])
    amplitudes_a = np.array([0.5, 1.0, 1.5])
    durations_a = np.array([0.1, 0.2, 0.3])
    amplitudes_b = np.array([0.6, 1.1, 1.6])
    durations_b = np.array([0.1, 0.2, 0.3])
    delays = np.array([0.05, 0.1, 0.15])
    forcing = sp.TwoPointForcing(
        total_pulses=3,
        arrival_times=arrival_times,
        amplitudes_a=amplitudes_a,
        durations_a=durations_a,
        amplitudes_b=amplitudes_b,
        durations_b=durations_b,
        delays=delays,
    )
    pulse = forcing.get_pulse_parameters_b(1)
    assert pulse.arrival_time == 2.1  # arrival_time + delay
    assert pulse.amplitude == 1.1
    assert pulse.duration == 0.2


def test_two_point_forcing_generator_default_distributions():
    """Test default distributions in TwoPointForcingGenerator."""
    generator = sp.TwoPointForcingGenerator()
    total_pulses = 5
    amplitudes = generator._amplitude_distribution(total_pulses)
    durations = generator._duration_distribution(total_pulses)
    delays = generator._delay_distribution(total_pulses)
    assert len(amplitudes) == total_pulses
    assert len(durations) == total_pulses
    assert len(delays) == total_pulses
    assert np.all(amplitudes > 0)  # Exponential distribution is positive
    np.testing.assert_array_equal(durations, np.ones(total_pulses))
    np.testing.assert_array_equal(delays, np.ones(total_pulses))


def test_two_point_forcing_generator_custom_distributions():
    """Test custom distributions in TwoPointForcingGenerator."""
    generator = sp.TwoPointForcingGenerator()
    generator.set_amplitude_distribution(lambda n: np.full(n, 2.0))
    generator.set_duration_distribution(lambda n: np.full(n, 0.5))
    generator.set_delay_distribution(lambda n: np.full(n, 0.1))
    amplitudes = generator._amplitude_distribution(3)
    durations = generator._duration_distribution(3)
    delays = generator._delay_distribution(3)
    np.testing.assert_array_equal(amplitudes, [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(durations, [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(delays, [0.1, 0.1, 0.1])


def test_two_point_forcing_generator_get_forcing():
    """Test get_forcing method of TwoPointForcingGenerator."""
    generator = sp.TwoPointForcingGenerator()
    times = np.linspace(0, 10, 100)
    waiting_time = 1.0
    forcing = generator.get_forcing(times, waiting_time)
    assert forcing.total_pulses > 0
    assert len(forcing.arrival_times) == forcing.total_pulses
    assert len(forcing.amplitudes_a) == forcing.total_pulses
    assert len(forcing.amplitudes_b) == forcing.total_pulses
    assert len(forcing.durations_a) == forcing.total_pulses
    assert len(forcing.durations_b) == forcing.total_pulses
    assert len(forcing.delays) == forcing.total_pulses
    assert np.all(forcing.arrival_times >= times[0])
    assert np.all(forcing.arrival_times <= times[-1])


def test_two_point_forcing_generator_with_custom_distributions():
    """Test TwoPointForcingGenerator with custom distributions."""
    generator = sp.TwoPointForcingGenerator()
    generator.set_amplitude_distribution(lambda n: np.random.randint(1, 5, size=n))
    generator.set_duration_distribution(lambda n: np.random.uniform(0.1, 0.5, size=n))
    generator.set_delay_distribution(lambda n: np.random.uniform(0.0, 0.2, size=n))
    times = np.linspace(0, 10, 100)
    waiting_time = 1.0
    forcing = generator.get_forcing(times, waiting_time)
    assert forcing.total_pulses > 0
    assert len(forcing.arrival_times) == forcing.total_pulses
    assert len(forcing.amplitudes_a) == forcing.total_pulses
    assert len(forcing.amplitudes_b) == forcing.total_pulses
    assert len(forcing.durations_a) == forcing.total_pulses
    assert len(forcing.durations_b) == forcing.total_pulses
    assert len(forcing.delays) == forcing.total_pulses
    assert np.all(forcing.amplitudes_a >= 1)
    assert np.all(forcing.amplitudes_a < 5)
    assert np.all(forcing.durations_a >= 0.1)
    assert np.all(forcing.durations_a <= 0.5)
    assert np.all(forcing.delays >= 0.0)
    assert np.all(forcing.delays <= 0.2)
