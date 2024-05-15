from typing import Callable
from superposedpulses.forcing import PulseParameters

import numpy as np


class TwoPointForcing:
    """Container class with the signal's forcing containing arrival times,
    amplitudes and durations for all the pulses for both signals.

    Random variables for different pulses are independent, but random
    variables for a single pulse may be correlated, even in different
    points.
    """

    def __init__(
        self,
        total_pulses: int,
        arrival_times: np.ndarray,
        amplitudes_a: np.ndarray,
        durations_a: np.ndarray,
        amplitudes_b: np.ndarray,
        durations_b: np.ndarray,
        delays: np.ndarray,
    ):
        assert total_pulses == len(arrival_times)
        if amplitudes_a is not None:
            assert total_pulses == len(amplitudes_a)
        if durations_a is not None:
            assert total_pulses == len(durations_a)
        if amplitudes_b is not None:
            assert total_pulses == len(amplitudes_b)
        if durations_b is not None:
            assert total_pulses == len(durations_b)
        if delays is not None:
            assert total_pulses == len(delays)

        self.total_pulses = total_pulses
        self.arrival_times = arrival_times
        self.amplitudes_a = amplitudes_a
        self.amplitudes_b = amplitudes_b
        self.durations_a = durations_a
        self.durations_b = durations_b
        self.delays = delays

    def get_pulse_parameters_a(self, pulse_index: int) -> PulseParameters:
        return PulseParameters(
            self.arrival_times[pulse_index],
            self.amplitudes_a[pulse_index],
            self.durations_a[pulse_index],
        )

    def get_pulse_parameters_b(self, pulse_index: int) -> PulseParameters:
        return PulseParameters(
            self.arrival_times[pulse_index] + self.delays[pulse_index],
            self.amplitudes_b[pulse_index],
            self.durations_b[pulse_index],
        )


class TwoPointForcingGenerator:
    """Responsible for generating a forcing for a two point model.

    The forcing consists of a set of amplitudes, durations, delays and arrival times.
        amplitudes_a and amplitudes_b are the amplitudes at each point, respectively. By default,
        they are exponentially distributed and uncorrelated.
        durations_a and durations_b are the duration times at each point, respectively. By
        default, they are equal and degenerate distributed.
        delays are the delays of arrival times between point B and point A. By default,
        they are degenerate distributed.
        arrival_times are the arrival times at point A.
    """

    def __init__(self):
        self._amplitude_distribution = lambda k: np.random.default_rng().exponential(
            size=k
        )
        self._duration_distribution = lambda k: np.ones(k)
        self._delay_distribution = lambda k: np.ones(k)

    def get_forcing(self, times: np.ndarray, waiting_time: float) -> TwoPointForcing:
        total_pulses = int(max(times) / waiting_time )
        arrival_times = np.random.default_rng().uniform(
            low=times[0], high=times[len(times) - 1], size=total_pulses
        )
        amplitudes_a = self._amplitude_distribution(total_pulses)
        durations_a = self._duration_distribution(total_pulses)
        amplitudes_b = self._amplitude_distribution(total_pulses)
        durations_b = durations_a
        delays = self._delay_distribution(total_pulses)
        return TwoPointForcing(
            total_pulses,
            arrival_times,
            amplitudes_a,
            durations_a,
            amplitudes_b,
            durations_b,
            delays,
        )

    def set_amplitude_distribution(
        self,
        amplitude_distribution_function: Callable[[int], np.ndarray],
    ):
        self._amplitude_distribution = amplitude_distribution_function

    def set_duration_distribution(
        self, duration_distribution_function: Callable[[int], np.ndarray]
    ):
        self._duration_distribution = duration_distribution_function

    def set_delay_distribution(
        self, delay_distribution_function: Callable[[int], np.ndarray]
    ):
        self._delay_distribution = delay_distribution_function
