from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class PulseParameters:
    """Container class with the parameters (arrival time, amplitude and
    duration) for a single pulse."""

    def __init__(self, arrival_time: float, amplitude: float, duration: float):
        self.arrival_time = arrival_time
        self.amplitude = amplitude
        self.duration = duration


class Forcing:
    """Container class with the signal forcing containing arrival times,
    amplitudes and durations for all the pulses."""

    def __init__(
        self,
        total_pulses: int,
        arrival_times: np.ndarray,
        amplitudes: np.ndarray,
        durations: np.ndarray,
    ):
        assert total_pulses == len(arrival_times)
        if amplitudes is not None:
            assert total_pulses == len(amplitudes)
        if durations is not None:
            assert total_pulses == len(durations)

        self.total_pulses = total_pulses
        self.arrival_times = arrival_times
        self.amplitudes = amplitudes
        self.durations = durations

    def get_pulse_parameters(self, pulse_index: int) -> PulseParameters:
        return PulseParameters(
            self.arrival_times[pulse_index],
            self.amplitudes[pulse_index],
            self.durations[pulse_index],
        )


class ForcingGenerator(ABC):
    """Abstract class used by PointModel to generate forcing.

    Implementations of this class should have a get_forcing method,
    returning a forcing with arrival times, amplitudes and durations for
    all pulses.
    """

    @abstractmethod
    def get_forcing(self, times: np.ndarray, gamma: float) -> Forcing:
        raise NotImplementedError

    @abstractmethod
    def set_amplitude_distribution(
        self,
        amplitude_distribution_function: Callable[[int], np.ndarray],
    ):
        raise NotImplementedError

    @abstractmethod
    def set_duration_distribution(
        self, duration_distribution_function: Callable[[int], np.ndarray]
    ):
        raise NotImplementedError


class StandardForcingGenerator(ForcingGenerator):
    """Generates a standard forcing, with uniformly distributed arrival times.

    The resulting process is therefore a Poisson process. Amplitude and
    duration distributions can be customized.
    """

    def __init__(self):
        self._amplitude_distribution = None
        self._duration_distribution = None

    def get_forcing(self, times: np.ndarray, gamma: float) -> Forcing:
        total_pulses = int(max(times) * gamma)
        arrival_times = np.random.default_rng().uniform(
            low=times[0], high=times[len(times) - 1], size=total_pulses
        )
        amplitudes = self._get_amplitudes(total_pulses)
        durations = self._get_durations(total_pulses)
        return Forcing(total_pulses, arrival_times, amplitudes, durations)

    def set_amplitude_distribution(
        self,
        amplitude_distribution_function: Callable[[int], np.ndarray],
    ):
        self._amplitude_distribution = amplitude_distribution_function

    def set_duration_distribution(
        self, duration_distribution_function: Callable[[int], np.ndarray]
    ):
        self._duration_distribution = duration_distribution_function

    def _get_amplitudes(self, total_pulses) -> np.ndarray:
        if self._amplitude_distribution is not None:
            return self._amplitude_distribution(total_pulses)
        return np.random.default_rng().exponential(scale=1.0, size=total_pulses)

    def _get_durations(self, total_pulses) -> np.ndarray:
        if self._duration_distribution is not None:
            return self._duration_distribution(total_pulses)
        return np.ones(total_pulses)
