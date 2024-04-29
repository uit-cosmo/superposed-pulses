from abc import ABC
from typing import Callable, Tuple, Union
import warnings

import numpy as np
from tqdm import tqdm
from superposedpulses.forcing import (
    Forcing,
    StandardForcingGenerator,
    ForcingGenerator,
    PulseParameters,
)
from superposedpulses.pulse_shape import (
    ShortPulseGenerator,
    ExponentialShortPulseGenerator,
    PulseGenerator,
)
from superposedpulses.two_point_forcing import TwoPointForcingGenerator
from scipy.signal import fftconvolve

__COMMON_DISTRIBUTIONS__ = ["exp", "deg"]


def _get_common_distribution(
    distribution_name: str, average: float
) -> Callable[[int], np.ndarray]:
    if distribution_name == "exp":
        return lambda k: np.random.default_rng().exponential(scale=average, size=k)
    elif distribution_name == "deg":
        return lambda k: average * np.ones(k)
    else:
        raise NotImplementedError


class AbstractModel(ABC):
    """
    Abstract class for FPP Models containing commonly used methods.
    Parameters
    ----------
    waiting_time Average time between pulses
    total_duration Total duration of the process
    dt Time step

    """

    def __init__(self, waiting_time: float, total_duration: float, dt: float):
        self.waiting_time = waiting_time
        self.T = total_duration
        self.dt = dt
        self._times: np.ndarray = np.arange(0, total_duration, dt)
        self._pulse_generator: ShortPulseGenerator = ExponentialShortPulseGenerator()
        self._last_used_forcing = None

    def _add_pulse_to_signal(
        self, signal: np.ndarray, pulse_parameters: PulseParameters
    ):
        """
        Adds a pulse to the provided signal array. Uses self._pulse_generator to generate the pulse shape, this can
        either be a ps.PulseGenerator or a ps.ShortPulseGenerator.
        Parameters
        ----------
        signal Signal array under construction
        pulse_parameters Parameters of the current pulse

        """
        if isinstance(self._pulse_generator, PulseGenerator):
            signal += pulse_parameters.amplitude * self._pulse_generator.get_pulse(
                self._times - pulse_parameters.arrival_time,
                pulse_parameters.duration,
            )
            return

        if isinstance(self._pulse_generator, ShortPulseGenerator):
            cutoff = self._pulse_generator.get_cutoff(pulse_parameters.duration)
            from_index = max(int((pulse_parameters.arrival_time - cutoff) / self.dt), 0)
            to_index = min(
                int((pulse_parameters.arrival_time + cutoff) / self.dt),
                len(self._times),
            )

            pulse = pulse_parameters.amplitude * self._pulse_generator.get_pulse(
                self._times[from_index:to_index] - pulse_parameters.arrival_time,
                pulse_parameters.duration,
            )
            signal[from_index:to_index] += pulse
            return

        raise NotImplementedError(
            "Pulse shape has to inherit from PulseShape or ShortPulseShape"
        )

    def get_last_used_forcing(self):
        """
        Returns the latest used forcing. If several realizations of the process are run only the latest forcing will be
        available.
        -------
        """
        return self._last_used_forcing

    def set_pulse_shape(
        self, pulse_generator: Union[PulseGenerator, ShortPulseGenerator]
    ):
        """
        Parameters
        ----------
        pulse_shape Instance of PulseShape, get_pulse will be called for each pulse when making a realization.
        """
        self._pulse_generator = pulse_generator


class PointModel(AbstractModel):
    """PointModel is a container for all model parameters and is responsible for
    generating a realization of the process through make_realization.

    Uses a ForcingGenerator to generate the forcing, this is by default
    a StandardForcingGenerator.

    Parameters
    ----------
    waiting_time Average time between pulses
    total_duration Total duration of the process
    dt Time step
    """

    def __init__(self, waiting_time: float, total_duration: float, dt: float):
        super(PointModel, self).__init__(waiting_time, total_duration, dt)
        self._forcing_generator: ForcingGenerator = StandardForcingGenerator()
        self._noise = None

    def make_realization(self) -> Tuple[np.ndarray, np.ndarray]:
        result = np.zeros(len(self._times))
        forcing = self._forcing_generator.get_forcing(self._times, waiting_time=self.waiting_time)

        for k in tqdm(range(forcing.total_pulses), position=0, leave=True):
            pulse_parameters = forcing.get_pulse_parameters(k)
            self._add_pulse_to_signal(result, pulse_parameters)

        if self._noise is not None:
            result += self._discretize_noise(forcing)

        self._last_used_forcing = forcing

        return self._times, result

    def set_custom_forcing_generator(self, forcing_generator: ForcingGenerator):
        self._forcing_generator = forcing_generator

    def set_amplitude_distribution(
        self, amplitude_distribution: str, average_amplitude: float = 1.0
    ):
        """Sets the amplitude distribution to be used by the forcing.

        Args:
            amplitude_distribution: str
                'exp': exponential with scale parameter average_amplitude
                'deg': degenerate with location average_amplitude
            average_amplitude: float, defaults to 1.
        """
        if amplitude_distribution in __COMMON_DISTRIBUTIONS__:
            self._forcing_generator.set_amplitude_distribution(
                _get_common_distribution(amplitude_distribution, average_amplitude)
            )
        else:
            raise NotImplementedError

    def set_duration_distribution(
        self, duration_distribution: str, average_duration: float = 1.0
    ):
        """Sets the amplitude distribution to be used by the forcing.

        Args:
            duration_distribution: str
                'exp': exponential with scale parameter average_duration
                'deg': degenerate with location average_duration
            average_duration: float, defaults to 1.
        """
        if duration_distribution in __COMMON_DISTRIBUTIONS__:
            self._forcing_generator.set_duration_distribution(
                _get_common_distribution(duration_distribution, average_duration)
            )
        else:
            raise NotImplementedError

    def add_noise(
        self,
        noise_to_signal_ratio: float,
        seed: Union[None, int] = None,
        noise_type: str = "additive",
    ) -> None:
        """
        Specifies noise for realization.
        Parameters
        ----------
        noise_to_signal_ratio: float, defined as X_rms/S_rms where X is noise and S is signal.
        seed: None or int, seed for the noise generator
        noise_type: str
            "additive": additive noise
            "dynamic": dynamic noise (only applicable for constant duration times)
            "both": both additive and dynamic noise
        """
        assert noise_type in {"additive", "dynamic", "both"}
        assert seed is None or isinstance(seed, int)
        assert noise_to_signal_ratio >= 0

        self._noise_type = noise_type
        self._noise_random_number_generator = np.random.RandomState(seed=seed)

        warnstring = """Calculation of noise rms 
                        (1) assumes average duration time is 1
                        (2) uses numerical mean amplitude"""
        warnings.warn(warnstring)
        mean_amplitude = self._forcing_generator.get_forcing(
            self._times, waiting_time=self.waiting_time
        ).amplitudes.mean()
        gamma = 1./self.waiting_time

        self._sigma = np.sqrt(noise_to_signal_ratio * gamma) * mean_amplitude

        self._noise = np.zeros(len(self._times))

    def _discretize_noise(self, forcing: Forcing) -> np.ndarray:
        """Discretizes noise for the realization"""

        if self._noise_type in {"additive", "both"}:
            self._noise += self._sigma * self._noise_random_number_generator.normal(
                size=len(self._times)
            )

        if self._noise_type in {"dynamic", "both"}:
            durations = forcing.durations
            pulse_duration_constant = np.all(durations == durations[0])
            assert (
                pulse_duration_constant
            ), "Dynamic noise is only applicable for constant duration times."

            kern = self._pulse_generator.get_pulse(
                np.arange(-self._times[-1] / 2, self._times[-1] / 2, self.dt),
                durations[0],
            )
            dW = self._noise_random_number_generator.normal(
                scale=np.sqrt(2 * self.dt), size=len(self._times)
            )
            self._noise += self._sigma * fftconvolve(dW, kern, "same")

        return self._noise


class TwoPointModel(AbstractModel):
    def __init__(self, waiting_time: float, total_duration: float, dt: float):
        super(TwoPointModel, self).__init__(waiting_time, total_duration, dt)
        self._forcing_generator: TwoPointForcingGenerator = TwoPointForcingGenerator()

    def make_realization(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        signal_a = np.zeros(len(self._times))
        signal_b = np.zeros(len(self._times))
        forcing = self._forcing_generator.get_forcing(self._times, waiting_time=self.waiting_time)

        for k in tqdm(range(forcing.total_pulses), position=0, leave=True):
            pulse_parameters_a = forcing.get_pulse_parameters_a(k)
            pulse_parameters_b = forcing.get_pulse_parameters_b(k)
            self._add_pulse_to_signal(signal_a, pulse_parameters_a)
            self._add_pulse_to_signal(signal_b, pulse_parameters_b)

        self._last_used_forcing = forcing
        return self._times, signal_a, signal_b

    def set_custom_forcing_generator(self, forcing_generator: TwoPointForcingGenerator):
        self._forcing_generator = forcing_generator
