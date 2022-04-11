from abc import ABC, abstractmethod
import numpy as np
from typing import Callable
import warnings


class PulseGenerator(ABC):
    """Abstract pulse shape, implementations should return a pulse shape vector
    given an array of time stamps with (possibly) different durations.

    For long time series this becomes expensive, and a ShortPulseShape
    implementation is preferred if the pulse function decays fast to 0
    or under machine error.
    """

    @abstractmethod
    def get_pulse(self, times: np.ndarray, duration: float) -> np.ndarray:
        """
        Abstract method, implementations should return np.ndarray with pulse shape with the same length as times.
        Parameters
        ----------
        times time array under which the pulse function is to be evaluated
        duration Duration time

        Returns
        -------
        np.ndarray with the pulse shape,

        """
        raise NotImplementedError


class StandardPulseGenerator(PulseGenerator):
    """Generates all pulse shapes previously supported."""

    __SHAPE_NAMES__ = {"1-exp", "lorentz", "2-exp", "gaussian"}
    # TODO: Implement the others

    def __init__(self, shape_name: str = "1-exp", **kwargs):
        """
        Parameters
        ----------
        shape_name Should be one of StandardPulseShapeGenerator.__SHAPE_NAMES__
        kwargs Additional arguments to be passed to special shapes:
            - "2-exp":
            - "lam" parameter for the asymmetry parameter
        """
        assert (
            shape_name in StandardPulseGenerator.__SHAPE_NAMES__
        ), "Invalid shape_name"
        self._shape_name: str = shape_name
        self._kwargs = kwargs

    def get_pulse(
        self, times: np.ndarray, duration: float, tolerance: float = 1e-5
    ) -> np.ndarray:

        kern = self._get_generator(self._shape_name)(times, duration, self._kwargs)
        err = max(np.abs(kern[0]), np.abs(kern[-1]))
        if err > tolerance:
            warnings.warn("Value at end point of kernel > tol, end effects may occur.")
        return kern

    @staticmethod
    def _get_generator(
        shape_name: str,
    ) -> Callable[[np.ndarray, float, dict], np.ndarray]:
        if shape_name == "1-exp":
            return StandardPulseGenerator._get_exponential_shape
        if shape_name == "2-exp":
            return StandardPulseGenerator._get_double_exponential_shape
        if shape_name == "lorentz":
            return StandardPulseGenerator._get_lorentz_shape
        if shape_name == "gaussian":
            return StandardPulseGenerator._get_gaussian_shape

    @staticmethod
    def _get_exponential_shape(
        times: np.ndarray, duration: float, kwargs
    ) -> np.ndarray:
        kern = np.zeros(len(times))
        kern[times >= 0] = np.exp(-times[times >= 0] / duration)
        return kern

    @staticmethod
    def _get_lorentz_shape(times: np.ndarray, duration: float, kwargs) -> np.ndarray:
        return (np.pi * (1 + (times / duration) ** 2)) ** (-1)

    @staticmethod
    def _get_double_exponential_shape(
        times: np.ndarray, duration: float, kwargs
    ) -> np.ndarray:
        lam = kwargs["lam"]
        assert (lam > 0.0) & (lam < 1.0)
        kern = np.zeros(len(times))
        kern[times < 0] = np.exp(times[times < 0] / lam / duration)
        kern[times >= 0] = np.exp(-times[times >= 0] / (1 - lam) / duration)
        return kern

    @staticmethod
    def _get_gaussian_shape(times: np.ndarray, duration: float, kwargs) -> np.ndarray:
        return np.exp(-((times / duration) ** 2) / 2) / np.sqrt(2 * np.pi)


class ShortPulseGenerator(ABC):
    """Abstract pulse shape, implementations should return a pulse shape vector
    with (possibly) different durations.

    The length of the returned array is not restricted, this is useful
    for pulse shapes such as the exponential or the box pulse, for which
    the signal becomes zero or under machine error very quickly.

    Implementations are responsible of deciding where to place the cutoff for the returned array.
    """

    def __init__(self, tolerance: float = 1e-50):
        self.tolerance = tolerance

    @abstractmethod
    def get_pulse(self, times: np.ndarray, duration: float) -> np.ndarray:
        """
        Abstract method, implementations should return np.ndarray with pulse shape. The returned array should contain
        the values of the shape for the times [-T, T] with sampling dt, where T is such that the value of the pulse
        function is under self.tolerance. The center of the pulse should be located at the center of the array.
        Parameters
        ----------
        dt Sampling time
        duration Duration time

        Returns
        -------
        np.ndarray with the pulse shape,

        """
        raise NotImplementedError

    @abstractmethod
    def get_cutoff(self, duration: float) -> float:
        """
        Abstract method, implementations should return the cutoff above (and below) which the pulse becomes negligible,
        i.e. return T such that abs(p(T/duration)) < self.tolerance and abs(p(-T/duration)) < self.tolerance
        Parameters
        ----------
        duration duration of the pulse

        Returns
        -------
        Cutoff to cut the pulse
        """
        raise NotImplementedError


class ExponentialShortPulseGenerator(ShortPulseGenerator):
    def __init__(
        self, lam: float = 0, tolerance: float = 1e-50, max_cutoff: float = 1e50
    ):
        """Exponential pulse generator, the length of the returned array is
        dynamically set to be the shortest to reach a pulse value under the
        given tolerance. That is, if the pulse shape is p(t), the returned
        array will be p(t) with t in [-T, T] such that p(-T), p(T) < tolerance.

        If a lam argument is provided different than 0, the pulse shape will be
        a double exponential:

        p(t) = exp(t/lam) for t<0; p(t) = exp(-t/(1-lam)) for t>=0.

        A max_cutoff is provided to avoid returning pulse arrays of arbitrarily long lengths.
        Parameters
        ----------
        tolerance Maximum error when cutting the pulse.
        max_cutoff
        lam Asymmetry parameter, defaults to 0 (exponential pulse shape).
        """
        super(ExponentialShortPulseGenerator, self).__init__(tolerance)
        self._max_cutoff = max_cutoff
        self.lam = lam

    def get_pulse(self, times: np.ndarray, duration: float) -> np.ndarray:
        kern = np.zeros(len(times))
        if self.lam == 0:
            kern[times >= 0] = np.exp(-times[times >= 0] / duration)
            return kern
        kern[times >= 0] = np.exp(-times[times >= 0] / (duration * (1 - self.lam)))
        kern[times < 0] = np.exp(times[times < 0] / (duration * self.lam))
        return kern

    def get_cutoff(self, duration: float) -> float:
        cutoff = -duration * np.log(self.tolerance)
        return min(cutoff, self._max_cutoff)


class LorentzShortPulseGenerator(ShortPulseGenerator):
    def __init__(self, tolerance: float = 1e-50, max_cutoff: float = 1e50):
        """Lorentz pulse generator, the length of the returned array is
        dynamically set to be the shortest to reach a pulse value under the
        given tolerance. That is, if the pulse shape is p(t), the returned
        array will be p(t) with t in [-T, T] such that p(-T), p(T) < tolerance.

        p(t) = 1/ (pi (t^2 + 1)).

        A max_cutoff is provided to avoid returning pulse arrays of arbitrarily long lengths.
        Parameters
        ----------
        tolerance Maximum error when cutting the pulse.
        max_cutoff
        """
        super(LorentzShortPulseGenerator, self).__init__(tolerance)
        self._max_cutoff = max_cutoff

    def get_pulse(self, times: np.ndarray, duration: float) -> np.ndarray:
        kern = np.zeros(len(times))
        kern = (np.pi * (1 + (times / duration) ** 2)) ** (-1)
        return kern

    def get_cutoff(self, duration: float) -> float:
        cutoff = duration * np.sqrt(1.0 / (self.tolerance * np.pi) - 1)
        return min(cutoff, self._max_cutoff)


class GaussianShortPulseGenerator(ShortPulseGenerator):
    def __init__(self, tolerance: float = 1e-50, max_cutoff: float = 1e50):
        """
        Gaussian pulse generator. The length of the returned array is
        dynamically set to be the shortest to reach a pulse value under the
        given tolerance. That is, if the pulse shape is p(t), the returned
        array will be p(t) with t in [-T, T] such that p(-T), p(T) < tolerance.

        p(t) = exp(-(t)^2 / 2) / sqrt(2*pi).

        A max_cutoff is provided to avoid returning pulse arrays of arbitrarily long lengths.
        Parameters
        ----------
        tolerance Maximum error when cutting the pulse.
        max_cutoff

        """
        super(GaussianShortPulseGenerator, self).__init__(tolerance)
        self._max_cutoff = max_cutoff

    def get_pulse(self, times: np.ndarray, duration: float) -> np.ndarray:
        kern = np.zeros(len(times))
        kern = np.exp(-((times / duration) ** 2) / 2) / np.sqrt(2 * np.pi)
        return kern

    def get_cutoff(self, duration: float) -> float:
        cutoff = duration * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * self.tolerance))
        return min(cutoff, self._max_cutoff)


class BoxShortPulseGenerator(ShortPulseGenerator):
    """Box shape p(t, tau):

    p(t, tau) = 1, for |t| < tau / 2
    p(t, tau) = 0, otherwise
    """

    def __init__(self, tolerance: float = 1e-50):
        super(BoxShortPulseGenerator, self).__init__(tolerance)

    def get_pulse(self, times: np.ndarray, duration: float) -> np.ndarray:
        kern = np.zeros(len(times))
        kern[abs(times) < duration / 2] = 1
        return kern

    def get_cutoff(self, duration: float) -> float:
        return duration


class TriangularShortPulseGenerator(ShortPulseGenerator):
    """Triangular shape p(t, tau):

    p(t, tau) = 1 - |t|/tau, for |t| < tau
    p(t, tau) = 0, otherwise
    """

    def __init__(self, tolerance: float = 1e-50):
        super(TriangularShortPulseGenerator, self).__init__(tolerance)

    def get_pulse(self, times: np.ndarray, duration: float) -> np.ndarray:
        kern = np.zeros(len(times))
        kern[abs(times) < duration] = 1 - abs(times[abs(times) < duration]) / duration
        return kern

    def get_cutoff(self, duration: float) -> float:
        return duration
