import superposedpulses as sp
import numpy as np


def test_moments_of_realization():
    """Testing the first two moments of a realization against theory.

    The analytical expressions are provided in the PhD thesis https://munin.uit.no/handle/10037/21170
    on page 38. Comment to reviewer: feel free to exchange the reference with an older one.

    Equation (3.9a) is used for the mean value of the realization.
    Equation (3.9b) is used for the root mean square value of the realization.

    """
    waiting_time = 100
    intermitency = 1 / waiting_time
    average_amplitude = 1
    average_squared_amplitude = 2
    integral_exponential_pulse_shape = 1
    integral_square_exponential_pulse_shape = 0.5

    model = sp.PointModel(waiting_time=waiting_time, total_duration=100_000, dt=0.01)
    times, signal = model.make_realization()

    mean_theory = intermitency * average_amplitude * integral_exponential_pulse_shape
    rms_squared_theory = (
        intermitency
        * average_squared_amplitude
        * integral_square_exponential_pulse_shape
    )

    mean_realization = np.mean(signal)
    rms_squared_realization = np.mean(signal**2)

    np.testing.assert_allclose(mean_realization, mean_theory, rtol=0.10)
    np.testing.assert_allclose(rms_squared_realization, rms_squared_theory, rtol=0.20)
