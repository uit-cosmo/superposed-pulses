import superposedpulses as sp
import numpy as np


def test_moments_of_realization():
    """Testing the first two moments of a realization against theory.

    The analytical expressions are provided in Garcia et al. Phys. Plasmas 23, 052308 (2016)

    Equation 16 is used for the mean value of the realization.
    Equation 23 is used for the root mean square value of the realization.

    """
    waiting_time = 3.7
    intermittency = 1 / waiting_time
    average_amplitude = 1
    average_squared_amplitude = 2
    integral_exponential_pulse_shape = 1
    integral_square_exponential_pulse_shape = 0.5

    model = sp.PointModel(waiting_time=waiting_time, total_duration=100_000, dt=0.1)
    times, signal = model.make_realization()

    mean_theory = intermittency * average_amplitude * integral_exponential_pulse_shape
    rms_squared_theory = (
        intermittency
        * average_squared_amplitude
        * integral_square_exponential_pulse_shape
    )

    mean_realization = np.mean(signal)
    rms_squared_realization = np.mean((signal - mean_realization)**2)

    print(mean_realization, mean_theory)
    print(rms_squared_realization, rms_squared_theory)

    np.testing.assert_allclose(mean_realization, mean_theory, rtol=0.05)
    np.testing.assert_allclose(rms_squared_realization, rms_squared_theory, rtol=0.05)
