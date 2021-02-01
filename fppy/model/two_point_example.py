import matplotlib.pyplot as plt
import numpy as np

import fppy.model.forcing as frc
import fppy.model.fpp_model as fpp
import fppy.model.pulse_shape as ps

# Simplest case, using defaults: exponential pulse shape, exponentially distributed amplitudes, constant duration times.

model = fpp.TwoPointFPPModel(gamma=0.1, total_duration=100, dt=0.01)
model.set_pulse_shape(ps.ExponentialShortPulseGenerator(tolerance=1e-50))
times, signal_a, signal_b = model.make_realization()

plt.plot(times, signal_a)
plt.plot(times, signal_b)
plt.show()
