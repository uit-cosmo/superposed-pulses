import matplotlib.pyplot as plt
import numpy as np

import model.forcing as frc
import model.point_model as pm
import model.pulse_shape as ps

# Simplest case, using defaults: exponential pulse shape, exponentially distributed amplitudes, constant duration times.

model = pm.PointModel(gamma=0.1, total_duration=100, dt=0.01)
times, signal = model.make_realization()

plt.plot(times, signal)
plt.show()

# Double exponential shape

model = pm.PointModel(gamma=0.1, total_duration=100, dt=0.01)
model.set_pulse_shape(ps.StandardPulseGenerator("2-exp", lam=0.35))
times, signal = model.make_realization()

plt.plot(times, signal)
plt.show()


# Say you want to customise your model a bit: use constant amplitude distribution, and box pulse shapes

model = pm.PointModel(gamma=0.1, total_duration=100, dt=0.01)
model.set_amplitude_distribution("deg")
model.set_pulse_shape(ps.BoxShortPulseGenerator())

times, signal = model.make_realization()

plt.plot(times, signal)
plt.show()

# If you want to implement your own distributions, you can do so by setting a custom ForcingGenerator. Say you want half
# of your pulses to have amplitude 1, and the other half to have amplitude 2.

model = pm.PointModel(gamma=0.1, total_duration=100, dt=0.01)
my_forcing_gen = frc.StandardForcingGenerator()
my_forcing_gen.set_amplitude_distribution(
    lambda k: np.random.randint(low=1, high=3, size=k)
)

model.set_custom_forcing_generator(my_forcing_gen)
times, s = model.make_realization()

plt.plot(times, s)
plt.show()


# Say you want to do something more fancy, for example make a joint distribution of the pulse parameters, then
# you will have to implement your own ForcingGenerator, which is as easy as inheriting from frc.ForcingGenerator and
# set whatever you want in the get_forcing method


class MyFancyForcingGenerator(frc.ForcingGenerator):
    def __init__(self):
        pass

    def get_forcing(self, times: np.ndarray, gamma: float) -> frc.Forcing:
        total_pulses = int(max(times) * gamma)
        arrival_time_indx = np.random.randint(0, len(times), size=total_pulses)
        amplitudes = np.random.default_rng().exponential(scale=1.0, size=total_pulses)
        durations = amplitudes + np.abs(
            0, np.random.normal(loc=0, scale=0.5, size=len(amplitudes))
        )
        return frc.Forcing(
            total_pulses, times[arrival_time_indx], amplitudes, durations
        )

    def set_amplitude_distribution(
        self,
        amplitude_distribution_function,
    ):
        pass

    def set_duration_distribution(self, duration_distribution_function):
        pass


model = pm.PointModel(gamma=10, total_duration=1000, dt=0.01)
model.set_custom_forcing_generator(MyFancyForcingGenerator())

times, s = model.make_realization()

plt.plot(times, s)
plt.show()

# You might also want to get the actual pulse parameters that were used in the realization, this can be done by getting
# the forcing, which is the set of amplitudes, arrival times and durations for each pulse.

forcing = model.get_last_used_forcing()
plt.hist(forcing.durations)
plt.show()
