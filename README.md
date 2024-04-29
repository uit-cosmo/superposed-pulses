# superposed-pulses
Collection of tools designed to generate realizations of the Poisson point process.

## Installation
The package is published to PyPI and can be installed with

```sh
pip install superposed-pulses
```

If you want the development version you must first clone the repo to your local machine,
then install the project in development mode:

```sh
git clone https://github.com/uit-cosmo/superposed-pulses.git
cd superposed-pulses
pip install -e .
```
## Usage
The simplest case, using defaults: exponential pulse shape, exponentially distributed amplitudes, constant duration times, write
```Python
import matplotlib.pyplot as plt
import superposedpulses.point_model as pm

model = pm.PointModel(waiting_time=10.0, total_duration=100, dt=0.01)
times, signal = model.make_realization()

plt.plot(times, signal)
plt.show()
```
Take a look at `superposed-pulses/superposedpulses/example.py` to find out how to change amplitudes, waiting times, duration times and the pulse shape of the process.
