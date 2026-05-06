# Data

This guide covers the different ways to get gravitational-wave data into Jim.

All detector data in Jim lives inside `Detector` objects. Jim ships with convenience constructors for common detectors — LIGO Hanford (`get_H1`), LIGO Livingston (`get_L1`), Virgo (`get_V1`), Einstein Telescope (`get_ET`), and Cosmic Explorer (`get_CE`):

```python
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1

H1 = get_H1()
L1 = get_L1()
V1 = get_V1()
```

You can also retrieve all presets at once as a dictionary with `get_detector_preset()`:

```python
from jimgw.core.single_event.detector import get_detector_preset

detectors = get_detector_preset()  # {"H1": ..., "L1": ..., "V1": ..., "ET": ..., "CE": ...}
H1 = detectors["H1"]
```

Note that `get_ET()` returns a **list** of three `GroundBased2G` objects (one for each of ET's triangular arms), while all others return a single detector.

Once you have a detector, you need to attach strain data and a PSD to it.

## Loading Data

### Fetch from GWOSC

Use `Data.from_gwosc()` to download public strain data from the [Gravitational Wave Open Science Center](https://gwosc.org/):

```python
from jimgw.core.single_event.data import Data

gps_start = 1126259446
gps_end = 1126259478

data = Data.from_gwosc("H1", gps_start, gps_end)
H1.set_data(data)
```

### Load from File

Use `Data.from_file()` to read a locally saved `.npz` file. The file must contain the keys `td` (time-domain strain), `dt` (time step in seconds), and `start_time` (GPS start time):

```python
data = Data.from_file("path/to/data.npz")
H1.set_data(data)
```

### Construct from Frequency-Domain Arrays

Use `Data.from_fd()` when you already have frequency-domain strain (e.g. from your own pipeline). The frequency array must form a valid rfft grid:

```python
import jax.numpy as jnp

duration = 4.0
sampling_frequency = 2048.0
n = int(duration * sampling_frequency)
frequencies = jnp.fft.rfftfreq(n, 1.0 / sampling_frequency)

fd_strain = jnp.zeros(len(frequencies), dtype=jnp.complex128)  # replace with your data
data = Data.from_fd(fd_strain, frequencies, start_time=0.0)
H1.set_data(data)
```

!!! warning
    The frequency array **must** come from `jnp.fft.rfftfreq`. Internally, `from_fd` reconstructs the full rfft grid and asserts it matches yours exactly. `jnp.linspace` produces floating-point values via a different arithmetic path, so the equality check will fail even for a nominally identical grid.

## PSD

Before you can evaluate a likelihood, each detector also needs a power spectral density (PSD).

### Default GWTC-2 ASD

The simplest approach downloads and sets the default GWTC-2 ASD for the detector:

```python
H1.load_and_set_psd()
```

### Load from a text file

`load_and_set_psd` also accepts paths to two-column whitespace-separated text files (frequency, PSD or ASD):

```python
# From a PSD file (units: Hz^{-1})
H1.load_and_set_psd(psd_file="path/to/psd.txt")

# From an ASD file (units: Hz^{-1/2}) — squared internally to give PSD
H1.load_and_set_psd(asd_file="path/to/asd.txt")
```

### Construct from arrays

Use `set_psd` with a `PowerSpectrum` object when you already have PSD values and frequencies as arrays:

```python
import jax.numpy as jnp
from jimgw.core.single_event.data import PowerSpectrum

psd_values = jnp.array(...)      # PSD in Hz^{-1}
frequencies = jnp.array(...)     # Frequencies in Hz

H1.set_psd(PowerSpectrum(psd_values, frequencies))
```

### Load from a `.npz` file

If you have a previously saved `PowerSpectrum`, reload it with `PowerSpectrum.from_file`. The file must contain `values` and `frequencies` arrays:

```python
H1.set_psd(PowerSpectrum.from_file("path/to/psd.npz"))
```

## Injecting a Simulated Signal

For testing and validation, you can inject a waveform directly into a detector. Set the PSD and frequency bounds first, then call `inject_signal`.

```python
import jax
from jimgw.core.single_event.waveform import RippleIMRPhenomD

gps_time = 1126259462.0

H1.load_and_set_psd()

waveform = RippleIMRPhenomD(f_ref=20.0)
injection_params = {
    "M_c": 28.0, "eta": 0.24,
    "s1_z": 0.0, "s2_z": 0.0,
    "d_L": 440.0, "t_c": 0.0,
    "phase_c": 0.0, "iota": 0.0,
    "psi": 0.3, "ra": 1.5, "dec": 0.5,
}

H1.inject_signal(
    duration=4.0,
    sampling_frequency=2048.0,
    trigger_time=gps_time,
    waveform_model=waveform,
    parameters=injection_params,
    f_min=20.0,
    f_max=1024.0,
    rng_key=jax.random.key(0),
)
```

Set `zero_noise=True` to get a noiseless injection.
