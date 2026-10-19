# Data

This guide covers the different ways to get gravitational-wave data into Jim.

All detector data in Jim lives inside `Detector` objects.
Jim ships with convenience constructors for common detectors — LIGO Hanford (`get_H1`), LIGO Livingston (`get_L1`), Virgo (`get_V1`), Einstein Telescope (`get_ET`), and Cosmic Explorer (`get_CE`):

```python
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1

H1 = get_H1()
L1 = get_L1()
V1 = get_V1()
```

You can also retrieve all presets at once as a dictionary with `get_detector_preset()`:

```python
from jimgw.core.single_event.detector import get_detector_preset

detectors = get_detector_preset()  # {"H1": ..., "L1": ..., "V1": ..., "ET": ..., "ET_2L": ..., "CE": ...}
H1 = detectors["H1"]
```

Note that `get_ET()` returns a **list** of three detectors (one for each of ET's triangular arms) and `get_ET_2L()` a list of two, while all others return a single detector.

Once you have a detector, you need to attach strain data and a PSD to it.

## Loading Data

### Fetch from GWOSC

Use `Data.from_gwosc()` to download public strain data from the [Gravitational Wave Open Science Center](https://gwosc.org/):

```python
from jimgw.core.single_event.data import Data

gps_start = 1126259432.4
gps_end = 1126259464.4

data = Data.from_gwosc("H1", gps_start, gps_end)
H1.set_data(data)
```

### Load from File

`Data.from_file()` detects the file format from the extension and dispatches accordingly.

#### NumPy archive (`.npz`)

The file must contain the keys `td` (time-domain strain), `dt` (time step in seconds), and `start_time` (GPS start time):

```python
data = Data.from_file("path/to/data.npz")
H1.set_data(data)
```

#### GWF / LAL frame file (`.gwf`)

Pass a channel name via the `channel` argument.
If omitted, a set of common LIGO/Virgo channel names is tried automatically:

```python
# Explicit channel (recommended)
data = Data.from_file("path/to/data.gwf", channel="H1:GDS-CALIB_STRAIN")
H1.set_data(data)

# Auto-detect channel from common presets
data = Data.from_file("path/to/data.gwf")
H1.set_data(data)
```

Optional `start_time` and `end_time` (GPS seconds) let you trim the segment read from the frame file.

#### HDF5 file (`.hdf5` / `.h5`)

```python
data = Data.from_file("path/to/data.hdf5", channel="H1:GDS-CALIB_STRAIN")
H1.set_data(data)
```

#### CSV file (`.csv`)

```python
data = Data.from_file("path/to/data.csv")
H1.set_data(data)
```

### Construct from Time-Domain Arrays

Use `Data.from_td()` when you already have time-domain strain:

```python
gps_time = 1126259462.4
data = Data.from_td(
    td_strain, delta_t=1.0 / 4096, start_time=gps_start, trigger_time=gps_time
)
H1.set_data(data)
```

### Construct from Frequency-Domain Arrays

Use `Data.from_fd()` when you already have frequency-domain strain.
The frequency array must form a valid rfft grid:

```python
import jax.numpy as jnp

duration = 4.0
sampling_frequency = 2048.0
n = int(duration * sampling_frequency)
frequencies = jnp.fft.rfftfreq(n, 1.0 / sampling_frequency)

fd = jnp.zeros(len(frequencies), dtype=jnp.complex128)  # replace with your data
data = Data.from_fd(fd, frequencies, start_time=0.0)
H1.set_data(data)
```

## Time-domain windowing

Time-domain data is multiplied by a Tukey window before its Fourier transform.
This applies only to time-domain inputs: configure the window immediately after creating or loading the data, before starting frequency-domain analysis.
Once frequency-domain data have been materialized, `set_tukey_window()` raises an error.
`Data.from_fd()` is already frequency-domain data and cannot be re-windowed.

The default taper has a 0.4-second roll-off on each side.
For a segment of duration \(T\), `roll_off` must be between 0 and \(T / 2\); equivalently, `alpha` must be between 0 and 1.
The two parameterizations cannot be provided together.
Consequently, the default requires a segment at least 0.8 seconds long.

For a segment of duration \(T\), the Tukey shape parameter is:

\[
\alpha = \frac{2\,t_{\mathrm{rolloff}}}{T},
\]

For a time-domain `Data` object, set a different roll-off time or provide the Tukey shape parameter directly:

```python
time_data = Data.from_file("path/to/data.npz")
# Choose one parameterization:
time_data.set_tukey_window(roll_off=0.2)  # seconds on each side
# time_data.set_tukey_window(alpha=0.1)   # direct scipy Tukey parameter
```

## Frequency range of the data

Each detector's data carry a frequency range, the range of frequencies on which they are used.
`set_zero_noise` and `simulate_and_set_noise` set it from their `f_min` and `f_max` arguments.
Data from GWOSC or from `Data.from_td` do not restrict it, so for them you give `f_min` and `f_max` to the likelihood; a `.npz` file written by `to_file` keeps the range it was saved with.
Only injecting into such data needs the range on the data, see [Injecting Simulated Signals](#injecting-simulated-signals).
A likelihood takes its `f_min` and `f_max` from the range of the data unless you pass them, and they may be narrower than the range of the data but not wider.
Because waveforms are not defined at 0 Hz, data without a range need a positive `f_min` in the likelihood.

## PSD

Before you can evaluate a likelihood, each detector also needs a power spectral density (PSD).

### Default design-sensitivity ASD

With no arguments, `load_and_set_psd` sets the design-sensitivity ASD that Jim ships for the detector:

```python
H1.load_and_set_psd()
```

| Detector | Curve | Source |
| --- | --- | --- |
| H1, L1 | Advanced LIGO design | LIGO-T1800044 (Barsotti, Fritschel, Evans, Gras) |
| V1 | Advanced Virgo design | LIGO-P1200087-v42 (LIGO Scientific Collaboration and Virgo Collaboration) |
| ET1, ET2, ET3 | Einstein Telescope 10 km baseline | ET-0482A-25 (ETO Design Task Force) |
| ET_Sar, ET_EMR | Einstein Telescope 15 km baseline | ET-0482A-25 (ETO Design Task Force) |
| CE | Cosmic Explorer 40 km | CE-T2000017 (Cosmic Explorer Project) |

Links, versions, frequency ranges and checksums are in the [noise curve README](https://github.com/GW-JAX-Team/Jim/blob/main/src/jimgw/core/single_event/noise_curves/README.md).
Outside the frequency range of a curve, the value at its nearest end is used, so keep the frequency range of your data inside it.
Please cite the source documents when you use these curves.

### Load from a file

There are two ways to load a PSD from a local file, depending on how much control you need.

#### One-liner: `load_and_set_psd`

`load_and_set_psd` is a convenience method that loads and sets the PSD in a single call.
It supports all the same formats as `PowerSpectrum.from_file` (`.npz`, `.txt`, `.dat`, `.csv`):

```python
# PSD file — any supported format (values in Hz^{-1})
H1.load_and_set_psd(psd_file="path/to/psd.txt")

# ASD file — any supported format (values in Hz^{-1/2}, squared internally)
H1.load_and_set_psd(asd_file="path/to/asd.npz")
```

#### Explicit: `PowerSpectrum.from_file` + `set_psd`

For full control, or when your file is not a plain text file, use `PowerSpectrum.from_file`.
Supported formats:

| Format | Extensions | Notes |
| --- | --- | --- |
| NumPy archive | `.npz` | Must contain `values` ($\mathrm{Hz}^{-1}$, or ASD with `is_asd=True`) and `frequencies` arrays |
| Text / dat | `.txt`, `.dat` | Two-column whitespace-separated: `(frequency, value)` |
| CSV | `.csv` | Two-column comma-separated: `(frequency, value)` |

```python
from jimgw.core.single_event.data import PowerSpectrum

# From an .npz archive
H1.set_psd(PowerSpectrum.from_file("path/to/psd.npz"))

# From a PSD text file (Hz^{-1})
H1.set_psd(PowerSpectrum.from_file("path/to/psd.txt"))

# From an ASD text file (Hz^{-1/2}) — pass is_asd=True to square the values
H1.set_psd(PowerSpectrum.from_file("path/to/asd.txt", is_asd=True))

# From a CSV file
H1.set_psd(PowerSpectrum.from_file("path/to/psd.csv"))
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

## Injecting Simulated Signals

For testing and validation, you can inject waveforms directly into a detector.
Set the PSD first, create the data segment, then call `inject_signal` once per signal.

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

H1.simulate_and_set_noise(
    duration=4.0,
    sampling_frequency=2048.0,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
    rng_key=jax.random.key(0),
)
optimal_snr, match_filtered_snr = H1.inject_signal(waveform, injection_params)
```

Use `set_zero_noise` instead of `simulate_and_set_noise` for a noiseless injection.
`inject_signal` cuts the signal to the frequency range of the data, which these two functions set.
To inject into other data, such as data from GWOSC, set the range first (rarely needed):

```python
H1.data.frequency_bounds = (20.0, 1024.0)
H1.inject_signal(waveform, injection_params)
```
`inject_signal` returns the optimal SNR of the signal and its matched-filter SNR against all the data.
Each call adds one signal to whatever the detector holds, so a second call adds an overlapping signal to the same data:

```python
second_params = {**injection_params, "M_c": 22.0, "eta": 0.2, "d_L": 300.0, "t_c": 0.2}
H1.inject_signal(waveform, second_params)
```

## Next-Generation Detectors

`get_ET()`, `get_ET_2L()` and `get_CE()` return `GroundBased3G` detectors, which respond like `GroundBased2G` unless you turn on one of two options:

- `earth_rotation=True` follows the Earth's rotation while the signal is in band.
  Turn it on for signals that stay in band for ten minutes or longer, such as binary neutron stars observed from about 10 Hz or lower, or for a few minutes if the signal is very loud.
  It needs `M_c`, `eta`, `s1_z` and `s2_z` among the likelihood parameters.
- `finite_arm_length=True` does not assume, as the usual antenna pattern does, that the arms are short compared to the gravitational wavelength.
  The effect grows with frequency and arm length, so turn it on for loud signals (SNR of a few hundred or more) with power at high frequencies, more so for the 40 km arms of CE than for the 10 km arms of ET.
  Unlike the rotation, it does not depend on the duration of the signal.

```python
from jimgw.core.single_event.detector import get_ET

ET = get_ET(earth_rotation=True, finite_arm_length=True)
```

Both are off by default.
Switching either on makes the likelihood more expensive.
The phase convention of the finite-arm-length response is given in the [technical notes](../notes/detector_response.md#finite-arm-length-phase-convention).
