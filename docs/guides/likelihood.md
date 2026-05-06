# Likelihood

The likelihood connects your detector data with a waveform model and scores how well a set of source parameters explains the observed strain.

## Waveform Model

Jim uses [ripple](https://github.com/GW-JAX-Team/ripple) waveform models, which are JAX-native and fully differentiable. Import any available model from `jimgw.core.single_event.waveform`:

```python
from jimgw.core.single_event.waveform import RippleIMRPhenomD

waveform = RippleIMRPhenomD(f_ref=20.0)
```

See the [ripple documentation](https://gw-jax-team.github.io/ripple/) for the full list of available waveforms (aligned-spin, precessing, tidal, burst, etc.).

## TransientLikelihoodFD

`TransientLikelihoodFD` is the standard frequency-domain likelihood for transient gravitational-wave signals:

```python
from jimgw.core.single_event.likelihood import TransientLikelihoodFD

likelihood = TransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
)
```

### Key Parameters

| Parameter | Description |
| --- | --- |
| `detectors` | List of `Detector` objects with data and PSD already set |
| `waveform` | A ripple waveform model instance |
| `trigger_time` | GPS trigger time of the event |
| `f_min` / `f_max` | Frequency range for the likelihood integral. Can be a single float (applied to all detectors) or a `dict[str, float]` keyed by detector name |
| `fixed_parameters` | Dictionary of parameter values to hold fixed during sampling |

### Analytic Marginalisation

The likelihood supports analytic marginalisation over coalescence time, phase, and/or luminosity distance. Each is activated by passing a typed config object or a plain dict shorthand:

```python
from jimgw.core.prior import PowerLawPrior

distance_prior = PowerLawPrior(xmin=100.0, xmax=5000.0, alpha=2.0, parameter_names=["d_L"])

likelihood = TransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
    time_marginalization={"tc_range": (-0.1, 0.1)},               # or time_marginalization=True for default range
    phase_marginalization=True,                                   # shorthand for PhaseMargConfig()
    distance_marginalization={"distance_prior": distance_prior},  # required: distance_prior
)
```

Marginalising over these parameters reduces the effective dimensionality of the problem and can significantly speed up sampling.

- `time_marginalization` — marginalises over `t_c` within the range set by `tc_range` (default `(-0.1, 0.1)`). Pass `{}` to use the default range, or `{"tc_range": (lo, hi)}` for a custom range.
- `phase_marginalization` — marginalises over `phase_c`. Pass `True`, `{}`, or a `PhaseMargConfig()` instance.
- `distance_marginalization` — marginalises over `d_L`. Pass a dict with `distance_prior` (a 1-D prior over luminosity distance) and optionally `n_dist_points` and `ref_dist`. Unlike the other two options, `True` is **not** supported and will raise a `ValueError` because `distance_prior` has no default; always pass `{"distance_prior": ...}`.

### Fixing Parameters

To fix some parameters at known values (e.g. for testing or when marginalising externally), pass them via `fixed_parameters`:

```python
likelihood = TransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
    fixed_parameters={
        "s1_z": 0.0,
        "s2_z": 0.0,
        "iota": 0.4,
    },
)
```

These values are automatically merged with the sampled parameters at evaluation time.

#### Derived fixed parameters (callables)

Sometimes the value you want to fix is not a constant but depends on other sampled parameters. A common example: you want to fix the detector arrival time `t_det` rather than the geocentric coalescence time `t_c`. The two are related by

$$t_c = t_{\text{det}} - \Delta t(\text{ra}, \text{dec})$$

so `t_c` depends on sky location, which is sampled. Passing a plain number for `"t_c"` would not capture this.

For this case every value in `fixed_parameters` may also be a **callable** `f(params) -> value`. The callable receives the full parameter dict at evaluation time and must return either a scalar or a full dict. When a dict is returned, Jim extracts only the value for the key being fixed.

The cleanest way to express this is to reuse the same transform you already define for Jim's likelihood-transform pipeline and pass its `backward` method directly:

```python
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
)

# Maps t_det -> t_c, conditional on (ra, dec)
transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
    trigger_time=trigger_time, ifo=H1
)

likelihood = TransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
    # transform.backward returns a dict; Jim extracts params["t_c"] automatically
    fixed_parameters={"t_c": transform.backward},
)
```

Alternatively use a plain lambda:

```python
from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time

gmst = greenwich_mean_sidereal_time(trigger_time)
t_det_value = 0.0  # the value you are fixing

likelihood = TransientLikelihoodFD(
    ...,
    fixed_parameters={
        "t_c": lambda p: t_det_value - H1.delay_from_geocenter(p["ra"], p["dec"], gmst),
    },
)
```

Both forms are `jax.jit`-compatible. Callables are evaluated in **insertion order**, so later entries in `fixed_parameters` can read values written by earlier ones.

## HeterodynedTransientLikelihoodFD

For faster evaluation, `HeterodynedTransientLikelihoodFD` uses the heterodyne (relative binning) technique.  It requires a set of *reference parameters* around which the binning is constructed.

### Providing reference parameters directly

```python
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD

likelihood = HeterodynedTransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
    reference_parameters=ref_params,  # dict with all waveform parameters
    phase_marginalization=True,
)
```

### Automatic reference-parameter search

If you do not have reference parameters, pass a `prior` (and any `likelihood_transforms`) and the constructor will call `maximize_likelihood` internally using `evosax.CMA_ES` (Covariance Matrix Adaptation Evolution Strategy):

```python
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.prior import CombinePrior, UniformPrior, SinePrior, CosinePrior
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform

prior = CombinePrior([
    UniformPrior(10.0, 100.0, parameter_names=["M_c"]),
    UniformPrior(0.125, 1.0,  parameter_names=["q"]),
    ...
    UniformPrior(0.0, 2*jnp.pi, parameter_names=["ra"]),
    CosinePrior(parameter_names=["dec"]),
])

likelihood = HeterodynedTransientLikelihoodFD(
    detectors=[H1, L1],
    waveform=waveform,
    trigger_time=gps_time,
    f_min=20.0,
    f_max=1024.0,
    prior=prior,
    likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
    optimizer_popsize=500,
    optimizer_n_steps=1000,
)
```

The optimiser runs `evosax.CMA_ES` with a JAX-native ask/tell loop, so the waveform evaluations are fully batched and JIT-compiled on CPU/GPU.
