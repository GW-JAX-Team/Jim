# CLI Config Reference

`jim-run` is driven entirely by a single TOML file. This page documents every section, field, and default value.

## CLI flags

```bash
jim-run [CONFIG] [OPTIONS]
```

| Flag | Description |
| --- | --- |
| `CONFIG` | Path to the TOML config file |
| `--init PATH` | Write a GW150914-style template config to `PATH` and exit |
| `--verbose`, `-v` | Enable DEBUG-level logging |
| `--help` | Show usage and exit |

---

## Top-level fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `seed` | int | `0` | JAX random seed |

---

## `[data]`

Selects where strain and PSD data come from. The `type` field is required and determines which other fields are valid.

### `type = "gwosc"` — fetch from GWOSC

Fetches public LIGO/Virgo/KAGRA strain and PSD from the Gravitational-Wave Open Science Center.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `detectors` | list[str] | — | Detector identifiers, e.g. `["H1", "L1", "V1"]` |
| `trigger_time` | float | — | GPS trigger time (seconds) |
| `duration` | float | — | Analysis segment length (seconds) |
| `post_trigger_duration` | float | `2.0` | Seconds after the trigger kept in the window |
| `psd_duration` | float | — | Off-source segment length for PSD estimation (seconds) |

### `type = "injection"` — synthetic signal in design noise

Injects a waveform into simulated design-sensitivity Gaussian noise.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `detectors` | list[str] | — | Detector identifiers |
| `trigger_time` | float | — | GPS trigger time |
| `duration` | float | — | Segment length (seconds) |
| `sampling_frequency` | float | — | Sample rate in Hz (e.g. `2048.0`) |
| `injection_parameters` | dict[str, float] | — | Signal parameters to inject |
| `zero_noise` | bool | `false` | If `true`, inject into zero noise (noiseless matched-filter test) |

Example:

```toml
[data]
type = "injection"
detectors = ["H1", "L1"]
trigger_time = 1126259462.4
duration = 4.0
sampling_frequency = 2048.0

[data.injection_parameters]
M_c     = 28.3
q       = 0.85
s1_z    = 0.0
s2_z    = 0.0
iota    = 0.4
d_L     = 440.0
t_c     = 0.0
phase_c = 0.0
psi     = 0.0
ra      = 1.375
dec     = -1.21
```

### `type = "file"` — load from local files

Loads strain and PSD from local files. Useful for offline or CI use.

Supported strain formats: `.npz`, `.gwf` / `.gwf.gz`, `.hdf5` / `.h5`, `.csv`.
Supported PSD formats: `.npz`, `.txt`, `.dat`, `.csv`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `detectors` | list[str] | — | Detector identifiers |
| `trigger_time` | float | — | GPS trigger time |
| `strain_files` | dict[str, path] | — | Map from detector name to strain file (any supported format) |
| `psd_files` | dict[str, path] | — | Map from detector name to PSD file (any supported format) |
| `strain_channels` | dict[str, str] | `{}` | Channel name per detector for GWF/HDF5 files (e.g. `"H1:GDS-CALIB_STRAIN"`). If omitted, common LIGO/Virgo preset channel names are tried for `.gwf` files. |
| `psd_is_asd` | dict[str, bool] | `{}` | Set to `true` for detectors whose PSD file contains ASD values (Hz⁻¹/²); they are squared automatically. Ignored for `.npz` files. |

Example with `.npz` files (no channels required):

```toml
[data]
type = "file"
detectors = ["H1", "L1"]
trigger_time = 1126259462.4

[data.strain_files]
H1 = "h1_strain.npz"
L1 = "l1_strain.npz"

[data.psd_files]
H1 = "h1_psd.npz"
L1 = "l1_psd.npz"
```

Example with `.gwf` frame files and `.txt` ASD curves:

```toml
[data]
type = "file"
detectors = ["H1", "L1"]
trigger_time = 1126259462.4

[data.strain_files]
H1 = "H-H1_LOSC_4_V2-1126259446-32.gwf"
L1 = "L-L1_LOSC_4_V2-1126259446-32.gwf"

[data.strain_channels]
H1 = "H1:GWOSC-4KHZ_R1_STRAIN"
L1 = "L1:GWOSC-4KHZ_R1_STRAIN"

[data.psd_files]
H1 = "O3-H1_asd.txt"
L1 = "O3-L1_asd.txt"

[data.psd_is_asd]
H1 = true
L1 = true
```

---

## `[waveform]`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `approximant` | str | — | Waveform model name |
| `f_ref` | float | `20.0` | Reference frequency in Hz for spin conventions |

---

## `[prior]`

Each entry maps a parameter name to a prior specification. The parameter name becomes the column name in the output samples.

```toml
[prior]
M_c = { type = "uniform", min = 10.0, max = 80.0 }
iota = { type = "sine" }
```

### Prior types

#### `uniform`

```toml
param = { type = "uniform", min = <float>, max = <float> }
```

Uniform distribution on [`min`, `max`].

#### `gaussian`

```toml
param = { type = "gaussian", loc = <float>, scale = <float> }
```

Normal distribution with mean `loc` and standard deviation `scale`.

#### `sine`

```toml
param = { type = "sine" }
```

Sine-weighted prior on $[0, \pi]$. Standard choice for inclination `iota`.

#### `cosine`

```toml
param = { type = "cosine" }
```

Cosine-weighted prior on $[-\pi/2, \pi/2]$. Standard choice for declination `dec`.

#### `power_law`

```toml
param = { type = "power_law", min = <float>, max = <float>, alpha = <float> }
```

Power-law $p(x) \propto x^\alpha$ on [`min`, `max`]. Use `alpha = 2` for a prior uniform in volume on luminosity distance.

#### `rayleigh`

```toml
param = { type = "rayleigh", scale = <float> }
```

Rayleigh distribution with scale parameter σ.

#### `uniform_sphere`

```toml
spin1 = { type = "uniform_sphere" }
```

Uniform distribution on the unit sphere. Generates **three** output parameters: `{name}_mag`, `{name}_theta`, `{name}_phi`. Use for 3-D spin vectors.

---

## `[sampling]`

Optional section that controls the coordinate system the sampler explores. The CLI auto-infers transforms for all other cases.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `time_frame` | str | `"detector"` | Controls the sampling parameter space of signal arrival times. `"detector"` (default): sample detector time rather than geocentric. Set to an explicit detector name (e.g. `"H1"`) to use a specific detector. `"geocentric"`: sample geocentric time directly. |
| `sky_frame` | str | `"detector"` | Controls the sampling parameter space of sky position. `"detector"` (default): sample `azimuth`/`zenith` relative to the detector baseline (requires ≥ 2 detectors). `"geocentric"`: sample `ra`/`dec` directly. |

---

## `[likelihood]`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `f_min` | float | — | Lower frequency cutoff in Hz |
| `f_max` | float | — | Upper frequency cutoff in Hz |
| `fixed_parameters` | dict[str, float] | `{}` | Parameters held fixed at these values and excluded from sampling |
| `phase_marginalization` | bool | `false` | Analytically marginalise over coalescence phase |
| `time_marginalization` | table | — | Optional — Analytically marginalise over coalescence time (see below) |
| `distance_marginalization` | table | — | Optional — Analytically marginalise over luminosity distance (see below) |
| `heterodyne` | table | — | Optional — Use the relative-binning likelihood (see below) |
| `multiband` | table | — | Optional — Use the multi-banded likelihood (see below) |

### `[likelihood.time_marginalization]`

```toml
[likelihood.time_marginalization]
tc_range = [-0.1, 0.1]  # seconds relative to trigger_time
```

### `[likelihood.distance_marginalization]`

```toml
[likelihood.distance_marginalization]
n_dist_points = 10000  # integration grid size
# ref_dist = 440.0     # optional: reference distance in Mpc; omit to auto-select

[likelihood.distance_marginalization.distance_prior]
d_L = { type = "power_law", min = 1.0, max = 2000.0, alpha = 2.0 }
```

### `[likelihood.heterodyne]`

Enables `HeterodynedTransientLikelihoodFD` (relative binning).

```toml
[likelihood.heterodyne]
n_bins = 1000

[likelihood.heterodyne.reference_parameters]
type = "optimizer"   # "optimizer" | "provided" | "injection"
popsize = 500
n_steps = 1000
```

| `reference_parameters.type` | Description |
| --- | --- |
| `"optimizer"` | Find reference parameters via CMA-ES optimisation (default) |
| `"provided"` | Supply explicit likelihood-space values via a `values` dict |
| `"injection"` | Use `data.injection_parameters` as reference (injection runs only) |

### `[likelihood.multiband]`

Enables `MultibandedTransientLikelihoodFD` (geometric frequency banding).

All fields are optional; when omitted, defaults are chosen adaptively based on the data and prior.

---

## `[sampler]`

The `type` field selects the backend. Each backend has its own set of tuning parameters.

### Sampler comparison

| `type` | Algorithm | Evidence | Extra install |
| --- | --- | --- | --- |
| `flowmc` | Normalizing-flow MCMC | No | No |
| `blackjax-smc` | Sequential Monte Carlo | Yes | No |
| `blackjax-nss` | Nested slice sampling | Yes | `uv sync --group nested-sampling` |
| `blackjax-ns-aw` | Nested sampling (acceptance-walk) | Yes | `uv sync --group nested-sampling` |

### `type = "flowmc"`

| Field | Default | Description |
| --- | --- | --- |
| `n_chains` | `1000` | Number of parallel chains |
| `n_local_steps` | `100` | Local MCMC steps per chain per global step |
| `n_global_steps` | `1000` | Global (normalizing-flow) steps per training loop |
| `n_training_loops` | `20` | Number of training loops |
| `n_production_loops` | `10` | Number of production (no-training) loops |
| `n_epochs` | `20` | Normalizing-flow training epochs per loop |
| `local_kernel` | `"MALA"` | Local kernel: `"MALA"`, `"HMC"`, or `"GRW"` |
| `n_NFproposal_batch_size` | `1000` | Flow proposal batch size |
| `global_thinning` | `1` | Keep every Nth global step in the production chain |
| `local_thinning` | `1` | Keep every Nth local step |
| `early_stopping` | `true` | Stop training when the loss plateaus |
| `verbose` | `false` | Print sampler-level progress |
| `parallel_tempering` | disabled | Set to `true` to enable with defaults, or provide a dict for custom settings |

Parallel-tempering sub-table:

```toml
[sampler.parallel_tempering]
n_temperatures = 5
max_temperature = 10.0
n_tempered_steps = 5
```

### `type = "blackjax-smc"`

| Field | Default | Description |
| --- | --- | --- |
| `n_particles` | `2000` | Number of particles |
| `n_mcmc_steps_per_dim` | `100` | MCMC steps per dimension per temperature |
| `target_ess_fraction` | `0.9` | Optional — target effective sample size as a fraction of `n_particles`; use instead of `target_ess`, not both |
| `target_ess` | — | Optional — target absolute ESS count; use instead of `target_ess_fraction`, not both |
| `initial_cov_scale` | `0.5` | Initial covariance scale factor |
| `target_acceptance_rate` | `0.234` | Target MCMC acceptance rate |
| `persistent_sampling` | `true` | Reuse particles across temperatures |
| `temperature_ladder` | — | Optional — fixed list of temperatures from 0.0 to 1.0; omit to use adaptive tempering |

### `type = "blackjax-nss"`

| Field | Default | Description |
| --- | --- | --- |
| `n_live` | `1000` | Number of live points |
| `n_delete_frac` | `0.5` | Fraction of live points replaced per iteration |
| `num_inner_steps_per_dim` | `10` | Slice steps per dimension for the nested kernel |
| `termination_dlogz` | `0.1` | Stop when the remaining log-evidence contribution falls below this |

### `type = "blackjax-ns-aw"`

Requires all sampling-space parameters to lie in $[0, 1]$. The CLI enforces this automatically.

| Field | Default | Description |
| --- | --- | --- |
| `n_live` | `1000` | Number of live points |
| `n_delete_frac` | `0.5` | Fraction of live points replaced per iteration |
| `n_target` | `60` | Target number of accepted steps per replacement |
| `max_mcmc` | `5000` | Maximum MCMC steps per replacement attempt |
| `max_proposals` | `1000` | Maximum proposals per MCMC step |
| `termination_dlogz` | `0.1` | Termination criterion on remaining log-evidence |

---

## `[output]`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `dir` | path | — | Directory to write all outputs into |
| `overwrite` | bool | `false` | Allow overwriting an existing output directory |
| `save_corner` | bool | `false` | Save a corner plot (`corner.png`); requires the `corner` package |
| `n_samples` | int | `0` | Number of posterior samples to save; `0` means save all |
| `corner_parameters` | list[str] | — | Optional — parameter names to include in the corner plot; default is all parameters |

### Output files

| File | Description |
| --- | --- |
| `samples.npz` | Posterior samples; keys `samples` (shape `n_samples × n_params`) and `parameter_names` |
| `diagnostics.json` | Scalar diagnostics: log evidence (nested samplers), acceptance rates, etc. |
| `diagnostics.npz` | Array diagnostics: log-probability traces, chain arrays, etc. |
| `config.final.toml` | The fully-resolved config including all defaults; re-run with `jim-run config.final.toml` |
| `corner.png` | Corner plot (only when `save_corner = true`) |
