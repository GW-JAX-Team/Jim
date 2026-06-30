# Samplers

Jim supports several JAX sampler backends behind a unified interface.
You select one by passing a typed config object to `Jim`.

After `jim.sample()`, retrieve posterior samples with:

```python
samples = jim.get_samples()  # dict[str, np.ndarray] keyed by parameter name
```

## Sampler overview

| Sampler | Algorithm | Evidence | Extra install | Prior constraint |
| --- | --- | --- | --- | --- |
| [flowMC](#flowmc) | normalizing-flow-enhanced MCMC | No | No | None |
| [NS-AW](#blackjax-ns-aw) | Nested sampling (bilby/dynesty-style acceptance-walk) | Yes | Yes (`nested-sampling`) | Uniform prior; unit-cube sampling space |
| [NSS](#blackjax-nss) | Nested slice sampling | Yes | Yes (`nested-sampling`) | Normalised prior |
| [SMC](#blackjax-smc) | Sequential Monte Carlo | Yes | No | Normalised prior |

---

## flowMC

flowMC runs parallel MCMC chains enhanced by a normalizing flow that learns the posterior shape during training, then uses that learned geometry to make global proposals during production.

```python
from jimgw.core.jim import Jim
from jimgw.samplers.config import FlowMCConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=FlowMCConfig(
        n_chains=1000,
        n_local_steps=100,
        n_global_steps=1000,
        n_training_loops=20,
        n_production_loops=10,
    ),
)

jim.sample()
samples = jim.get_samples()
```

Key parameters:

- `n_chains` — number of parallel MCMC chains.
- `n_training_loops` / `n_production_loops` — how many rounds of training (flow updates) and production (sample collection) to run.
- `n_local_steps` / `n_global_steps` — local MCMC steps and flow-proposal steps per loop.
- `local_kernel` — MCMC kernel for local steps; one of `"MALA"` (default), `"HMC"`, or `"GRW"`.
- `parallel_tempering` — parallel tempering settings; disabled by default. Enable with `parallel_tempering=True` (uses defaults), a plain dict of settings such as `{"n_temperatures": 8}`, or a `ParallelTemperingConfig` instance.

**Repository:** [GW-JAX-Team/flowMC](https://github.com/GW-JAX-Team/flowMC)

**References:** Wong, K. W. K., Gabrié, M., Foreman-Mackey, D., *"flowMC: Normalizing flow enhanced sampling package for probabilistic inference in JAX"*, [arXiv:2211.06397](https://arxiv.org/abs/2211.06397), JOSS 8 (83) 5021 (2023). Wong, K. W. K., Isi, M., Edwards, T. D. P., *"Fast Gravitational-wave Parameter Estimation without Compromises"*, [arXiv:2302.05333](https://arxiv.org/abs/2302.05333), ApJ 958 129 (2023).

---

## BlackJAX SMC

Sequential Monte Carlo (SMC) maintains a population of particles and gradually shifts them from the prior toward the posterior through a sequence of intermediate temperature steps.

> **Normalised-prior requirement** — SMC computes a Bayesian evidence estimate and therefore requires a normalised prior. All built-in Jim priors are normalised. If you add custom constraints, check whether the resulting distribution is still normalised; if so, override `is_normalized` to return `True`. Jim raises a `ValueError` at construction if this condition is not met.

```python
from jimgw.samplers.config import BlackJAXSMCConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXSMCConfig(
        n_particles=2000,
        n_mcmc_steps_per_dim=100,
    ),
)
jim.sample()
samples = jim.get_samples()
```

Key parameters:

- `n_particles` — particle ensemble size.
- `n_mcmc_steps_per_dim` — MCMC steps per dimension at each temperature step.
- `target_ess_fraction` — target ESS as a fraction of `n_particles` (default `0.9`). The algorithm advances the temperature when the fraction of effectively contributing particles hits this threshold.  Values in `(0, 1]` are valid when `persistent_sampling=False`; persistent sampling may exceed `1.0` because particles are recycled across steps.  Only used with adaptive temperature selection (no effect with a fixed `temperature_ladder`).
- `target_ess` — target ESS as an absolute particle count. `target_ess_fraction` and `target_ess` are mutually exclusive; set one or the other, not both. When `persistent_sampling=False`, must be `<= n_particles`.
- `persistent_sampling` — whether to retain particles from all temperature steps (default `True`).
- `temperature_ladder` — explicit temperature schedule. If given, the sampler advances through this fixed ladder and ignores `target_ess_fraction` and `target_ess`.

**Repository:** [blackjax-devs/blackjax](https://github.com/blackjax-devs/blackjax)

---

## BlackJAX nested samplers

The two BlackJAX nested-sampling backends require additional dependencies.
The required modules are merged into upstream BlackJAX but not yet released on PyPI; install from main with:

```bash
uv sync --group nested-sampling
```

This pulls in:

- **blackjax** — pinned to `blackjax-devs/blackjax` until a PyPI release ships the nested-sampling modules.

---

### BlackJAX NS-AW

Nested sampling with a bilby/dynesty-style adaptive differential-evolution acceptance-walk inner kernel.

> **Unit-cube requirement** — this sampler works in the unit hypercube `[0, 1]^n_dims`.  All parameters must be mapped into `[0, 1]` via sample transforms, which the CLI constructs automatically.

```python
from jimgw.samplers.config import BlackJAXNSAWConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXNSAWConfig(
        n_live=1000,
        n_delete_frac=0.5,
        n_target=60,
        max_mcmc=5000,
        max_proposals=1000,
        termination_dlogz=0.1,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)
jim.sample()
samples = jim.get_samples()
```

Key parameters:

- `n_live` — number of live points; more live points → more accurate but slower.
- `n_delete_frac` — fraction of live points replaced per iteration (e.g. `0.5` replaces half the live points each step).
- `n_target` — target number of accepted proposals per walk.
- `max_mcmc` — maximum number of proposals before giving up on a dead point.

**Reference:** Prathaban, M., Yallup, D., Alvey, J., Yang, M., Templeton, W., Handley, W., *"Gravitational-wave inference at GPU speed: A bilby-like nested sampling kernel within blackjax-ns"*, arXiv:2509.04336 (Sep 2025).

---

### BlackJAX NSS

Nested sampling with a slice-sampling inner kernel.
Unlike NS-AW, it does not require a unit-cube prior and works in any bounded sampling space.

> **Normalised-prior requirement** — NSS computes a Bayesian evidence estimate and therefore requires a normalised prior. All built-in Jim priors are normalised. If you add custom constraints, check whether the resulting distribution is still normalised; if so, override `is_normalized` to return `True`. Jim raises a `ValueError` at construction if this condition is not met.

```python
from jimgw.samplers.config import BlackJAXNSSConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXNSSConfig(
        n_live=1000,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=20,
        termination_dlogz=0.1,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)
jim.sample()
samples = jim.get_samples()
```

Key parameters:

- `n_live` — number of live points.
- `n_delete_frac` — fraction of live points replaced per iteration.
- `num_inner_steps_per_dim` — slice-sampler steps per dimension per dead point; increase for strongly correlated posteriors.

**Repository:** [blackjax-devs/blackjax](https://github.com/blackjax-devs/blackjax)

**References:** Yallup, D., Prathaban, M., Alvey, J., Handley, W., *"Parallel Nested Slice Sampling for Gravitational Wave Parameter Estimation"*, [arXiv:2509.24949](https://arxiv.org/abs/2509.24949) (Sep 2025). Yallup, D., Kroupa, N., Handley, W., *"Nested Slice Sampling"*, [OpenReview](https://openreview.net/forum?id=ekbkMSuPo4) (2025).

---

## Checkpointing and resuming

All samplers support checkpoint/resume so long-running jobs can survive interruptions.
Set `checkpoint_dir` to a directory and `checkpoint_interval` to the minimum wall-clock seconds between writes:

```python
from jimgw.samplers.config import BlackJAXSMCConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXSMCConfig(
        n_particles=2000,
        checkpoint_dir="./my_run",
        checkpoint_interval=600,  # write at most every 10 minutes
    ),
)
jim.sample()
```

The checkpoint is written atomically (`checkpoint.pkl.tmp` → `checkpoint.pkl`) so a mid-write crash never leaves a corrupt file.
To resume after an interruption, construct the same config pointing at the same `checkpoint_dir` and call `jim.sample()` again — the sampler detects the existing file and picks up from the last saved state:

```python
# resume — identical config, same checkpoint_dir
jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXSMCConfig(
        n_particles=2000,
        checkpoint_dir="./my_run",
        checkpoint_interval=600,
    ),
)
jim.sample()  # resumes from ./my_run/checkpoint.pkl
```

The same fields work identically for `FlowMCConfig`, `BlackJAXNSAWConfig`, and `BlackJAXNSSConfig`.

| Field | Default | Notes |
| --- | --- | --- |
| `checkpoint_dir` | `None` (disabled) | Directory where `checkpoint.pkl` is written. Created automatically if absent. |
| `checkpoint_interval` | `0.0` (disabled) | Minimum seconds between writes. `0` disables checkpointing entirely. |

> **Validation** — setting `checkpoint_interval > 0` without `checkpoint_dir` raises a `ValidationError` at config construction time.

When using the [CLI](cli.md), checkpointing is enabled automatically (600 s, writing to `output.dir`).
Set `checkpoint_interval = 0` in the `[sampler]` block to opt out.

---

## Periodic parameters

All samplers accept a `periodic` field to handle parameters that wrap around (e.g. angles).
Pass a dict of `parameter_name: (lower, upper)` bounds:

```python
config = FlowMCConfig(
    ...,
    periodic={"phase_c": (0.0, 6.283185), "psi": (0.0, 3.141593)},
)
```

BlackJAX NS-AW operates in `[0, 1]` per dimension, so its `periodic` field takes a plain list of parameter names:

```python
config = BlackJAXNSAWConfig(
    ...,
    periodic=["phase_c", "psi"],
)
```

---

## Posterior samples

`jim.get_samples()` is the primary way to retrieve posterior samples.

```python
samples = jim.get_samples()
# keys: prior parameter names + "log_likelihood"
samples["M_c"]             # np.ndarray — chirp mass in prior space
samples["log_likelihood"]  # np.ndarray — per-sample log-likelihood
```

Each backend's `get_samples()` returns equally-weighted posterior samples:

- **NS-AW / NSS**: uses anesthetic's `posterior_points()` to resample the dead-point collection to equal-weight samples.
- **SMC (persistent)**: resamples all-temperature particles weighted by the persistent-sampling weights.
- **SMC (non-persistent)**: returns all final-temperature particles.
- **flowMC**: returns all production samples across all chains.

Pass `n_samples` to `jim.get_samples()` to further downsample uniformly without replacement:

```python
samples = jim.get_samples(n_samples=2000)
```

---

## Run diagnostics

`jim.get_diagnostics()` is a thin wrapper around the sampler's own `get_diagnostics()`, which returns a plain `dict[str, Any]`.

```python
diag = jim.get_diagnostics()

diag["n_likelihood_evaluations"]  # int   — total number of likelihood calls
diag["sampling_time"]             # float — wall-clock sampling time in seconds
```

Backend-specific keys:

```python
# flowMC
diag["n_training_loops_actual"]         # int        — training loops run (may be less than configured)
diag["training_loss_history"]           # np.ndarray — normalizing-flow loss per epoch
diag["acceptance_training_local"]       # np.ndarray — local acceptance rate per training loop
diag["acceptance_training_global"]      # np.ndarray — global acceptance rate per training loop
diag["acceptance_production_local"]     # np.ndarray — local acceptance rate per production loop
diag["acceptance_production_global"]    # np.ndarray — global acceptance rate per production loop

# NS-AW and NSS — also include evidence estimate
diag["n_iterations"]              # int   — number of nested-sampling steps
diag["log_Z"]                     # float — log Bayesian evidence
diag["log_Z_error"]               # float — standard deviation from 100 bootstrap samples

# SMC
diag["acceptance_history"]        # np.ndarray — mean acceptance rate at each step
diag["ess_history"]               # np.ndarray — ESS at each step
# Adaptive temperature only:
diag["n_iterations"]              # int        — number of temperature steps
diag["tempering_schedule"]        # np.ndarray — inverse temperature at each step
# Persistent sampling only:
diag["persistent_log_Z"]          # np.ndarray — cumulative log Z after each step
diag["log_Z"]                     # float      — final log Bayesian evidence
```

---

## Writing your own sampler

> This section is for advanced users who want to integrate a custom sampling backend with Jim.  It requires familiarity with JAX and the Jim sampler internals.

Subclass `Sampler`, implement three methods, and register it:

- `_sample(rng_key, initial_position)` — run the sampler and store results. The base class wraps this in `sample()`, which also records `sampling_time`.
- `get_samples()` — return a dict with `"samples"` and `"log_likelihood"` keys.
- `_get_diagnostics()` — return a plain dict with diagnostic information. The base class wraps this in `get_diagnostics()`, which injects `sampling_time`.

```python
from typing import Any, Literal, Optional
import numpy as np
from jimgw.samplers import register_sampler
from jimgw.samplers.base import Sampler
from jimgw.samplers.config import BaseSamplerConfig


class MyConfig(BaseSamplerConfig):
    type: Literal["my-sampler"] = "my-sampler"
    n_steps: int = 1000


class MySampler(Sampler):
    _config: MyConfig

    def __init__(self, *, n_dims, log_prior_fn, log_likelihood_fn,
                 log_posterior_fn, config: Optional[MyConfig] = None,
                 parameter_names=(), periodic=None):
        if config is None:
            config = MyConfig()
        super().__init__(n_dims=n_dims, log_prior_fn=log_prior_fn,
                         log_likelihood_fn=log_likelihood_fn,
                         log_posterior_fn=log_posterior_fn, config=config)
        self._result = None

    def _sample(self, rng_key, initial_position) -> None:
        # initial_position: shape (n_chains, n_dims), drawn from the prior by Jim.
        # ... run your sampler for self._config.n_steps steps ...
        self._result = np.asarray(initial_position)

    def get_samples(self) -> dict[str, np.ndarray]:
        if self._result is None:
            raise RuntimeError("call sample() first")
        return {
            "samples": self._result,
            "log_likelihood": np.zeros(self._result.shape[0]),
        }

    def _get_diagnostics(self) -> dict[str, Any]:
        if self._result is None:
            raise RuntimeError("call sample() first")
        return {
            "n_likelihood_evaluations": self._config.n_steps,
        }


register_sampler("my-sampler", lambda: MySampler)
```

Then pass `MyConfig(n_steps=500)` as `sampler_config` to `Jim`.
