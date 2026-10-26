# Use Your Own Likelihood

Jim's samplers only need a function that returns a log-likelihood, so nothing here is specific to gravitational waves.
There are two ways to use them with your own likelihood:

- **With `Jim`** — when you want named parameters, [priors](../guides/prior.md) and named results.
- **Without `Jim`** — when you already have a log-density of a flat array and only want a sampler.

## With `Jim`

Subclass `LikelihoodBase` and implement `evaluate`.
It receives a dictionary of parameter values and returns the log-likelihood:

```python
import jax.numpy as jnp
from jimgw.core.base import LikelihoodBase


class SinusoidLikelihood(LikelihoodBase):
    def __init__(self, t, y, noise_std):
        self.t = jnp.asarray(t)
        self.y = jnp.asarray(y)
        self.noise_std = noise_std

    def evaluate(self, params):
        model = params["amplitude"] * jnp.sin(
            2 * jnp.pi * params["frequency"] * self.t + params["phase"]
        )
        return -0.5 * jnp.sum((self.y - model) ** 2) / self.noise_std**2
```

Give each parameter a prior with the same name, and pass both to `Jim`:

```python
from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.samplers.config import FlowMCConfig

prior = CombinePrior(
    [
        UniformPrior(0.0, 3.0, parameter_names=["amplitude"]),
        UniformPrior(3.0, 5.0, parameter_names=["frequency"]),
        UniformPrior(0.0, 6.28, parameter_names=["phase"]),
    ]
)

jim = Jim(
    SinusoidLikelihood(t, y, noise_std=0.5),
    prior,
    sampler_config=FlowMCConfig(n_chains=500, n_training_loops=10),
)
jim.sample()
samples = jim.get_samples()  # one array per parameter, plus "log_likelihood"
```

A complete script, with synthetic data and a corner plot, is [`examples/custom_likelihood.py`](https://github.com/GW-JAX-Team/Jim/blob/main/examples/custom_likelihood.py).

!!! note "Things to know"
    - **`evaluate` is compiled by JAX.** Use `jnp` functions, and `jnp.where` or `jax.lax.cond` instead of a Python `if` on a parameter value, which raises `TracerBoolConversionError`.
    - **Parameter names are not checked.** The keys you read in `evaluate` must match the prior's `parameter_names`. A misspelt key raises `KeyError` when `Jim` is built, but a prior parameter that `evaluate` never reads is accepted without warning, and its posterior is just its prior.
    - **Samplers and priors have their own constraints**, such as a normalised prior or, for NS AW, the unit cube. They are listed in the [sampler overview](../guides/samplers.md#sampler-overview); the [Transforms guide](../guides/transforms.md#mapping-a-prior-to-the-unit-cube-for-ns-aw) shows how to map a prior to the unit cube.
    - **Importing `Jim` turns on 64-bit floats** for all JAX code in the process, as a side effect of importing ripple. It matters if you rely on single precision.
    - **Not available:** the SwiG sampler, which needs a gravitational-wave likelihood with waveform caching, and the `jim-run` command, which cannot load a custom likelihood.

## Without `Jim`

`build_sampler` takes three functions of a flat array of shape `(n_dims,)`: the log-prior, the log-likelihood and their sum.
It does not import any gravitational-wave code and does not change the JAX float precision.

```python
import jax
import jax.numpy as jnp
from jimgw.samplers import build_sampler
from jimgw.samplers.config import FlowMCConfig

lower = jnp.array([0.0, 3.0, 0.0])
upper = jnp.array([3.0, 5.0, 6.28])


def log_prior(x):
    inside = jnp.all((x > lower) & (x < upper))
    return jnp.where(inside, -jnp.sum(jnp.log(upper - lower)), -jnp.inf)


def log_likelihood(x):
    amplitude, frequency, phase = x
    model = amplitude * jnp.sin(2 * jnp.pi * frequency * t + phase)
    return -0.5 * jnp.sum((y - model) ** 2) / 0.5**2


config = FlowMCConfig(n_chains=500, n_training_loops=10)
sampler = build_sampler(
    config,
    n_dims=3,
    log_prior_fn=log_prior,
    log_likelihood_fn=log_likelihood,
    log_posterior_fn=lambda x: log_prior(x) + log_likelihood(x),
)

initial_position = lower + (upper - lower) * jax.random.uniform(
    jax.random.key(1), (config.n_chains, 3)
)
sampler.sample(jax.random.key(0), initial_position)
result = sampler.get_samples()  # "samples" of shape (n, 3) and "log_likelihood"
```

You provide the bounds, by returning `-jnp.inf` from `log_prior` outside them, and the starting points.
The [Samplers guide](../guides/samplers.md) gives each sampler's config and how many starting points it needs.
