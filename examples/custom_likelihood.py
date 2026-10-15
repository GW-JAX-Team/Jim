"""Sample a user-defined likelihood: fit a sinusoid to noisy data.

Nothing here is gravitational-wave specific. The likelihood is any class that
subclasses `LikelihoodBase` and implements `evaluate`.
"""

import time
from pathlib import Path

import corner
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jimgw.core.base import LikelihoodBase
from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.samplers.config import FlowMCConfig

# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------

truth = {"amplitude": 1.5, "frequency": 4.0, "phase": 0.7}
noise_std = 0.5

t = np.linspace(0.0, 1.0, 200)
rng = np.random.default_rng(0)
y = truth["amplitude"] * np.sin(
    2 * np.pi * truth["frequency"] * t + truth["phase"]
) + rng.normal(0.0, noise_std, t.size)

# ----------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------


class SinusoidLikelihood(LikelihoodBase):
    """Gaussian likelihood of a sinusoid fitted to data with known noise."""

    def __init__(self, t, y, noise_std):
        self.t = jnp.asarray(t)
        self.y = jnp.asarray(y)
        self.noise_std = noise_std

    def evaluate(self, params):
        model = params["amplitude"] * jnp.sin(
            2 * jnp.pi * params["frequency"] * self.t + params["phase"]
        )
        return -0.5 * jnp.sum((self.y - model) ** 2) / self.noise_std**2


likelihood = SinusoidLikelihood(t, y, noise_std)

# ----------------------------------------------------------------------
# Prior
# ----------------------------------------------------------------------

prior = CombinePrior(
    [
        UniformPrior(0.0, 3.0, parameter_names=["amplitude"]),
        UniformPrior(3.0, 5.0, parameter_names=["frequency"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase"]),
    ]
)

# ----------------------------------------------------------------------
# Sample
# ----------------------------------------------------------------------

jim = Jim(
    likelihood,
    prior,
    sampler_config=FlowMCConfig(
        n_chains=500,
        n_local_steps=50,
        n_global_steps=100,
        n_training_loops=10,
        n_production_loops=5,
        global_thinning=10,
    ),
)

start_time = time.time()
jim.sample()
print(f"Sampling time: {time.time() - start_time:.1f} seconds")

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

chains = jim.get_samples()

for name, value in truth.items():
    print(
        f"{name:>10}: {chains[name].mean():.2f} +/- {chains[name].std():.2f}"
        f"  (truth {value})"
    )

fig = corner.corner(
    np.stack([chains[name] for name in truth]).T,
    labels=list(truth),
    truths=list(truth.values()),
)
fig.savefig(Path(__file__).parent / "custom_likelihood.png")
