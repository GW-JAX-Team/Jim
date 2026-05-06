# Prior

Jim priors are built by composing individual prior components with `CombinePrior`, which joins them into a joint prior. Each component can cover one or more parameters.

!!! note "Sampler prior requirements"
    Some samplers impose extra constraints on the prior. BlackJAX NS-AW requires a uniform prior on the unit hypercube; BlackJAX NSS and SMC require a normalised prior. See the [Samplers guide](samplers.md) before choosing a backend.

## CombinePrior

`CombinePrior` takes a list of priors and treats them as independent:

```python
import jax.numpy as jnp
from jimgw.core.prior import CombinePrior, UniformPrior, SinePrior, CosinePrior, PowerLawPrior

prior = CombinePrior([
    UniformPrior(10.0, 80.0, ["M_c"]),
    UniformPrior(0.125, 1.0, ["q"]),
    UniformPrior(-0.99, 0.99, ["s1_z"]),
    UniformPrior(-0.99, 0.99, ["s2_z"]),
    PowerLawPrior(10.0, 2000.0, 2.0, ["d_L"]),
    UniformPrior(-0.1, 0.1, ["t_c"]),
    UniformPrior(0.0, 2.0 * jnp.pi, ["phase_c"]),
    SinePrior(["iota"]),
    UniformPrior(0.0, jnp.pi, ["psi"]),
    UniformPrior(0.0, 2.0 * jnp.pi, ["ra"]),
    CosinePrior(["dec"]),
])
```

The order of parameters in `prior.parameter_names` follows the order they appear in this list.

## Basic Priors

All priors are importable from `jimgw.core.prior`.

### UniformPrior

Flat distribution over `[xmin, xmax]`:

```python
UniformPrior(xmin, xmax, ["parameter_name"])
```

### PowerLawPrior

Power-law distribution $p(x) \propto x^\alpha$ over `[xmin, xmax]`:

```python
PowerLawPrior(xmin, xmax, alpha, ["parameter_name"])
```

!!! note
    `xmin` must be positive.

### SinePrior

$p(\theta) \propto \sin(\theta)$ over $\lbrack 0, \pi \rbrack$. Commonly used for inclination:

```python
SinePrior(["iota"])
```

### CosinePrior

$p(\delta) \propto \cos(\delta)$ over $\lbrack -\pi/2, \pi/2 \rbrack$. Commonly used for declination:

```python
CosinePrior(["dec"])
```

### UniformSpherePrior

Uniform prior on the surface of a unit sphere, parameterised by magnitude, polar angle, and azimuthal angle. Useful for spin vectors:

```python
from jimgw.core.prior import UniformSpherePrior

UniformSpherePrior(["s1"])  # creates s1_mag, s1_theta, s1_phi
```

### GaussianPrior

Gaussian distribution with given mean and standard deviation:

```python
from jimgw.core.prior import GaussianPrior

GaussianPrior(mean, std, ["parameter_name"])
```

### RayleighPrior

Rayleigh distribution with a given scale:

```python
from jimgw.core.prior import RayleighPrior

RayleighPrior(sigma, ["parameter_name"])
```

## Constraints

!!! warning
    When custom constraints are applied, the resulting prior is generally **not normalised**. flowMC tolerates this because it never needs the normalisation constant. However, BlackJAX NS-AW, NSS, and SMC compute Bayesian evidence and therefore require a normalised prior. If you know your constrained prior is normalised, override `is_normalized` to return `True`. Jim enforces this at construction time and will raise a `ValueError` if `is_normalized` is `False` for those backends.

### Single-parameter bounds with BoundedMixin

`BoundedMixin` enforces hard bounds on a single parameter: the log-probability is set to $-\infty$ for any sample outside `[xmin, xmax]`. You can use it to add bounds to your own priors by subclassing `BoundedMixin` before the base prior class:

```python
from jimgw.core.prior import BoundedMixin, GaussianPrior

class BoundedGaussianPrior(BoundedMixin, GaussianPrior):
    xmin: float
    xmax: float

    def __init__(self, mean, std, xmin, xmax, parameter_names):
        super().__init__(mean, std, parameter_names)
        self.xmin = xmin
        self.xmax = xmax
```

The `BoundedMixin` must appear **before** the base prior class in the inheritance list so that its `log_prob` override is resolved first.

### Multi-parameter constraints

For constraints that span multiple parameters, subclass `CombinePrior` and override `log_prob` to add a $0 / {-\infty}$ penalty. For example, to enforce $m_1 > m_2$:

```python
import jax.numpy as jnp
from jimgw.core.prior import CombinePrior, UniformPrior

class OrderedMassPrior(CombinePrior):
    def log_prob(self, z):
        base = super().log_prob(z)
        constraint = jnp.where(z["m1"] > z["m2"], 0.0, -jnp.inf)
        return base + constraint

prior = OrderedMassPrior([
    UniformPrior(1.0, 100.0, ["m1"]),
    UniformPrior(1.0, 100.0, ["m2"]),
])
```
