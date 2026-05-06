# Quick Start

This page gives you a bird's-eye view of Jim's main components and how they fit together. For deeper dives into each piece, see the [Guides](guides/index.md).

## Overview

A Jim analysis is assembled from the following building blocks:

```mermaid
flowchart LR
    WM[Waveform Model] & D[Data] --> L[Likelihood]
    L --> J((Jim))
    L --- LT[Likelihood Transforms]
    P[Prior] --- LT
    LT --> J
    P --> J
    P --- ST[Sample Transforms]
    ST --> J
    J <--> S[Sampler]

    click WM "#waveform-model"
    click D "#data"
    click L "#likelihood"
    click LT "#transforms"
    click P "#prior"
    click ST "#transforms"
    click S "#sampler"
    click J "#putting-it-together"
```

### Data

Detector data lives in `Detector` objects. You can fetch public LIGO/Virgo strain from GWOSC, load local files, or inject a simulated signal:

```python
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.data import Data

H1 = get_H1()
L1 = get_L1()

# Option 1: fetch from GWOSC
data = Data.from_gwosc("H1", gps_start, gps_end)
H1.set_data(data)

# Option 2: load from a .npz file
data = Data.from_file("path/to/data.npz")
H1.set_data(data)
```

### Waveform Model

Jim uses [ripple](https://github.com/GW-JAX-Team/ripple) waveform models, which are JAX-native and fully differentiable:

```python
from jimgw.core.single_event.waveform import RippleIMRPhenomD

waveform = RippleIMRPhenomD(f_ref=20.0)
```

### Likelihood

The likelihood connects detector data with a waveform model. `TransientLikelihoodFD` is the standard frequency-domain likelihood for compact binary signals:

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

### Prior

Priors are built by combining components with `CombinePrior`:

```python
from jimgw.core.prior import CombinePrior, UniformPrior, SinePrior, CosinePrior

prior = CombinePrior([
    UniformPrior(10.0, 80.0, ["M_c"]),
    UniformPrior(0.125, 1.0, ["q"]),
    SinePrior(["iota"]),
    CosinePrior(["dec"]),
    # ... add more parameters
])
```

### Transforms

Jim uses two kinds of transforms to bridge three parameter spaces:

```mermaid
flowchart LR
    PS[Prior Space] --> LT[Likelihood Transforms] --> LS[Likelihood Space]
    PS --> ST[Sample Transforms] --> SS[Sampling Space]
```

- **Likelihood transforms** — map from the prior parameter space to the likelihood parameter space. The likelihood space is fixed by your waveform model (e.g. ripple expects `eta`, Cartesian spins), so the likelihood transforms you need depend on how you define your prior. For example, if your prior is on mass ratio `q` but the waveform expects symmetric mass ratio `eta`, a likelihood transform handles that conversion.

- **Sample transforms** — map from the prior space to the sampling space. This lets the sampler explore a different parameterisation than the one your prior is defined in, typically one where correlations between parameters are reduced (e.g. sampling in detector-frame sky coordinates instead of equatorial coordinates).

### Sampler

Jim's sampler is selected by passing a typed config object.  Four backends are available:

| Backend | Config class | Evidence | Extra install |
| --- | --- | --- | --- |
| **flowMC** | `FlowMCConfig` | No | No |
| **BlackJAX NS-AW** | `BlackJAXNSAWConfig` | Yes | Yes — `uv sync --group nested-sampling` |
| **BlackJAX NSS** | `BlackJAXNSSConfig` | Yes | Yes — `uv sync --group nested-sampling` |
| **BlackJAX SMC** | `BlackJAXSMCConfig` | Yes | No |

flowMC is a normalizing-flow-enhanced MCMC sampler.
BlackJAX NS-AW and NSS are nested samplers with different sampling algorithms.
BlackJAX SMC uses a particle population tempered from prior to posterior.

See the [Samplers guide](guides/samplers.md) for configuration details and per-backend requirements.

### Putting It Together

```python
from jimgw.core.jim import Jim
from jimgw.samplers.config import FlowMCConfig

jim = Jim(
    likelihood=likelihood,
    prior=prior,
    sampler_config=FlowMCConfig(
        n_chains=500,
        n_training_loops=20,
        n_production_loops=10,
        # See the Samplers guide (guides/samplers) for the full parameter reference.
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)

jim.sample()
samples = jim.get_samples()
```

For a full worked example, see the [Getting Started tutorial](tutorials/getting_started). For production-grade scripts, browse the [`examples/` directory on GitHub](https://github.com/GW-JAX-Team/jim/tree/main/examples).
