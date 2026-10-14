# Reduce GPU Memory Use

If a run fails because the GPU is out of memory, shrink how much work runs simultaneously.
The right setting depends on which sampler backend you use.

## flowMC (`FlowMCConfig`)

flowMC has two independent memory bottlenecks:

1. **The NF proposal step — usually the bottleneck.**
During each global step, normalizing-flow proposals are generated for all chains at once.
Reduce `n_NFproposal_batch_size` (default `1000`); flowMC then evaluates the proposals in smaller `jax.lax.map` chunks instead of one big batch.
**Try this first.**

2. **The local step — when the likelihood is very expensive.**
Sometimes the problem is not the NF proposals, but that even a *single* likelihood evaluation per chain across all `n_chains` chains will not fit.
In that case also set `chain_batch_size` to a small positive integer (default `0` means all chains at once; smaller values use less memory) so chains are processed in sub-batches.

```python
from jimgw.samplers.config import FlowMCConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=FlowMCConfig(
        n_chains=1000,
        n_NFproposal_batch_size=100,  # smaller NF-proposal batches (try this first)
        chain_batch_size=100,         # also batch chains if the likelihood alone is too big
        # ... other parameters
    ),
    ...
)
```
