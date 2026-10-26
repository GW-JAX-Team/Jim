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
In that case also set `chain_batch_size` so chains are processed in sequential sub-batches of that size; smaller values use less memory.
Sub-batching only happens when `chain_batch_size` is greater than 1 and less than `n_chains`; otherwise (including the default `0`) all chains are processed in one `vmap`.

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
