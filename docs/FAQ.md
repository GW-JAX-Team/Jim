# FAQ

## JAX

### Float precision

JAX defaults to float32, but gravitational-wave analyses require float64 precision. Without it, you may see inaccurate likelihoods or unexpected NaN values. Always enable it at the top of your script, **before** any JAX operations:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

### JIT compilation is slow

The first call to a JIT-compiled function triggers XLA compilation, which can take significant time for complex likelihoods (sometimes several minutes for a full GW inference run). This is normal — subsequent calls will be fast.

To disable JIT for debugging:

```python
jax.config.update("jax_disable_jit", True)
```

### RNG key errors

JAX uses an explicit PRNG system. Each random operation consumes a key. Always split or fold keys for separate operations — reusing the same key gives deterministic, non-random results:

```python
key = jax.random.key(0)
key1, key2 = jax.random.split(key)
```

## Sampler Tuning

### The sampler is not accepting proposals

This usually means the step size is too large. Reduce it. If the step size is already small and acceptance is still near zero, check that your likelihood is well-defined (no NaN values) within the entire prior support.

### Chains are highly correlated

A very small step size leads to slow exploration and correlated samples. Try increasing the step size first. If that does not help, your parameters likely have very different scales — a small step in a tightly constrained direction prevents larger steps elsewhere. Set per-parameter step sizes, or reparameterise so all parameters have similar numerical scale.

### How many chains should I use?

Use as many chains as your hardware allows, especially on a GPU. More chains improve exploration and help discover multi-modal posteriors. Increase the number until you see a significant performance hit or run out of memory.

### How long should I run training?

Most computation goes into the training phase. The production phase with a trained normalizing flow is usually cheap. When in doubt, increase `n_training_loops` in the `Jim` constructor — blow it up until you cannot stand waiting.

## GPU and Memory

### I am running out of GPU memory

The right knob depends on which sampler backend you use. In every case the goal is the same: shrink how much work runs simultaneously.

#### flowMC (`FlowMCConfig`)

flowMC has two independent memory bottlenecks:

1. **The NF proposal step — usually the bottleneck.** During each global step the likelihood is evaluated at `n_chains × n_global_steps` normalizing-flow proposals at once. Reduce `n_NFproposal_batch_size` (default `1000`); flowMC then evaluates the proposals in smaller `jax.lax.map` chunks instead of one big batch. **Try this first.**

2. **The local step — when the likelihood is very expensive.** Sometimes the problem is not the `n_chains × n_global_steps` proposals, but that even a *single* likelihood evaluation per chain across all `n_chains` chains will not fit. In that case also set `chain_batch_size` to a small positive integer (default `0` means all chains at once; smaller values use less memory) so chains are processed in sub-batches.

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

#### BlackJAX nested sampling (`BlackJAXNSAWConfig`, `BlackJAXNSSConfig`)

Each iteration deletes and replaces `n_live × n_delete_frac` live points in parallel. Reduce `n_delete_frac` (default `0.5`) to evaluate fewer replacements at a time.

#### BlackJAX SMC (`BlackJAXSMCConfig`)

Set `batch_size` to a small positive integer (default `0` = full `jax.vmap` over all particles; smaller values use less memory). The MCMC update then runs over particles in `jax.lax.map` batches instead of vmapping all `n_particles` at once.

## Quality Assessment

### How do I know if my run converged?

After sampling, always check:

1. **Trace plots** — Do all chains look well-mixed with no visible trends or drifts?
2. **Effective sample size (ESS)** — A low ESS relative to the number of raw samples indicates high correlation between draws.
3. **Posterior predictive checks** — Simulate data from the posterior and compare to the observed data.
