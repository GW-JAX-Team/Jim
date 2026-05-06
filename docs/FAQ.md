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

This section applies to the flowMC backend.

The most targeted fix is to set `chain_batch_size` inside `FlowMCConfig`. By default, (`chain_batch_size=0`) all chains are evaluated simultaneously; setting it to a smaller integer processes chains in batches, directly reducing peak memory at the cost of slightly slower throughput:

```python
from jimgw.samplers.config import FlowMCConfig

jim = Jim(
    likelihood,
    prior,
    sampler_config=FlowMCConfig(
        n_chains=1000,
        chain_batch_size=100,  # process 100 chains at a time instead of all at once
        # ... other parameters
    ),
    ...
)
```

If memory is still tight, reducing `n_chains` inside `FlowMCConfig` will also help.

## Quality Assessment

### How do I know if my run converged?

After sampling, always check:

1. **Trace plots** — Do all chains look well-mixed with no visible trends or drifts?
2. **Effective sample size (ESS)** — A low ESS relative to the number of raw samples indicates high correlation between draws.
3. **Posterior predictive checks** — Simulate data from the posterior and compare to the observed data.
