"""Post-run diagnostics and resampling for the BlackJAX SMC sampler modes.

Pure functions over a finished SMC state -- no coupling to the sampling loop.
Kept separate so each mode class's ``get_samples``/``_get_diagnostics`` stay
thin composition and the actual math has a name and a docstring of its own.
"""

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.smc.persistent_sampling import (
    compute_log_persistent_weights,
    compute_persistent_ess,
)

# Fixed key for the persistent-sampling posterior resample in get_samples().
_RESAMPLE_KEY = jax.random.key(123)


def resample_persistent_particles(ps, n_dims: int) -> dict[str, np.ndarray]:
    """Draw the posterior sample from a finished persistent-sampling (ap/fp) state.

    Resamples with replacement from all-temperature particles, weighted by the
    persistent-sampling weight formula, to a target count approximately equal to
    the effective sample size ``1 / max(weights)``.

    Args:
        ps: The algorithm's persistent-sampling state (``PersistentSamplingState``),
            i.e. ``state.sampler_state`` (mode ap) or ``state`` (mode fp).
        n_dims: Dimension of the sampling space.

    Returns:
        Dict with keys ``"samples"`` (shape ``(n, n_dims)``) and
        ``"log_likelihood"`` (shape ``(n,)``).
    """
    n_iter = int(ps.iteration)
    all_particles = ps.persistent_particles[: n_iter + 1].reshape(-1, n_dims)
    all_log_likelihoods = ps.persistent_log_likelihoods[: n_iter + 1].reshape(-1)

    log_w, _ = compute_log_persistent_weights(
        ps.persistent_log_likelihoods,
        ps.persistent_log_Z,
        ps.tempering_schedule,
        ps.iteration,
        include_current=True,
    )
    weights = jax.nn.softmax(log_w[: n_iter + 1].reshape(-1))

    n_available = all_particles.shape[0]
    n_target = max(1, int(1.0 / float(jnp.max(weights))))
    n_target = min(n_target, n_available)
    indices = jax.random.choice(
        _RESAMPLE_KEY,
        n_available,
        shape=(n_target,),
        replace=True,
        p=weights,
    )
    return {
        "samples": np.asarray(all_particles[indices]),
        "log_likelihood": np.asarray(all_log_likelihoods[indices]),
    }


def persistent_ess_history(ps, n_iterations: int) -> np.ndarray:
    """Per-iteration persistent-sampling ESS (modes ap/fp).

    Args:
        ps: The algorithm's persistent-sampling state, as in
            :func:`resample_persistent_particles`.
        n_iterations: Number of completed iterations.

    Returns:
        ESS at each iteration ``1..n_iterations``, shape ``(n_iterations,)``.
    """
    ess_hist = np.zeros(n_iterations)
    for t in range(1, n_iterations + 1):
        log_w, _ = compute_log_persistent_weights(
            ps.persistent_log_likelihoods,
            ps.persistent_log_Z,
            ps.tempering_schedule,
            t,
            include_current=True,
        )
        ess_hist[t - 1] = float(
            compute_persistent_ess(log_w.reshape(-1), normalize_weights=True)
        )
    return ess_hist


def persistent_log_z_error(ps, n_iterations: int) -> float:
    """Delta-method log-evidence standard error for persistent sampling (modes ap/fp).

    At step k, importance weights are ``exp(delta_beta * logL)`` over all k
    batches of accumulated particles; ``Var(log Z_k) = Var(w) / (N_eff * E[w]^2)``,
    summed across steps.

    Args:
        ps: The algorithm's persistent-sampling state, as in
            :func:`resample_persistent_particles`.
        n_iterations: Number of completed iterations.

    Returns:
        Standard deviation of the final cumulative ``log_Z``.
    """
    var_list = []
    for k in range(1, n_iterations + 1):
        delta_beta = float(ps.tempering_schedule[k]) - float(
            ps.tempering_schedule[k - 1]
        )
        log_L_accum = ps.persistent_log_likelihoods[:k].reshape(-1)
        log_w_k = delta_beta * log_L_accum
        m = float(jnp.max(log_w_k))
        u = jnp.exp(log_w_k - m)
        mean_u = float(jnp.mean(u))
        if mean_u > 0:
            var_list.append(float(jnp.var(u)) / (len(log_w_k) * mean_u**2))
    return float(np.sqrt(np.sum(var_list)))


def kish_ess_history(is_weights_history: np.ndarray) -> np.ndarray:
    """Per-iteration Kish effective sample size for tempered modes (at/ft).

    Importance weights are already normalized per iteration, so Kish ESS is
    ``1 / sum(w**2)``.

    Args:
        is_weights_history: Per-iteration normalized IS weights,
            shape ``(n_iterations, n_particles)``.

    Returns:
        ESS at each iteration, shape ``(n_iterations,)``.
    """
    return 1.0 / np.sum(is_weights_history**2, axis=-1)


def kish_log_z_error(is_weights_history: np.ndarray) -> float:
    """Delta-method log-evidence standard error for tempered modes (at/ft).

    ``Var(log Z_k) = sum(w**2) - 1/N`` per step for normalized weights, summed
    across steps.

    Args:
        is_weights_history: Per-iteration normalized IS weights,
            shape ``(n_iterations, n_particles)``.

    Returns:
        Standard deviation of the final cumulative ``log_Z``.
    """
    n_particles = is_weights_history.shape[1]
    var_per_step = np.sum(is_weights_history**2, axis=-1) - 1.0 / n_particles
    return float(np.sqrt(float(np.clip(np.sum(var_per_step), 0.0, None))))
