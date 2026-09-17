"""Normalizing-flow preconditioning for the BlackJAX SMC sampler.

Trains a flowMC :class:`~flowMC.resource.model.nf_model.rqSpline.MaskedCouplingRQSpline`
on the current particle population and runs the Gaussian random-walk mutation kernel
in the flow's latent space instead of the raw sampling space, where the target is
close to a standard Gaussian and random-walk Metropolis mixes far more efficiently.
This mirrors pocoMC's normalizing-flow preconditioning (Karamanis et al. 2022,
MNRAS 516(2), 1644-1653), reusing flowMC's existing flow-training code as the
JAX-native normalizing-flow backend.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from blackjax.mcmc import random_walk
from blackjax.smc.resampling import systematic
from flowMC.resource.model.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, Key, PyTree

from jimgw.samplers.blackjax.smc.typing import PreconditionedMCMCStep
from jimgw.samplers.config import PreconditionConfig

# Floor for to_latent/from_latent's whitening scale -- without it, a training
# population whose variance collapses in some dimension (e.g. after heavy
# resampling leaves few distinct ancestors) divides by (near-)zero and
# produces NaN.
_WHITENING_SCALE_FLOOR = 1e-6

# Diagonal jitter added to weighted_covariance's output, sized for the flow's
# latent space (whitened to ~unit scale by construction).
_COVARIANCE_JITTER = 1e-6


@dataclass
class PreconditionState:
    """Mutable bundle of one SMC run's flow-preconditioning objects.

    Threaded through a run's setup and iteration loop by ``_BlackJAXSMCBase``.
    ``flow``/``flat_params`` are reassigned after each retrain;
    ``optimizer.optim_state`` is mutated in place by flowMC's own ``Optimizer``
    (see ``train_flow``). ``unravel_fn``/``static``/``mcmc_step`` are fixed for
    the run's flow architecture and never change.
    """

    flow: MaskedCouplingRQSpline
    optimizer: Optimizer
    flat_params: Float[Array, " n_flat"]
    unravel_fn: Callable[[Float[Array, " n_flat"]], PyTree]
    static: MaskedCouplingRQSpline
    mcmc_step: PreconditionedMCMCStep


def build_flow(
    n_dims: int, config: PreconditionConfig, key: Key
) -> MaskedCouplingRQSpline:
    """Construct the normalizing flow used for SMC preconditioning.

    Args:
        n_dims: Dimension of the sampling space.
        config: Preconditioning config; reads the ``rq_spline_*`` fields.
        key: JAX PRNG key for parameter initialization.

    Returns:
        A freshly initialized (untrained) flow.
    """
    return MaskedCouplingRQSpline(
        n_features=n_dims,
        n_layers=config.rq_spline_n_layers,
        hidden_size=config.rq_spline_hidden_units,
        num_bins=config.rq_spline_n_bins,
        key=key,
    )


def build_optimizer(
    flow: MaskedCouplingRQSpline, config: PreconditionConfig
) -> Optimizer:
    """Construct the AdamW optimizer (with gradient clipping) used to train the flow.

    Args:
        flow: The flow the optimizer will train.
        config: Preconditioning config; reads ``flow_learning_rate``.

    Returns:
        A freshly initialized ``Optimizer``.
    """
    return Optimizer(model=flow, learning_rate=config.flow_learning_rate)


def partition_flow(
    flow: MaskedCouplingRQSpline,
) -> tuple[
    Float[Array, " n_flat"],
    Callable[[Float[Array, " n_flat"]], PyTree],
    MaskedCouplingRQSpline,
]:
    """Split a flow into a flat trainable-parameter array plus a rebuild closure.

    ``flat_params`` is a 1-D array suitable for BlackJAX's shared-parameter channel
    (``extend_params``/``inner_kernel_tuning``), letting the flow's trained weights
    ride across SMC iterations without forcing the jitted step to retrace:
    ``static``/``unravel_fn`` are closed over once at kernel-construction time, and
    only the flat array's values change.

    Args:
        flow: The (trained or untrained) flow to partition.

    Returns:
        flat_params: 1-D array of the flow's trainable (inexact) leaves.
        unravel_fn: Maps a flat array of the same layout back to the
            trainable-parameters pytree. Stable for a fixed architecture; safe to
            close over for the lifetime of a sampler run.
        static: The non-trainable part of the flow (architecture, buffers).
            Recombine via ``eqx.combine(unravel_fn(flat_params), static)``.
    """
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    flat_params, unravel_fn = ravel_pytree(params)
    return flat_params, unravel_fn, static


def combine_flow(
    flat_params: Float[Array, " n_flat"],
    unravel_fn: Callable[[Float[Array, " n_flat"]], PyTree],
    static: MaskedCouplingRQSpline,
) -> MaskedCouplingRQSpline:
    """Reconstruct a flow from its flattened trainable parameters (inverse of ``partition_flow``).

    Args:
        flat_params: Flat trainable-parameter array, from ``partition_flow``/``flatten_flow``.
        unravel_fn: From ``partition_flow`` on the same flow.
        static: From ``partition_flow`` on the same flow.

    Returns:
        The reconstructed flow.
    """
    return eqx.combine(unravel_fn(flat_params), static)


def flatten_flow(flow: MaskedCouplingRQSpline) -> Float[Array, " n_flat"]:
    """Flatten a flow's trainable parameters to the same layout as ``partition_flow``.

    Used after retraining, since the retrained array only ever feeds back through
    the original ``unravel_fn``/``static`` (the architecture never changes).

    Args:
        flow: The (retrained) flow to flatten.

    Returns:
        Flat array of the flow's trainable (inexact) leaves.
    """
    params, _ = eqx.partition(flow, eqx.is_inexact_array)
    flat_params, _ = ravel_pytree(params)
    return flat_params


def _whitening_scale(flow: MaskedCouplingRQSpline) -> Float[Array, " n_dim"]:
    """Per-dimension whitening scale, floored away from zero.

    Guards a particle-variance collapse in ``flow.data_cov``'s diagonal
    (flowMC's ``train()`` puts no floor of its own on it); see
    ``precondition.md`` for the failure mode this prevents.
    """
    return jnp.maximum(jnp.sqrt(jnp.diag(flow.data_cov)), _WHITENING_SCALE_FLOOR)


def to_latent(
    flow: MaskedCouplingRQSpline, x: Float[Array, " n_dim"]
) -> tuple[Float[Array, " n_dim"], Float[Array, ""]]:
    """Map a point from sampling space to the flow's latent space.

    Composes flowMC's mean/std whitening with ``NFModel.forward`` to expose a clean
    sampling-space <-> latent-space mapping with a tracked log-determinant (flowMC
    applies whitening outside ``forward``/``inverse``, only inside
    ``sample``/``log_prob``).

    Args:
        flow: The (trained or untrained) flow.
        x: Point in sampling space.

    Returns:
        latent: The corresponding point in the flow's latent space.
        log_det: ``log|d latent / d x|``.
    """
    scale = _whitening_scale(flow)
    u = (x - flow.data_mean) / scale
    latent, log_det_fwd = flow.forward(u)
    return latent, log_det_fwd - jnp.sum(jnp.log(scale))


def from_latent(
    flow: MaskedCouplingRQSpline, latent: Float[Array, " n_dim"]
) -> tuple[Float[Array, " n_dim"], Float[Array, ""]]:
    """Map a point from the flow's latent space back to sampling space.

    Inverse of :func:`to_latent`, modulo the round-trip imprecision documented
    in ``precondition.md``.

    Args:
        flow: The (trained or untrained) flow.
        latent: Point in the flow's latent space.

    Returns:
        x: The corresponding point in sampling space.
        log_det: ``log|d x / d latent|``.
    """
    scale = _whitening_scale(flow)
    u, log_det_inv = flow.inverse(latent)
    return u * scale + flow.data_mean, log_det_inv + jnp.sum(jnp.log(scale))


def weighted_covariance(
    particles: Float[Array, "n_particles n_dim"],
    weights: Optional[Float[Array, " n_particles"]] = None,
) -> Float[Array, "n_dim n_dim"]:
    """Covariance of ``particles``, with a small diagonal jitter for positive-definiteness.

    Uses ``ddof=0`` so ``jnp.cov``'s ``aweights`` normalization can't divide by
    exactly zero at total weight collapse (see ``precondition.md``) -- same
    cost as the default ``ddof=1``, just a different divisor, so this is free.
    Deliberately does *not* also branch to an unweighted fallback below some
    ESS threshold: that would compute a second covariance every call in a
    sampling hot path to guard a rare, self-correcting case (one iteration's
    proposal covariance is degenerate, not the NaN this ``ddof`` alone
    already prevents).

    Args:
        particles: Particle population, shape ``(n_particles, n_dim)``.
        weights: Normalized importance weights, shape ``(n_particles,)``, or
            ``None`` for an already-equal-weight population.

    Returns:
        Covariance matrix, shape ``(n_dim, n_dim)``.
    """
    n_dim = particles.shape[-1]
    jitter = _COVARIANCE_JITTER * jnp.eye(n_dim)
    if weights is None:
        return jnp.atleast_2d(jnp.cov(particles.T, ddof=0)) + jitter
    return jnp.atleast_2d(jnp.cov(particles.T, aweights=weights, ddof=0)) + jitter


def training_particles_from_weighted(
    rng_key: Key,
    particles: Float[Array, "n_particles n_dim"],
    weights: Float[Array, " n_particles"],
    n_particles: int,
) -> Float[Array, "n_particles n_dim"]:
    """Resample a weighted particle population into an unweighted training set.

    flowMC's flow-training loss has no support for sample weights. ``ap``/``fp``
    (persistent modes) already hand the mutation phase equal-weight particles, but
    ``at``/``ft`` (plain tempered modes) expose genuinely weighted particles after
    ``step()`` — this draws a fresh, training-only equal-weight sample via SMC's own
    systematic-resampling scheme, without mutating the real SMC state.

    Args:
        rng_key: JAX PRNG key.
        particles: Weighted particle population, shape ``(n_particles, n_dim)``.
        weights: Normalized importance weights, shape ``(n_particles,)``.
        n_particles: Number of particles to draw.

    Returns:
        Unweighted (equal-weight) training particles, shape ``(n_particles, n_dim)``.
    """
    idx = systematic(rng_key, weights, n_particles)
    return particles[idx]


def train_flow(
    rng_key: Key,
    flow: MaskedCouplingRQSpline,
    optimizer: Optimizer,
    particles: Float[Array, "n_particles n_dim"],
    config: PreconditionConfig,
) -> tuple[Key, MaskedCouplingRQSpline, Optimizer, Float[Array, " n_epochs"]]:
    """Retrain the flow on the given particles, warm-starting from its current weights.

    Thin wrapper around flowMC's existing ``NFModel.train(...)``. Defaults to
    full-batch training (``flow_train_batch_size == 0``) to avoid a flowMC quirk
    where the per-epoch recorded/best-model-selection loss is the *last
    minibatch's* loss rather than the epoch mean unless ``batch_size >=
    n_training_particles``.

    Args:
        rng_key: JAX PRNG key.
        flow: The current flow (warm-start).
        optimizer: The current optimizer (its ``optim_state`` is warm-started too;
            mutated in place, matching flowMC's own ``Optimizer`` usage pattern).
        particles: Unweighted training particles, shape ``(n_particles, n_dim)``.
        config: Preconditioning config; reads the ``flow_*`` fields.

    Returns:
        rng_key: Updated JAX PRNG key.
        flow: The retrained flow (best epoch by training loss).
        optimizer: The same ``Optimizer`` object, with ``optim_state`` updated to
            match the returned flow.
        loss_history: Per-epoch loss values, shape ``(flow_n_epochs,)``.
    """
    n_particles = particles.shape[0]
    batch_size = config.flow_train_batch_size
    if batch_size <= 0:
        batch_size = n_particles

    rng_key, flow, optim_state, loss_history = flow.train(
        rng=rng_key,
        data=particles,
        optim=optimizer.optim,
        state=optimizer.optim_state,
        num_epochs=config.flow_n_epochs,
        batch_size=batch_size,
        verbose=False,
    )
    optimizer.optim_state = optim_state  # plain mutable Resource; not a JAX pytree.
    return rng_key, flow, optimizer, loss_history


def build_preconditioned_mcmc_step(
    unravel_fn: Callable[[Float[Array, " n_flat"]], PyTree],
    static: MaskedCouplingRQSpline,
    position_wrapper: Callable[[Float[Array, " n_dim"]], Float[Array, " n_dim"]],
) -> PreconditionedMCMCStep:
    """Return a flow-preconditioned RWMH step: ``(key, state, logdensity, cov, flow_params) -> (state, info)``.

    Same symmetric-proposal RWMH step as ``_BlackJAXSMCBase._build_mcmc_step``'s
    plain kernel, but proposed in the flow's latent space instead of sampling
    space (pocoMC's preconditioned mutation kernel; Karamanis et al. 2022).
    Positions stay in sampling space at the call boundary -- the flow is internal.

    The returned step closes over ``unravel_fn``/``static`` once, at this call;
    it doesn't retrace when only ``flow_params``'s *value* changes between
    calls, since that travels as a shared parameter (BlackJAX's
    ``extend_params``) rather than being closed over -- what makes it safe to
    reuse across a run's SMC iterations despite retraining the flow between them.

    ``position_wrapper`` wraps periodic dimensions back into range after
    ``from_latent``; this is an approximation (uncorrected single-branch
    ratio). Rejection is guarded explicitly with ``info.is_accepted`` since
    ``from_latent(to_latent(x))`` only round-trips approximately. Both are
    known limitations, quantified in ``precondition.md``, not fixed here.

    Args:
        unravel_fn: From ``partition_flow`` on the flow used to build this run's
            kernel. Stable for the run's fixed flow architecture.
        static: From ``partition_flow`` on the same flow (its architecture/buffers).
        position_wrapper: Wraps a sampling-space position's periodic dimensions
            into range; pass ``to_position_wrapper(None, n_dims)`` (a no-op) if
            there are none.

    Returns:
        A step callable with the same shared-parameter calling convention as the
        plain kernel's ``cov``, with an additional shared ``flow_params`` array
        (the flow's current flattened trainable weights).
    """
    kernel = random_walk.build_additive_step()

    def step(key, state, logdensity, cov, flow_params):
        flow = combine_flow(flow_params, unravel_fn, static)

        def logdensity_latent(latent):
            x, log_det = from_latent(flow, latent)
            # Wrap after from_latent (never before), or the recovery below breaks.
            return logdensity(position_wrapper(x)) + log_det

        # Reuses state.logdensity (already exact) instead of recomputing via from_latent.
        latent0, log_det_fwd = to_latent(flow, state.position)
        latent_state = state._replace(
            position=latent0, logdensity=state.logdensity - log_det_fwd
        )

        def proposal_distribution(key, position):
            return jax.random.multivariate_normal(key, jnp.zeros_like(position), cov)

        new_latent_state, info = kernel(
            key, latent_state, logdensity_latent, proposal_distribution
        )

        # Reuses new_latent_state.logdensity -- avoids an extra call since from_latent is pure.
        x_proposed, log_det_new = from_latent(flow, new_latent_state.position)  # type: ignore[arg-type]  # RWState.position is ArrayTree; always a flat Array here
        x_proposed = position_wrapper(x_proposed)  # after from_latent, before use
        x_new = jnp.where(info.is_accepted, x_proposed, state.position)
        logdensity_new = jnp.where(
            info.is_accepted,
            new_latent_state.logdensity - log_det_new,
            state.logdensity,
        )
        new_state = state._replace(position=x_new, logdensity=logdensity_new)
        return new_state, info

    return step
