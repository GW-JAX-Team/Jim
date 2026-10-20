"""Ensemble mutation for the BlackJAX SMC sampler modes.

A mutation moves every particle at one temperature.  The particles first go to
coordinates fitted to the current ensemble: periodic angles are re-centred onto
the real line and, when a flow is configured, mapped through a normalizing flow
trained on the particles.  A kernel (a Gaussian random walk or t-pCN) then takes
Metropolis-Hastings steps in those coordinates.  The acceptance ratio includes
the Jacobian of the map, so the particles still target Jim's posterior.  BlackJAX
runs the mutation through its ``update_strategy`` hook.
"""

from collections.abc import Callable
from typing import Any, Literal, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.smc import extend_params
from blackjax.smc.base import map_fn
from flowMC.resource.model.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, Key

from jimgw.samplers.config import (
    SMCFlowConfig,
    SMCGRWConfig,
    SMCKernelConfig,
    SMCTPCNConfig,
)
from jimgw.samplers.periodic import to_recentring_map

# ----------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------


class _Kernel:
    """Base of the mutation kernels, which work in the mutation's coordinates ``z``.

    A kernel provides:

    * ``fit(z)``: what it fits to the particles at the start of a mutation;
    * ``statistic(z, fit)``: a per-particle value its proposal needs;
    * ``propose(key, z, statistic, fit, scale)``: the proposed ``z``, its
      statistic and the log proposal ratio ``log q(z | z') - log q(z' | z)``;
    * ``adapt(scale, n_steps, mean_acceptance)``: the step size after a step;
    * ``stop(n_steps, patience_count, scale)``: whether the mutation ends;
    * ``next_scale(scale, acceptance)``: the step size the next mutation starts with;
    * ``stop_reason(n_steps)``: why the mutation ended, for the log.
    """

    # Ridge added to a covariance, relative to each variance.
    ridge = 1e-6

    def particle_covariance(
        self, z: Float[Array, "n n_dim"]
    ) -> tuple[Float[Array, "n_dim n_dim"], Float[Array, "n_dim n_dim"]]:
        """The particles' covariance plus the ridge, and the ridge itself."""
        cov = jnp.atleast_2d(jnp.cov(z.T))
        ridge = self.ridge * jnp.diag(jnp.diag(cov))
        return cov + ridge, ridge

    def cholesky(self, cov: Float[Array, "n_dim n_dim"]) -> Float[Array, "n_dim n_dim"]:
        """Cholesky factor of ``cov``, or its diagonal square root if that fails."""
        chol = jnp.linalg.cholesky(cov)
        return jnp.where(
            jnp.all(jnp.isfinite(chol)), chol, jnp.diag(jnp.sqrt(jnp.diag(cov)))
        )


class _GRWKernel(_Kernel):
    """Gaussian random walk, proposing ``z + sqrt(s) L eps``.

    ``L L^T`` is the particles' covariance.  A mutation takes a fixed number of
    steps; between mutations, ``s`` moves the acceptance towards its target.
    """

    scale_name = "s"

    def __init__(self, config: SMCGRWConfig, n_dims: int) -> None:
        self.config = config
        self.max_steps = config.n_mcmc_steps_per_dim * n_dims
        self.initial_scale = config.initial_cov_scale

    def fit(self, z: Float[Array, "n n_dim"]) -> dict[str, Array]:
        cov, _ = self.particle_covariance(z)
        return {"chol": self.cholesky(cov), "nu": jnp.asarray(jnp.nan, z.dtype)}

    def statistic(self, z: Float[Array, "n n_dim"], fit: dict) -> Float[Array, " n"]:
        return jnp.zeros(z.shape[0], z.dtype)

    def propose(self, key, z, statistic, fit, scale):
        eps = jax.random.normal(key, z.shape, z.dtype) @ fit["chol"].T
        return z + jnp.sqrt(scale) * eps, statistic, jnp.zeros_like(statistic)

    def adapt(self, scale, n_steps, mean_acceptance):
        return scale

    def stop(self, n_steps, patience_count, scale):
        return n_steps >= self.max_steps

    def next_scale(self, scale, acceptance):
        gain = self.config.scale_adaptation_gain
        return scale * jnp.exp(gain * (acceptance - self.config.target_acceptance_rate))

    def stop_reason(self, n_steps: int) -> str:
        return "fixed"


class _TPCNKernel(_Kernel):
    """t-pCN, proposing ``mu + sqrt(1 - rho^2) (z - mu) + rho sqrt(W) L eps``.

    ``mu``, ``L L^T`` and ``nu`` are a Student-t fit to the particles, and ``W``
    is an inverse-gamma variance drawn per particle.  ``rho`` adapts during a
    mutation, which stops once the mean log posterior stops improving.
    """

    scale_name = "rho"
    # Degrees of freedom tried for the Student-t fit, and expectation-maximization
    # iterations per value.
    n_nu = 24
    em_iterations = 50

    def __init__(self, config: SMCTPCNConfig, n_dims: int) -> None:
        self.config = config
        self.max_steps = config.n_max_per_dim * n_dims
        self.patience = config.n_patience_per_dim * n_dims
        self.max_scale = min(2.38 / n_dims**0.5, 0.99)
        self.initial_scale = self.max_scale

    def fit(self, z: Float[Array, "n n_dim"]) -> dict[str, Array]:
        n, d = z.shape
        mean0 = jnp.mean(z, axis=0)
        cov0, ridge = self.particle_covariance(z)

        def mahalanobis(mean, chol):
            # The inverse factor turns one triangular solve per particle into a
            # matrix product (much faster on CPU).
            return jnp.sum(((z - mean) @ jnp.linalg.inv(chol).T) ** 2, 1)

        def fit_nu(nu):
            def em_step(_, carry):
                mean, cov = carry
                delta = mahalanobis(mean, jnp.linalg.cholesky(cov))
                weights = (nu + d) / (nu + delta)
                mean = jnp.sum(weights[:, None] * z, 0) / jnp.sum(weights)
                centred = z - mean
                return mean, (weights[:, None] * centred).T @ centred / n + ridge

            mean, cov = jax.lax.fori_loop(0, self.em_iterations, em_step, (mean0, cov0))
            chol = jnp.linalg.cholesky(cov)
            log_likelihood = jnp.sum(
                gammaln(0.5 * (nu + d))
                - gammaln(0.5 * nu)
                - 0.5 * d * jnp.log(nu * jnp.pi)
                - jnp.sum(jnp.log(jnp.diag(chol)))
                - 0.5 * (nu + d) * jnp.log1p(mahalanobis(mean, chol) / nu)
            )
            return mean, cov, log_likelihood

        nu_grid = jnp.logspace(0.0, 3.0, self.n_nu, dtype=z.dtype)
        means, covs, log_likelihoods = jax.vmap(fit_nu)(nu_grid)
        log_likelihoods = jnp.where(
            jnp.isfinite(log_likelihoods), log_likelihoods, -jnp.inf
        )
        best = jnp.argmax(log_likelihoods)
        fitted = jnp.isfinite(log_likelihoods[best])
        chol = self.cholesky(jnp.where(fitted, covs[best], cov0))
        return {
            "mean": jnp.where(fitted, means[best], mean0),
            "chol": chol,
            "inverse_chol": jnp.linalg.inv(chol),
            "nu": jnp.where(fitted, nu_grid[best], nu_grid[-1]),
        }

    def statistic(self, z: Float[Array, "n n_dim"], fit: dict) -> Float[Array, " n"]:
        return jnp.sum(((z - fit["mean"]) @ fit["inverse_chol"].T) ** 2, 1)

    def propose(self, key, z, statistic, fit, scale):
        n, d = z.shape
        gamma_key, normal_key = jax.random.split(key)
        nu, mean = fit["nu"], fit["mean"]
        eps = jax.random.normal(normal_key, z.shape, z.dtype) @ fit["chol"].T
        # One inverse-gamma variance per particle makes the move reversible
        # with respect to the Student-t fit.
        gamma = jax.random.gamma(gamma_key, 0.5 * (nu + d), (n,), z.dtype)
        variance = (nu + statistic) / (2.0 * gamma)
        z_new = (
            mean
            + jnp.sqrt(1.0 - scale**2) * (z - mean)
            + scale * jnp.sqrt(variance)[:, None] * eps
        )
        statistic_new = self.statistic(z_new, fit)
        log_q = (
            0.5 * (nu + d) * (jnp.log1p(statistic_new / nu) - jnp.log1p(statistic / nu))
        )
        return z_new, statistic_new, log_q

    def adapt(self, scale, n_steps, mean_acceptance):
        target = self.config.target_acceptance_rate
        updated = scale + (n_steps + 1.0) ** -0.75 * (mean_acceptance - target)
        return jnp.abs(jnp.minimum(updated, self.max_scale))

    def stop(self, n_steps, patience_count, scale):
        plateau = patience_count >= self.patience * (self.max_scale / scale) ** 2
        return plateau | (n_steps >= self.max_steps)

    def next_scale(self, scale, acceptance):
        return scale

    def stop_reason(self, n_steps: int) -> str:
        return "n_max" if n_steps >= self.max_steps else "plateau"


# ----------------------------------------------------------------------
# Mutation
# ----------------------------------------------------------------------


class EnsembleMutation:
    """Moves all SMC particles at one temperature, in coordinates fitted to them.

    Hand ``read_parameters``, ``mcmc_init_fn``, ``update_strategy`` and
    ``mcmc_parameter_update_fn`` to ``blackjax.inner_kernel_tuning``, with
    ``extend_params(initial_parameter_value())`` as the initial parameter value and
    ``max_steps`` as ``num_mcmc_steps``.  The only parameter carried from one
    mutation to the next is the kernel's step size; a flow is trained afresh
    at every mutation.

    Args:
        n_dims: Dimension of the sampling space.
        periodic: Periodic dimensions as ``{index: (lower, upper)}``, or ``None``.
        kernel_config: Config of the kernel that moves the particles.
        flow_config: Config of the normalizing flow, or ``False`` for none.
    """

    def __init__(
        self,
        *,
        n_dims: int,
        periodic: Optional[dict[int, tuple[float, float]]],
        kernel_config: SMCKernelConfig,
        flow_config: Union[SMCFlowConfig, Literal[False]],
    ) -> None:
        self._kernel = (
            _TPCNKernel(kernel_config, n_dims)
            if isinstance(kernel_config, SMCTPCNConfig)
            else _GRWKernel(kernel_config, n_dims)
        )
        self._flow_config = flow_config
        self._cuts_fn, self._to_unbounded, self._from_unbounded = to_recentring_map(
            periodic, n_dims
        )
        self.max_steps = self._kernel.max_steps
        if flow_config is not False:
            # Every mutation starts from this flow and a fresh optimizer state.
            self._initial_flow = MaskedCouplingRQSpline(
                n_features=n_dims,
                n_layers=flow_config.n_layers,
                hidden_size=flow_config.hidden_size,
                num_bins=flow_config.n_bins,
                key=jax.random.key(0),
            )
            self._optimizer = Optimizer(
                self._initial_flow, learning_rate=flow_config.learning_rate
            )

    # ------------------------------------------------------------------
    # BlackJAX hooks
    # ------------------------------------------------------------------

    def initial_parameter_value(self) -> dict[str, Array]:
        """Carried parameters for the first mutation (pass through ``extend_params``)."""
        # A strong dtype, as the scale computed by a mutation has: a weak one would
        # make jit compile the step a second time.
        return {"scale": jnp.asarray(self._kernel.initial_scale, dtype=float)}

    def read_parameters(self, **parameters: Array) -> dict[str, Array]:
        """BlackJAX's ``mcmc_step_fn``, used only to read the carried parameters.

        BlackJAX binds the carried parameters to ``mcmc_step_fn``;
        ``update_strategy`` calls it with no arguments to get them back.
        """
        return parameters

    def mcmc_init_fn(self, position: Array, logdensity_fn: Callable) -> Any:
        """BlackJAX's ``mcmc_init_fn``; never called, as there is no per-particle state."""
        raise NotImplementedError("the ensemble mutation keeps no per-particle state")

    def mcmc_parameter_update_fn(
        self, key: Key[Array, ""], state: Any, info: Any
    ) -> dict[str, Array]:
        """BlackJAX's ``mcmc_parameter_update_fn``: carries the parameters forward."""
        return extend_params(info.update_info["next_parameters"])  # ty: ignore[invalid-return-type]  # blackjax stubs: extend_params returns its dict

    def update_strategy(
        self,
        mcmc_init_fn: Callable,
        logposterior_fn: Callable,
        shared_mcmc_step_fn: Callable,
        num_mcmc_steps: int,
        n_particles: int,
        batch_size: int = 0,
    ) -> tuple[Callable, int]:
        """BlackJAX's ``update_strategy``: returns ``(update_fn, n_particles)``.

        ``update_fn`` runs one mutation of all particles and returns the moved
        particles and the mutation's ``info`` dict.
        """
        parameters = shared_mcmc_step_fn()
        log_posterior = map_fn(logposterior_fn, batch_size)
        kernel = self._kernel

        def update_fn(keys, x0, unshared_parameters):
            coordinates_key, unique_key, loop_key = jax.random.split(keys[0], 3)
            cuts = self._cuts_fn(x0)
            u0 = jax.vmap(self._to_unbounded, in_axes=(0, None))(x0, cuts)
            to_latent, from_latent, training_loss = self._coordinates(
                coordinates_key, u0, cuts
            )
            z0 = jax.vmap(to_latent)(u0)
            _, log_det0 = jax.vmap(from_latent)(z0)
            fit = kernel.fit(z0)
            lp0 = log_posterior(x0)

            # ----------------------------------------------------------
            # Metropolis-Hastings steps until the kernel stops
            # ----------------------------------------------------------

            def step(carry):
                key, propose_key, accept_key = jax.random.split(carry["key"], 3)
                z_new, statistic_new, log_q = kernel.propose(
                    propose_key, carry["z"], carry["statistic"], fit, carry["scale"]
                )
                x_new, log_det_new = jax.vmap(from_latent)(z_new)
                lp_new = log_posterior(x_new)
                log_target_new = lp_new + log_det_new
                log_ratio = log_target_new - carry["log_target"] + log_q
                log_ratio = jnp.where(jnp.isfinite(log_ratio), log_ratio, -jnp.inf)
                accept = jnp.log(jax.random.uniform(accept_key, lp_new.shape)) < (
                    log_ratio
                )
                mean_acceptance = jnp.mean(jnp.exp(jnp.minimum(log_ratio, 0.0)))
                lp = jnp.where(accept, lp_new, carry["lp"])
                # Plateau rule: count steps since the mean log posterior last rose.
                mean_lp = jnp.mean(lp)
                improved = mean_lp > carry["best_mean_lp"]
                patience_count = jnp.where(improved, 0, carry["patience_count"] + 1)
                n_steps = carry["n_steps"] + 1
                scale = kernel.adapt(carry["scale"], n_steps, mean_acceptance)
                return {
                    "key": key,
                    "z": jnp.where(accept[:, None], z_new, carry["z"]),
                    "x": jnp.where(accept[:, None], x_new, carry["x"]),
                    "lp": lp,
                    "log_target": jnp.where(
                        accept, log_target_new, carry["log_target"]
                    ),
                    "statistic": jnp.where(accept, statistic_new, carry["statistic"]),
                    "scale": scale,
                    "best_mean_lp": jnp.maximum(mean_lp, carry["best_mean_lp"]),
                    "patience_count": patience_count,
                    "n_steps": n_steps,
                    "acceptance_sum": carry["acceptance_sum"] + mean_acceptance,
                    "done": kernel.stop(n_steps, patience_count, scale),
                }

            initial = {
                "key": loop_key,
                "z": z0,
                "x": x0,
                "lp": lp0,
                "log_target": lp0 + log_det0,
                "statistic": kernel.statistic(z0, fit),
                "scale": parameters["scale"],
                "best_mean_lp": jnp.mean(lp0),
                "patience_count": jnp.asarray(0),
                "n_steps": jnp.asarray(0),
                "acceptance_sum": jnp.zeros((), x0.dtype),
                "done": jnp.asarray(False),
            }
            final = jax.lax.while_loop(lambda carry: ~carry["done"], step, initial)

            acceptance = final["acceptance_sum"] / final["n_steps"]
            # Distinct particles, counted along a random projection.
            projection = x0 @ jax.random.normal(unique_key, (x0.shape[1],), x0.dtype)
            info = {
                "acceptance_rate": acceptance,
                "n_steps": final["n_steps"],
                "scale": final["scale"],
                "nu": fit["nu"],
                "n_unique": 1 + jnp.sum(jnp.diff(jnp.sort(projection)) != 0),
                "next_parameters": {
                    "scale": kernel.next_scale(final["scale"], acceptance)
                },
            }
            if training_loss is not None:
                info["training_loss"] = training_loss
            return final["x"], info

        return update_fn, n_particles

    # ------------------------------------------------------------------
    # Coordinates and reporting
    # ------------------------------------------------------------------

    def _coordinates(
        self,
        key: Key[Array, ""],
        u0: Float[Array, "n n_dim"],
        cuts: Float[Array, " n_dim"],
    ) -> tuple[Callable, Callable, Optional[Array]]:
        """The kernel's coordinates ``z`` for one mutation.

        Returns ``(to_latent, from_latent, training_loss)``:

        * ``to_latent(u) -> z`` from the re-centred coordinates ``u``;
        * ``from_latent(z) -> (x, log_det)`` to the sampling space, with
          ``log_det = log|dx/dz|``;
        * ``training_loss``: the flow's loss per epoch.

        With a flow, a new flow first trains on ``u0``.  Without one, ``z = u``
        and ``training_loss`` is ``None``.
        """
        from_unbounded = self._from_unbounded

        if self._flow_config is False:

            def identity(u):
                return u

            def from_unbounded_latent(z):
                return from_unbounded(z, cuts)

            return identity, from_unbounded_latent, None

        _, flow, _, training_loss = self._initial_flow.train_jit(
            key,
            u0,
            self._optimizer.optim,
            self._optimizer.optim_state,
            self._flow_config.n_epochs,
            self._flow_config.batch_size,
        )
        # The flow rescales its input with the data statistics it was trained with.
        mean = flow.data_mean
        std = jnp.sqrt(jnp.diag(flow.data_cov))
        log_std = jnp.sum(jnp.log(std))

        def to_latent(u):
            return flow.forward((u - mean) / std)[0]

        def from_latent(z):
            y, log_det_flow = flow.inverse(z)
            x, log_det_periodic = from_unbounded(mean + std * y, cuts)
            return x, log_det_flow + log_std + log_det_periodic

        return to_latent, from_latent, training_loss

    def summary(self, update_info: dict) -> dict[str, Any]:
        """One mutation's numbers as Python values, for the mutation history and logging."""
        n_steps = int(update_info["n_steps"])
        training_loss = update_info.get("training_loss")
        return {
            "acceptance": float(update_info["acceptance_rate"]),
            "n_steps": n_steps,
            "stop_reason": self._kernel.stop_reason(n_steps),
            "scale": float(update_info["scale"]),
            "scale_name": self._kernel.scale_name,
            "nu": float(update_info["nu"]),
            "n_unique": int(update_info["n_unique"]),
            "training_loss": None
            if training_loss is None
            else np.asarray(training_loss),
        }
