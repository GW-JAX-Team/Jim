import logging
from collections.abc import Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key
from ripplegw.interfaces import Waveform

from jimgw.core.base import LikelihoodBase
from jimgw.core.prior import Prior
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.core.single_event.likelihood import (
    SingleEventLikelihood,
    TransientLikelihoodFD,
)
from jimgw.samplers import Sampler, SamplerConfig, build_sampler

logger = logging.getLogger(__name__)

# Number of prior draws used to verify the posterior at construction time.
# More than half returning NaN is treated as a hard error; any non-zero count
# triggers a warning.
_NAN_TEST_POINTS = 10
_NAN_FAIL_THRESHOLD = 5

# Fixed key used for downsampling in get_samples.
_DOWNSAMPLE_KEY: Key = jax.random.key(42)


class Jim:
    """Master class for gravitational-wave parameter estimation.

    Wires together a [`LikelihoodBase`][jimgw.core.base.LikelihoodBase], a
    [`Prior`][jimgw.core.prior.Prior], optional parameter transforms, and a
    pluggable JAX [`Sampler`][jimgw.samplers.base.Sampler] selected via a typed
    ``sampler_config`` object.
    """

    likelihood: LikelihoodBase
    prior: Prior
    sample_transforms: Sequence[BijectiveTransform]
    likelihood_transforms: Sequence[NtoMTransform]
    parameter_names: tuple[str, ...]
    sampler: Sampler

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sampler_config: SamplerConfig,
        *,
        sample_transforms: Sequence[BijectiveTransform] = (),
        likelihood_transforms: Sequence[NtoMTransform] = (),
        periodic: Optional[list[str] | dict[str, tuple[float, float]]] = None,
        seed: int = 0,
    ) -> None:
        """Initialise Jim and build the internal sampler.

        Args:
            likelihood: The likelihood to evaluate.
            prior: The prior distribution.
            sampler_config: Pydantic config selecting and configuring the
                sampler backend (e.g. [`FlowMCConfig`][jimgw.samplers.config.FlowMCConfig]).
            sample_transforms: Bijective transforms applied in the sampling
                space (reversed when retrieving posterior samples).
            likelihood_transforms: Transforms applied to reach the likelihood
                parameter space from the prior parameter space.
            periodic: Periodic sampling-space parameters.  For most samplers,
                pass a ``dict`` mapping parameter name to ``(lo, hi)`` bounds
                (e.g. ``{"phase_c": (0.0, 6.2832)}``).  For the BlackJAX
                NS-AW sampler (unit-cube space), pass a ``list`` of parameter
                names (bounds are implicit as ``[0, 1]``).
            seed: Integer random seed. The key for the sampling run is derived
                from this seed at construction time, so `sample` is
                reproducible regardless of any intermediate operations (sanity
                checks, initial-position draws, etc.).
        """
        self._validate_problem(likelihood, prior, likelihood_transforms)
        self._setup_problem(likelihood, prior, sample_transforms, likelihood_transforms)
        self._validate_normalized_prior(prior, sampler_config)
        root_key: Key = jax.random.key(seed)

        # Reserve _sampler_key immediately so sampling is reproducible even if
        # sanity checks or other internal splits consume _rng_key first.
        self._rng_key, self._sampler_key = jax.random.split(root_key)
        self._sampler_config = sampler_config

        # Resolve periodic parameter names → dimension indices
        if periodic is not None:
            names = self.parameter_names

            unknown = [n for n in periodic if n not in names]
            if unknown:
                raise ValueError(
                    f"Periodic parameter(s) {unknown} not found in "
                    f"sampling parameters {tuple(self.parameter_names)}."
                )

            if isinstance(periodic, list):
                # NS-AW style: list[str] → list[int].
                if periodic and sampler_config.type != "blackjax-ns-aw":
                    raise ValueError(
                        "List-form periodic (names without bounds) is only supported for "
                        "the 'blackjax-ns-aw' sampler. For other samplers pass a dict "
                        "mapping parameter names to (lo, hi) bounds, e.g. "
                        '{"phase_c": (0.0, 6.2832)}.'
                    )
                periodic_resolved = [names.index(n) for n in periodic]
            elif isinstance(periodic, dict):
                # dict[str, (lo, hi)] → dict[int, (lo, hi)]
                periodic_resolved = {names.index(k): v for k, v in periodic.items()}
        else:
            periodic_resolved = None

        self.sampler = build_sampler(
            sampler_config,
            n_dims=len(self.parameter_names),
            log_prior_fn=self._log_prior_fn,
            log_likelihood_fn=self._log_likelihood_fn,
            log_posterior_fn=self._log_posterior_fn,
            periodic=periodic_resolved,
        )
        self._verify_posterior()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_problem(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        likelihood_transforms: Sequence[NtoMTransform],
    ) -> None:
        """Validate that the prior and likelihood parameter spaces are compatible.

        Args:
            likelihood: The likelihood to evaluate.
            prior: The prior distribution.
            likelihood_transforms: Transforms from prior space to likelihood space.

        Raises:
            ValueError: If prior parameters overlap with fixed parameters, if prior
                parameters are not consumed by the likelihood, or if the likelihood
                requires parameters not provided by the prior or fixed_parameters.
        """
        if not isinstance(likelihood, SingleEventLikelihood):
            return

        lh_space_names: tuple[str, ...] = prior.parameter_names
        for transform in likelihood_transforms:
            lh_space_names = transform.propagate_name(lh_space_names)

        if likelihood.fixed_parameters:
            overlap = set(lh_space_names) & set(likelihood.fixed_parameters.keys())
            if overlap:
                raise ValueError(
                    f"Prior defines parameter(s) {sorted(overlap)} that are "
                    "also in fixed_parameters. Either remove them from the prior "
                    "or from fixed_parameters."
                )

        # Waveforms that publish a `parameter_names` attribute can be
        # cross-checked against the prior.
        wf_param_names = getattr(likelihood.waveform, "parameter_names", None)
        if not (
            isinstance(likelihood.waveform, Waveform) and wf_param_names is not None
        ):
            return

        consumed: set[str] = set(wf_param_names)
        consumed |= {"ra", "dec", "psi", "t_c"}
        if isinstance(likelihood, TransientLikelihoodFD):
            if likelihood.time_marginalization:
                consumed.discard("t_c")
            if likelihood.phase_marginalization:
                consumed.discard("phase_c")
            if likelihood.distance_marginalization:
                consumed.discard("d_L")

        provided = set(lh_space_names)
        if likelihood.fixed_parameters:
            provided |= set(likelihood.fixed_parameters.keys())

        unused = provided - consumed
        if unused:
            raise ValueError(
                f"Prior defines parameter(s) {sorted(unused)} that are not "
                "consumed by the likelihood. Remove them from the prior or "
                "add appropriate likelihood_transforms."
            )
        missing = consumed - provided
        if missing:
            raise ValueError(
                f"Likelihood requires parameter(s) {sorted(missing)} that are "
                "not provided by the prior or fixed_parameters. Add them to "
                "the prior or to fixed_parameters."
            )

    def _setup_problem(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: Sequence[BijectiveTransform],
        likelihood_transforms: Sequence[NtoMTransform],
    ) -> None:
        """Wire together likelihood, prior, and transforms; build sampling-space callables.

        Constructs ``_log_prior_fn``, ``_log_likelihood_fn``, and
        ``_log_posterior_fn`` — flat-array callables injected into the sampler.
        Validation is performed separately by :meth:`_validate_problem` before
        this method is called.

        Args:
            likelihood: The likelihood to evaluate.
            prior: The prior distribution.
            sample_transforms: Bijective transforms from prior space to sampling space.
            likelihood_transforms: Transforms from prior space to likelihood space.
        """
        self.likelihood = likelihood
        self.prior = prior
        self.sample_transforms = sample_transforms
        self.likelihood_transforms = likelihood_transforms

        self.parameter_names = prior.parameter_names
        if not sample_transforms:
            logger.info(
                "No sample transforms provided. Using prior parameters as sampling parameters."
            )
        else:
            logger.info("Using sample transforms.")
            for transform in sample_transforms:
                self.parameter_names = transform.propagate_name(self.parameter_names)
                logger.debug(
                    f"  Applied transform {type(transform).__name__}: parameter_names = {self.parameter_names}"
                )

        if not likelihood_transforms:
            logger.info(
                "No likelihood transforms provided. Using prior parameters as likelihood parameters."
            )
        else:
            logger.debug(
                f"Using {len(likelihood_transforms)} likelihood transform(s): {[type(t).__name__ for t in likelihood_transforms]}"
            )

        # Build sampling-space callables. These operate on flat arrays of shape
        # (n_dims,) and are injected into the sampler.
        names = self.parameter_names

        def _log_prior_fn(arr: Float[Array, " n_dims"]) -> Float:
            named = dict(zip(names, arr, strict=True))
            jac: Float = 0.0
            for transform in reversed(sample_transforms):
                named, j = transform.inverse(named)
                jac += j
            return prior.log_prob(named) + jac

        def _log_likelihood_fn(arr: Float[Array, " n_dims"]) -> Float:
            named = dict(zip(names, arr, strict=True))
            for transform in reversed(sample_transforms):
                named, _ = transform.inverse(named)
            for transform in likelihood_transforms:
                named = transform.forward(named)
            return likelihood.evaluate(named, {})

        def _log_posterior_fn(arr: Float[Array, " n_dims"]) -> Float:
            named = dict(zip(names, arr, strict=True))
            jac: Float = 0.0
            for transform in reversed(sample_transforms):
                named, j = transform.inverse(named)
                jac = jac + j
            log_prior = prior.log_prob(named) + jac
            for transform in likelihood_transforms:
                named = transform.forward(named)
            return likelihood.evaluate(named, {}) + log_prior

        self._log_prior_fn = _log_prior_fn
        self._log_likelihood_fn = _log_likelihood_fn
        self._log_posterior_fn = _log_posterior_fn

    def _verify_posterior(self) -> None:
        """Draw test points from the prior and verify the posterior is not mostly NaN.

        Raises:
            ValueError: If more than ``_NAN_FAIL_THRESHOLD`` out of
                ``_NAN_TEST_POINTS`` test points return NaN posterior values.
        """
        self._rng_key, check_key = jax.random.split(self._rng_key)
        check_positions = self._draw_initial_positions(check_key, _NAN_TEST_POINTS)
        log_posteriors = jax.vmap(self._log_posterior_fn)(check_positions)
        n_nan = int(jnp.sum(jnp.isnan(log_posteriors)))
        if n_nan > _NAN_FAIL_THRESHOLD:
            raise ValueError(
                f"The posterior returned NaN for {n_nan}/{_NAN_TEST_POINTS} test "
                "points sampled from the prior. Check your likelihood and "
                "transforms for correctness."
            )
        elif n_nan > 0:
            logger.warning(
                "%d/%d test points sampled from the prior returned NaN posterior "
                "values. This may indicate issues at the boundaries of your prior.",
                n_nan,
                _NAN_TEST_POINTS,
            )

    def _validate_normalized_prior(
        self, prior: Prior, sampler_config: SamplerConfig
    ) -> None:
        """Raise if a normalization-requiring sampler is paired with an unnormalized prior.

        Args:
            prior: The prior to check.
            sampler_config: The sampler configuration to check against.

        Raises:
            ValueError: If ``sampler_config`` is a
                [`BlackJAXNSSConfig`][jimgw.samplers.config.BlackJAXNSSConfig] or
                [`BlackJAXSMCConfig`][jimgw.samplers.config.BlackJAXSMCConfig] and
                ``prior.is_normalized`` is ``False``.
        """
        from jimgw.samplers.config import BlackJAXNSSConfig, BlackJAXSMCConfig

        if (
            isinstance(sampler_config, (BlackJAXNSSConfig, BlackJAXSMCConfig))
            and not prior.is_normalized
        ):
            raise ValueError(
                f"{type(sampler_config).__name__} computes Bayesian evidence and "
                "therefore requires a normalized prior (∫ exp(log_prob(x)) dx = 1). "
                "If your custom prior is normalized, override the is_normalized "
                "property to return True."
            )

    def _draw_initial_positions(self, key: Key, n: int) -> Float[Array, "n n_dims"]:
        """Sample ``n`` initial positions from the prior in sampling space.

        Args:
            key: JAX PRNG key.
            n: Number of positions to draw.

        Returns:
            Array of shape ``(n, n_dims)`` in sampling space.

        Raises:
            ValueError: If any drawn position contains non-finite values.
        """
        initial = self.prior.sample(key, n)
        for transform in self.sample_transforms:
            initial = jax.vmap(transform.forward)(initial)
        arr = jnp.array([initial[name] for name in self.parameter_names]).T
        if not jnp.all(jnp.isfinite(arr)):
            raise ValueError(
                f"Initial positions contain non-finite values (NaN or inf) after {max_attempts} attempts. "
                "Check your priors and transforms for validity."
            )
        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_name(self, x: Float[Array, " n_dims"]) -> dict[str, Float]:
        """Convert a flat sampling-space array to a named dict."""
        return dict(zip(self.parameter_names, x, strict=True))

    def evaluate_prior(self, params: Float[Array, " n_dims"]) -> Float:
        """Log-prior in the sampling space (with Jacobian corrections from sample_transforms)."""
        return self._log_prior_fn(params)

    def evaluate_posterior(self, params: Float[Array, " n_dims"]) -> Float:
        """Log-posterior in the sampling space."""
        return self._log_posterior_fn(params)

    def sample_initial_positions(
        self,
        n_points: int,
        rng_key: Optional[Key] = None,
    ) -> Float[Array, "n_points n_dims"]:
        """Draw ``n_points`` initial positions from the prior in sampling space.

        Args:
            n_points: Number of positions to draw.
            rng_key: Optional explicit PRNG key. If ``None``, Jim's internal
                auxiliary key is advanced automatically.

        Returns:
            Array of shape ``(n_points, n_dims)`` in sampling space.
        """
        if rng_key is None:
            self._rng_key, rng_key = jax.random.split(self._rng_key)
        return self._draw_initial_positions(rng_key, n_points)

    def sample(
        self,
        initial_position: Optional[Float[Array, "n_chains n_dims"]] = None,
    ) -> None:
        """Run the sampler.

        The sampling key is pre-reserved at construction time from ``seed``,
        so results are reproducible regardless of any calls made before this
        method (e.g. the construction-time posterior verification).

        Args:
            initial_position: Starting positions in sampling space, or
                ``None`` (default) to draw them from the prior. The
                expected shape depends on the backend:

                - flowMC: ``(n_chains, n_dims)`` or ``(n_dims,)`` (broadcast
                  to all chains).
                - BlackJAX NS-AW / NSS: exactly ``(n_live, n_dims)``.
                - BlackJAX SMC: exactly ``(n_particles, n_dims)``.

                The concrete sampler validates the shape and raises
                ``ValueError`` on mismatch.
        """
        if initial_position is None:
            cfg = self._sampler_config
            counts = {
                attr: getattr(cfg, attr)
                for attr in ("n_chains", "n_live", "n_particles")
                if hasattr(cfg, attr)
            }
            if len(counts) != 1:
                raise TypeError(
                    f"Cannot determine number of initial positions from "
                    f"{type(cfg).__name__}: expected exactly one of n_chains, "
                    f"n_live, n_particles, found {list(counts)}"
                )
            n = next(iter(counts.values()))
            self._rng_key, init_key = jax.random.split(self._rng_key)
            initial_position = self._draw_initial_positions(init_key, n)
        self.sampler.sample(self._sampler_key, initial_position)

    def get_samples(
        self,
        n_samples: int = 0,
    ) -> dict[str, np.ndarray]:
        """Retrieve posterior samples in prior space, optionally downsampled.

        Calls [`Sampler.get_samples`][jimgw.samplers.base.Sampler.get_samples] on the
        underlying sampler, which returns equally-weighted posterior samples.
        Pass ``n_samples`` to further downsample.

        Args:
            n_samples: Target number of samples.  If 0 (default) returns all
            available samples, otherwise downsample uniformly without replacement.

        Returns:
            Dict mapping prior parameter names to 1-D numpy arrays in prior
            space, plus an extra item containing the log-likelihood values.
        """
        result = self.sampler.get_samples()
        sample_array = result["samples"]  # (n, n_dims) in sampling space
        log_likelihood = result["log_likelihood"]  # (n,)
        n_available = sample_array.shape[0]

        if n_samples > 0:
            if n_samples > n_available:
                logger.warning(
                    "Requested %d samples but only %d available. Returning all available samples.",
                    n_samples,
                    n_available,
                )
                n_samples = n_available
            if n_samples < n_available:
                indices = np.array(
                    jax.random.choice(
                        _DOWNSAMPLE_KEY, n_available, shape=(n_samples,), replace=False
                    )
                )
                sample_array = sample_array[indices]
                log_likelihood = log_likelihood[indices]

        # Backward-transform from sampling space to prior space and add names.
        named = jax.vmap(self.add_name)(jnp.array(sample_array))
        for transform in reversed(self.sample_transforms):
            named = jax.vmap(transform.backward)(named)
        out = {k: np.array(named[k]) for k in self.prior.parameter_names}
        out["log_likelihood"] = np.asarray(log_likelihood)
        return out

    def get_diagnostics(self) -> dict[str, Any]:
        """Return run-level diagnostics from the most recent `sample` call.

        Returns:
            Plain dict of backend-specific diagnostics.
        """
        return self.sampler.get_diagnostics()
