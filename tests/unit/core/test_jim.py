"""Unit tests for the Jim class."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.transforms import BoundToUnbound
from jimgw.samplers.config import FlowMCConfig
from tests.utils import assert_all_finite


class MockLikelihood:
    """Simple mock likelihood for testing."""

    def evaluate(self, params):
        return jnp.sum(jnp.array([params[key] for key in params]))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tiny_flowmc_config(pt: Optional[dict] = None, **kwargs) -> FlowMCConfig:
    """Minimal FlowMCConfig for fast unit tests."""
    defaults: dict = dict(
        n_chains=5,
        n_local_steps=2,
        n_global_steps=2,
        global_thinning=1,
        n_training_loops=1,
        n_production_loops=1,
        n_epochs=1,
        parallel_tempering=pt if pt is not None else False,
    )
    defaults.update(kwargs)
    return FlowMCConfig(**defaults)


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gw_prior():
    return CombinePrior(
        [
            UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
            UniformPrior(0.125, 1.0, parameter_names=["q"]),
        ]
    )


@pytest.fixture
def mock_likelihood():
    return MockLikelihood()


@pytest.fixture
def bound_to_unbound_transform():
    return BoundToUnbound(
        name_mapping=(["M_c"], ["M_c_unbounded"]),
        original_lower_bound=10.0,
        original_upper_bound=80.0,
    )


@pytest.fixture
def basic_jim(mock_likelihood, gw_prior):
    return Jim(
        likelihood=mock_likelihood,
        prior=gw_prior,
        sampler_config=_tiny_flowmc_config(),
    )


@pytest.fixture
def jim_with_sample_transforms(mock_likelihood, gw_prior, bound_to_unbound_transform):
    return Jim(
        likelihood=mock_likelihood,
        prior=gw_prior,
        sampler_config=_tiny_flowmc_config(),
        sample_transforms=[bound_to_unbound_transform],
    )


@pytest.fixture
def mass_ratio_to_eta_transform():
    from jimgw.core.single_event.transforms import (
        MassRatioToSymmetricMassRatioTransform,
    )

    return MassRatioToSymmetricMassRatioTransform


@pytest.fixture
def jim_with_likelihood_transforms(
    mock_likelihood, gw_prior, mass_ratio_to_eta_transform
):
    return Jim(
        likelihood=mock_likelihood,
        prior=gw_prior,
        sampler_config=_tiny_flowmc_config(),
        likelihood_transforms=[mass_ratio_to_eta_transform],
    )


@pytest.fixture
def jim_sampler(mock_likelihood, gw_prior, monkeypatch):
    """Jim with get_samples() mocked to avoid actual sampling.

    sampling space = prior space (no sample_transforms), so
    samples is a flat (N, 2) array with [M_c, q] columns.
    """
    jim = Jim(
        likelihood=mock_likelihood,
        prior=gw_prior,
        sampler_config=_tiny_flowmc_config(n_chains=10),
    )

    n_samples = 20
    # Flat arrays in sampling space (= prior space here).
    mock_samples = np.column_stack(
        [np.ones(n_samples) * 30.0, np.ones(n_samples) * 0.5]
    )
    mock_result = {
        "samples": mock_samples,
        "log_likelihood": np.ones(n_samples) * -2.0,
    }

    monkeypatch.setattr(jim.sampler, "get_samples", lambda: mock_result)
    jim.sampler._sampled = True
    return jim


# ---------------------------------------------------------------------------
# TestGetSamples
# ---------------------------------------------------------------------------


class TestGetSamples:
    def test_get_samples_returns_numpy(self, jim_sampler):
        samples = jim_sampler.get_samples()
        assert isinstance(samples, dict)
        for key, val in samples.items():
            assert isinstance(val, np.ndarray), f"Expected numpy.ndarray for key {key}"
            assert not isinstance(val, jax.Array), (
                f"Should not be JAX array for key {key}"
            )

    def test_get_samples_shape(self, jim_sampler):
        samples = jim_sampler.get_samples()
        assert "M_c" in samples and "q" in samples
        assert "log_likelihood" in samples
        assert samples["M_c"].shape == samples["q"].shape
        assert samples["M_c"].ndim == 1

    def test_get_samples_with_downsampling(self, jim_sampler):
        n_samples = 5
        samples = jim_sampler.get_samples(n_samples=n_samples)
        for key, val in samples.items():
            assert val.shape[0] == n_samples, (
                f"Expected {n_samples} samples for key {key}"
            )

    def test_get_samples_deterministic(self, jim_sampler):
        n_samples = 10
        samples1 = jim_sampler.get_samples(n_samples=n_samples)
        samples2 = jim_sampler.get_samples(n_samples=n_samples)
        for key in samples1:
            np.testing.assert_array_equal(samples1[key], samples2[key])

    def test_get_samples_with_sample_transforms(
        self, mock_likelihood, gw_prior, bound_to_unbound_transform, monkeypatch
    ):
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            sampler_config=_tiny_flowmc_config(),
            sample_transforms=[bound_to_unbound_transform],
        )
        # sampling space is ("M_c_unbounded", "q") — use flat array in sampling space.
        n = 10
        # M_c_unbounded=0.0 → M_c = sigmoid(0) * 70 + 10 ≈ 45 (in prior range)
        mock_result = {
            "samples": np.column_stack([np.zeros(n), np.ones(n) * 0.5]),
            "log_likelihood": np.ones(n) * -2.0,
        }
        monkeypatch.setattr(jim.sampler, "get_samples", lambda: mock_result)
        jim.sampler._sampled = True

        samples = jim.get_samples()
        assert "M_c" in samples
        assert "q" in samples
        assert "M_c_unbounded" not in samples
        assert "log_likelihood" in samples
        for val in samples.values():
            assert isinstance(val, np.ndarray)
            assert_all_finite(val)

    def test_get_samples_warning_when_requesting_more_than_available(
        self, jim_sampler, caplog
    ):
        n_available = 20
        with caplog.at_level("WARNING"):
            samples = jim_sampler.get_samples(n_samples=100)
        assert any(
            "Requested 100 samples" in r.message
            and f"only {n_available} available" in r.message
            for r in caplog.records
        )
        assert samples["M_c"].shape[0] == n_available


# ---------------------------------------------------------------------------
# TestJimInitialization
# ---------------------------------------------------------------------------


class TestJimInitialization:
    def test_basic_initialization(self, basic_jim, mock_likelihood, gw_prior):
        assert basic_jim.likelihood == mock_likelihood
        assert basic_jim.prior == gw_prior
        assert len(basic_jim.parameter_names) == 2

    def test_parameter_names_propagation(self, basic_jim):
        assert "M_c" in basic_jim.parameter_names
        assert "q" in basic_jim.parameter_names


# ---------------------------------------------------------------------------
# TestJimWithTransforms
# ---------------------------------------------------------------------------


class TestJimWithTransforms:
    def test_sample_transforms(self, jim_with_sample_transforms):
        assert "M_c_unbounded" in jim_with_sample_transforms.parameter_names
        assert "q" in jim_with_sample_transforms.parameter_names

    def test_likelihood_transforms(self, jim_with_likelihood_transforms):
        assert jim_with_likelihood_transforms.likelihood_transforms is not None
        assert len(jim_with_likelihood_transforms.likelihood_transforms) == 1


# ---------------------------------------------------------------------------
# TestJimTempering
# ---------------------------------------------------------------------------


class TestJimTempering:
    def test_with_tempering_enabled(self, mock_likelihood, gw_prior):
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            sampler_config=_tiny_flowmc_config(
                parallel_tempering=True,
            ),
        )
        assert "parallel_tempering" in jim.sampler.strategy_order  # type: ignore[attr-defined]

    def test_with_tempering_disabled(self, mock_likelihood, gw_prior):
        jim = Jim(
            likelihood=mock_likelihood,
            prior=gw_prior,
            sampler_config=_tiny_flowmc_config(parallel_tempering=False),
        )
        assert "parallel_tempering" not in jim.sampler.strategy_order  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TestJimPosteriorEvaluation
# ---------------------------------------------------------------------------


class TestJimPosteriorEvaluation:
    def test_evaluate_posterior_valid_sample(self, basic_jim):
        samples_valid = jnp.array([30.0, 0.5])
        log_posterior = basic_jim.evaluate_posterior(samples_valid)
        assert jnp.isfinite(log_posterior)

    def test_evaluate_posterior_invalid_sample(self, basic_jim):
        samples_invalid = jnp.array([100.0, 0.5])
        log_posterior = basic_jim.evaluate_posterior(samples_invalid)
        assert log_posterior == -jnp.inf

    def test_evaluate_posterior_with_likelihood_transforms(
        self, gw_prior, mass_ratio_to_eta_transform
    ):
        class EtaLikelihood:
            def evaluate(self, params):
                assert "eta" in params and "M_c" in params
                return params["M_c"] + params["eta"]

        jim = Jim(
            likelihood=EtaLikelihood(),
            prior=gw_prior,
            sampler_config=_tiny_flowmc_config(),
            likelihood_transforms=[mass_ratio_to_eta_transform],
        )
        assert jnp.isfinite(jim.evaluate_posterior(jnp.array([30.0, 0.5])))

    def test_evaluate_posterior_with_sample_transforms(
        self, jim_with_sample_transforms
    ):
        assert "M_c_unbounded" in jim_with_sample_transforms.parameter_names
        samples_transformed = jnp.array([0.5, 0.6])
        assert jnp.isfinite(
            jim_with_sample_transforms.evaluate_posterior(samples_transformed)
        )


# ---------------------------------------------------------------------------
# TestJimUtilityMethods
# ---------------------------------------------------------------------------


class TestJimUtilityMethods:
    def test_add_name(self, basic_jim):
        params_array = jnp.array([30.0, 0.5])
        params_dict = basic_jim.add_name(params_array)
        assert isinstance(params_dict, dict)
        assert params_dict["M_c"] == 30.0
        assert params_dict["q"] == 0.5

    def test_evaluate_prior(self, basic_jim):
        assert jnp.isfinite(basic_jim.evaluate_prior(jnp.array([30.0, 0.5])))

    def test_evaluate_prior_with_sample_transforms(self, jim_with_sample_transforms):
        log_prior = jim_with_sample_transforms.evaluate_prior(jnp.array([0.5, 0.6]))
        assert jnp.isfinite(log_prior)

    def test_sample_initial_positions(self, basic_jim):
        initial_samples = basic_jim.sample_initial_positions(5)
        assert initial_samples.shape == (5, 2)
        assert_all_finite(initial_samples)

    def test_sample_initial_positions_with_n_points(self, basic_jim):
        initial_samples = basic_jim.sample_initial_positions(7)
        assert initial_samples.shape == (7, 2)
        assert_all_finite(initial_samples)

    def test_sample_initial_positions_with_sample_transforms(
        self, jim_with_sample_transforms
    ):
        initial_samples = jim_with_sample_transforms.sample_initial_positions(5)
        assert initial_samples.shape == (5, 2)
        assert_all_finite(initial_samples)

    def test_sample_initial_positions_raises_on_non_finite(self, mock_likelihood):
        class BadPrior:
            n_dims = 2
            parameter_names = ("param1", "param2")

            def sample(self, rng_key, n_samples):
                return {
                    "param1": jnp.full(n_samples, jnp.nan),
                    "param2": jnp.full(n_samples, jnp.nan),
                }

            def log_prob(self, params):
                return 0.0

        with pytest.raises(ValueError, match="non-finite"):
            Jim(
                likelihood=mock_likelihood,
                prior=BadPrior(),
                sampler_config=_tiny_flowmc_config(),
            )


# ---------------------------------------------------------------------------
# TestJimSampleMethod — shape validation lives in FlowMCSampler.sample()
# ---------------------------------------------------------------------------


class TestJimSampleMethod:
    def test_sample_raises_on_wrong_1d_shape(self, basic_jim):
        """Wrong 1D shape (3 instead of n_dims=2) raises ValueError."""
        with pytest.raises(ValueError, match="initial_position must have shape"):
            basic_jim.sample(initial_position=jnp.array([30.0, 0.5, 0.8]))

    def test_sample_raises_on_wrong_2d_shape(self, basic_jim):
        """Wrong 2D shape (3 chains instead of n_chains=5) raises ValueError."""
        with pytest.raises(ValueError, match="initial_position must have shape"):
            basic_jim.sample(initial_position=jnp.ones((3, 2)))

    def test_sample_raises_on_3d_initial_position(self, basic_jim):
        """3D initial position raises ValueError."""
        with pytest.raises(ValueError, match="initial_position must have shape"):
            basic_jim.sample(initial_position=jnp.ones((5, 2, 3)))


# ---------------------------------------------------------------------------
# TestJimPriorLikelihoodConsistencyChecks
# ---------------------------------------------------------------------------


class TestJimPriorLikelihoodConsistencyChecks:
    def _make_mock_single_event_likelihood(
        self,
        waveform_parameter_names: tuple[str, ...],
        fixed_parameters: Optional[dict] = None,
    ):
        from jimgw.core.single_event.likelihood import SingleEventLikelihood
        from ripplegw.interfaces import Waveform as RippleWaveform
        from jaxtyping import Float

        class MockWaveform(RippleWaveform):
            def __init__(self, param_names):
                self._param_names = param_names

            @property
            def parameter_names(self) -> tuple[str, ...]:
                return self._param_names

            def __call__(self, axis, params):
                return {"p": axis, "c": axis}

        class FakeSingleEventLikelihood(SingleEventLikelihood):
            def __init__(self, waveform, fixed_parameters):
                self.waveform = waveform
                self.fixed_parameters = fixed_parameters or {}

            def _likelihood(self, params) -> Float:
                return 0.0

        return FakeSingleEventLikelihood(
            waveform=MockWaveform(waveform_parameter_names),
            fixed_parameters=fixed_parameters or {},
        )

    def test_prior_shadows_fixed_parameter_raises(self):
        prior = CombinePrior(
            [
                UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
            ]
        )
        lh = self._make_mock_single_event_likelihood(
            waveform_parameter_names=("M_c", "ra", "dec", "psi", "t_c"),
            fixed_parameters={"M_c": 30.0},
        )
        with pytest.raises(ValueError, match="also in fixed_parameters"):
            Jim(likelihood=lh, prior=prior, sampler_config=_tiny_flowmc_config())

    def test_prior_with_unused_parameter_raises(self):
        prior = CombinePrior(
            [
                UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
                UniformPrior(0.125, 1.0, parameter_names=["q"]),
            ]
        )
        lh = self._make_mock_single_event_likelihood(
            waveform_parameter_names=("M_c", "ra", "dec", "psi", "t_c"),
        )
        with pytest.raises(ValueError, match="not consumed by the likelihood"):
            Jim(likelihood=lh, prior=prior, sampler_config=_tiny_flowmc_config())

    def test_likelihood_requires_missing_parameter_raises(self):
        prior = CombinePrior(
            [
                UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
                UniformPrior(0.0, 3.14, parameter_names=["ra"]),
                UniformPrior(-1.57, 1.57, parameter_names=["dec"]),
                UniformPrior(0.0, 3.14, parameter_names=["psi"]),
                # t_c intentionally omitted
            ]
        )
        lh = self._make_mock_single_event_likelihood(
            waveform_parameter_names=("M_c", "ra", "dec", "psi", "t_c"),
        )
        with pytest.raises(
            ValueError, match="not provided by the prior or fixed_parameters"
        ):
            Jim(likelihood=lh, prior=prior, sampler_config=_tiny_flowmc_config())

    def test_missing_parameter_covered_by_fixed_no_error(self):
        prior = CombinePrior(
            [
                UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
                UniformPrior(0.0, 3.14, parameter_names=["ra"]),
                UniformPrior(-1.57, 1.57, parameter_names=["dec"]),
                UniformPrior(0.0, 3.14, parameter_names=["psi"]),
                # t_c provided via fixed_parameters instead
            ]
        )
        lh = self._make_mock_single_event_likelihood(
            waveform_parameter_names=("M_c", "ra", "dec", "psi", "t_c"),
            fixed_parameters={"t_c": 0.0},
        )
        Jim(likelihood=lh, prior=prior, sampler_config=_tiny_flowmc_config())

    def test_prior_all_consumed_no_error(self):
        prior = CombinePrior(
            [
                UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
                UniformPrior(0.0, 3.14, parameter_names=["ra"]),
                UniformPrior(-1.57, 1.57, parameter_names=["dec"]),
                UniformPrior(0.0, 3.14, parameter_names=["psi"]),
                UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
            ]
        )
        lh = self._make_mock_single_event_likelihood(
            waveform_parameter_names=("M_c", "ra", "dec", "psi", "t_c"),
        )
        Jim(likelihood=lh, prior=prior, sampler_config=_tiny_flowmc_config())

    def test_sample_transform_overwrites_unconsumed_prior_parameter_raises(self):
        # Prior defines both M_c and M_c_unbounded; sample transform maps
        # M_c → M_c_unbounded without consuming M_c_unbounded — silent overwrite.
        prior = CombinePrior(
            [
                UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
                UniformPrior(0.0, 1.0, parameter_names=["M_c_unbounded"]),  # conflict
                UniformPrior(0.0, 3.14, parameter_names=["ra"]),
                UniformPrior(-1.57, 1.57, parameter_names=["dec"]),
                UniformPrior(0.0, 3.14, parameter_names=["psi"]),
                UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
            ]
        )
        lh = self._make_mock_single_event_likelihood(
            waveform_parameter_names=(
                "M_c",
                "M_c_unbounded",
                "ra",
                "dec",
                "psi",
                "t_c",
            ),
        )
        sample_transform = BoundToUnbound(
            name_mapping=(["M_c"], ["M_c_unbounded"]),
            original_lower_bound=10.0,
            original_upper_bound=80.0,
        )
        with pytest.raises(ValueError, match="already exist in the parameter space"):
            Jim(
                likelihood=lh,
                prior=prior,
                sampler_config=_tiny_flowmc_config(),
                sample_transforms=[sample_transform],
            )

    def test_sample_transform_valid_rename_no_error(self):
        # Prior defines only M_c; sample transform maps M_c → M_c_unbounded. No conflict.
        prior = CombinePrior(
            [
                UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
                UniformPrior(0.0, 3.14, parameter_names=["ra"]),
                UniformPrior(-1.57, 1.57, parameter_names=["dec"]),
                UniformPrior(0.0, 3.14, parameter_names=["psi"]),
                UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
            ]
        )
        lh = self._make_mock_single_event_likelihood(
            waveform_parameter_names=("M_c", "ra", "dec", "psi", "t_c"),
        )
        sample_transform = BoundToUnbound(
            name_mapping=(["M_c"], ["M_c_unbounded"]),
            original_lower_bound=10.0,
            original_upper_bound=80.0,
        )
        Jim(
            likelihood=lh,
            prior=prior,
            sampler_config=_tiny_flowmc_config(),
            sample_transforms=[sample_transform],
        )

    def test_likelihood_transform_overwrites_unconsumed_prior_parameter_raises(self):
        # Prior defines both q and eta; transform maps q → eta without consuming eta.
        # The prior-sampled eta would be silently overwritten — this must be caught.
        from jimgw.core.single_event.transforms import (
            MassRatioToSymmetricMassRatioTransform,
        )

        prior = CombinePrior(
            [
                UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
                UniformPrior(0.125, 1.0, parameter_names=["q"]),
                UniformPrior(0.1, 0.25, parameter_names=["eta"]),
                UniformPrior(0.0, 3.14, parameter_names=["ra"]),
                UniformPrior(-1.57, 1.57, parameter_names=["dec"]),
                UniformPrior(0.0, 3.14, parameter_names=["psi"]),
                UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
            ]
        )
        lh = self._make_mock_single_event_likelihood(
            waveform_parameter_names=("M_c", "eta", "ra", "dec", "psi", "t_c"),
        )
        with pytest.raises(ValueError, match="already exist in the parameter space"):
            Jim(
                likelihood=lh,
                prior=prior,
                sampler_config=_tiny_flowmc_config(),
                likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
            )

    def test_likelihood_transform_valid_rename_no_error(self):
        # Prior defines only q (not eta); transform maps q → eta. No conflict.
        from jimgw.core.single_event.transforms import (
            MassRatioToSymmetricMassRatioTransform,
        )

        prior = CombinePrior(
            [
                UniformPrior(10.0, 80.0, parameter_names=["M_c"]),
                UniformPrior(0.125, 1.0, parameter_names=["q"]),
                UniformPrior(0.0, 3.14, parameter_names=["ra"]),
                UniformPrior(-1.57, 1.57, parameter_names=["dec"]),
                UniformPrior(0.0, 3.14, parameter_names=["psi"]),
                UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
            ]
        )
        lh = self._make_mock_single_event_likelihood(
            waveform_parameter_names=("M_c", "eta", "ra", "dec", "psi", "t_c"),
        )
        Jim(
            likelihood=lh,
            prior=prior,
            sampler_config=_tiny_flowmc_config(),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
        )


# ---------------------------------------------------------------------------
# TestJimNaNPosteriorCheck
# ---------------------------------------------------------------------------


class TestJimNaNPosteriorCheck:
    def test_all_nan_posterior_raises(self):
        class NaNLikelihood:
            def evaluate(self, params):
                return jnp.nan

        prior = CombinePrior([UniformPrior(10.0, 80.0, parameter_names=["M_c"])])
        with pytest.raises(ValueError, match="posterior returned NaN"):
            Jim(
                likelihood=NaNLikelihood(),
                prior=prior,
                sampler_config=_tiny_flowmc_config(),
            )

    def test_some_nan_posterior_warns(self, caplog, monkeypatch):
        class SometimesNaNLikelihood:
            def evaluate(self, params):
                return jnp.where(params["M_c"] > 70.0, jnp.nan, -1.0)

        prior = CombinePrior([UniformPrior(10.0, 80.0, parameter_names=["M_c"])])

        # Provide fixed test positions (shape n_points x n_dims) so the NaN
        # count is deterministic: 2 out of 10 have M_c > 70.
        _fixed_positions = jnp.array(
            [
                [20.0],
                [30.0],
                [40.0],
                [45.0],
                [50.0],
                [55.0],
                [60.0],
                [65.0],
                [75.0],
                [78.0],
            ]
        )
        monkeypatch.setattr(
            Jim,
            "sample_initial_positions",
            lambda self, n_points=None, rng_key=None: _fixed_positions,
        )

        with caplog.at_level("WARNING"):
            Jim(
                likelihood=SometimesNaNLikelihood(),
                prior=prior,
                sampler_config=_tiny_flowmc_config(),
            )
        assert any("NaN" in r.message for r in caplog.records)

    def test_no_nan_posterior_no_error(self):
        class FiniteLikelihood:
            def evaluate(self, params):
                return -1.0

        prior = CombinePrior([UniformPrior(10.0, 80.0, parameter_names=["M_c"])])
        Jim(
            likelihood=FiniteLikelihood(),
            prior=prior,
            sampler_config=_tiny_flowmc_config(),
        )


# ---------------------------------------------------------------------------
# TestJimPeriodic
# ---------------------------------------------------------------------------


class TestJimPeriodic:
    """Tests for the ``periodic`` argument of Jim.__init__."""

    def _make_prior(self):
        return CombinePrior(
            [
                UniformPrior(0.0, 1.0, parameter_names=["x"]),
                UniformPrior(0.0, 6.2832, parameter_names=["phase"]),
            ]
        )

    def _make_likelihood(self):
        return MockLikelihood()

    def test_periodic_none_default_constructs(self):
        """Jim(periodic=None) should construct without error."""
        Jim(
            likelihood=self._make_likelihood(),
            prior=self._make_prior(),
            sampler_config=_tiny_flowmc_config(),
        )

    def test_periodic_dict_valid_names_constructs(self):
        """Jim(periodic=dict) resolves known names to indices."""
        jim = Jim(
            likelihood=self._make_likelihood(),
            prior=self._make_prior(),
            sampler_config=_tiny_flowmc_config(),
            periodic={"phase": (0.0, 6.2832)},
        )
        # FlowMCSampler stores the index-keyed dict; phase is index 1.
        assert jim.sampler._periodic_index_dict == {1: (0.0, 6.2832)}

    def test_periodic_list_raises_for_non_nsaw(self):
        """Jim(periodic=list) with a FlowMCSampler raises because list-form has no bounds."""
        with pytest.raises(ValueError, match="List-form periodic"):
            Jim(
                likelihood=self._make_likelihood(),
                prior=self._make_prior(),
                sampler_config=_tiny_flowmc_config(),
                periodic=["phase"],
            )

    def test_periodic_dict_unknown_name_raises(self):
        """Jim(periodic=dict) with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="not found in sampling parameters"):
            Jim(
                likelihood=self._make_likelihood(),
                prior=self._make_prior(),
                sampler_config=_tiny_flowmc_config(),
                periodic={"nonexistent": (0.0, 1.0)},
            )

    def test_periodic_list_unknown_name_raises(self):
        """Jim(periodic=list) with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="not found in sampling parameters"):
            Jim(
                likelihood=self._make_likelihood(),
                prior=self._make_prior(),
                sampler_config=_tiny_flowmc_config(),
                periodic=["nonexistent"],
            )

    def test_periodic_empty_list_constructs(self):
        """Explicit empty list for periodic should construct without error."""
        Jim(
            likelihood=self._make_likelihood(),
            prior=self._make_prior(),
            sampler_config=_tiny_flowmc_config(),
            periodic=[],
        )

    def test_periodic_empty_dict_constructs(self):
        """Explicit empty dict for periodic should construct without error."""
        Jim(
            likelihood=self._make_likelihood(),
            prior=self._make_prior(),
            sampler_config=_tiny_flowmc_config(),
            periodic={},
        )
