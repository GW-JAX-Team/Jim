import jax
import jax.numpy as jnp
import pytest
from pathlib import Path
from jimgw.core.single_event.likelihood import (
    ZeroLikelihood,
    TransientLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
)
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.prior import CombinePrior, PowerLawPrior, UniformPrior
from tests.utils import assert_all_finite, common_keys_allclose

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"


@pytest.fixture
def detectors_and_waveform():
    gps = 1126259462.4
    fmin = 20.0
    fmax = 1024.0
    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        data = Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
        ifo.set_data(data)
        psd = PowerSpectrum.from_file(
            str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
        )
        ifo.set_psd(psd)
    waveform = RippleIMRPhenomD(f_ref=20.0)
    return ifos, waveform, fmin, fmax, gps


def example_params():
    return {
        "M_c": 30.0,
        "eta": 0.249,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "d_L": 400.0,
        "phase_c": 0.0,
        "t_c": 0.0,
        "iota": 0.0,
        "ra": 1.375,
        "dec": -1.2108,
        "psi": 0.0,
    }


class TestZeroLikelihood:
    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = ZeroLikelihood()
        assert isinstance(likelihood, ZeroLikelihood)
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert result == 0.0


class TestTransientLikelihoodFD:
    # ------------------------------------------------------------------
    # Happy-path initialisation tests
    # ------------------------------------------------------------------

    def test_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        assert isinstance(likelihood, TransientLikelihoodFD)
        assert likelihood.frequencies[0] == fmin
        assert likelihood.frequencies[-1] == fmax
        assert likelihood.trigger_time == 1126259462.4
        assert hasattr(likelihood, "gmst")

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_uninitialized_data_raises_error(self):
        """Test that initializing likelihood with detectors that have no data raises an error."""
        gps = 1126259462.4

        # Create detectors with PSD but without data
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            psd = PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
            )
            ifo.set_psd(psd)
            # Intentionally not setting data

        waveform = RippleIMRPhenomD(f_ref=20.0)

        # Should raise ValueError when trying to initialize likelihood
        with pytest.raises(ValueError, match="does not have initialized data"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
            )

    def test_partially_initialized_data_raises_error(self, detectors_and_waveform):
        """Test that having only some detectors with data raises an error."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        # Add a detector with PSD but no data
        new_detector = get_H1()
        psd = PowerSpectrum.from_file(
            str(FIXTURES_DIR / f"GW150914_psd_{new_detector.name}.npz")
        )
        new_detector.set_psd(psd)
        # Intentionally not setting data for this detector

        ifos_mixed = ifos + [new_detector]

        # Should raise ValueError mentioning the detector name
        with pytest.raises(ValueError, match="H1.*does not have initialized data"):
            TransientLikelihoodFD(
                detectors=ifos_mixed,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    def test_uninitialized_psd_raises_error(self):
        """Test that initializing likelihood with detectors that have no PSD raises an error."""
        gps = 1126259462.4

        # Create detectors with data but no PSD
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            data = Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
            ifo.set_data(data)
            # Intentionally not setting PSD

        waveform = RippleIMRPhenomD(f_ref=20.0)

        # Should raise ValueError when trying to initialize likelihood
        with pytest.raises(ValueError, match="does not have initialized PSD"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
            )

    def test_partially_initialized_psd_raises_error(self, detectors_and_waveform):
        """Test that having only some detectors with PSD raises an error."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        # Add a detector with data but no PSD
        new_detector = get_H1()
        data = Data.from_file(
            str(FIXTURES_DIR / f"GW150914_strain_{new_detector.name}.npz")
        )
        new_detector.set_data(data)
        # Intentionally not setting PSD for this detector

        ifos_mixed = ifos + [new_detector]

        # Should raise ValueError mentioning the detector name and PSD
        with pytest.raises(ValueError, match="H1.*does not have initialized PSD"):
            TransientLikelihoodFD(
                detectors=ifos_mixed,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params()

        log_likelihood = likelihood.evaluate(params, {})
        assert jnp.isfinite(log_likelihood), "Log likelihood should be finite"

        log_likelihood_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.isfinite(log_likelihood_jit), "Log likelihood should be finite"

        assert jnp.allclose(
            log_likelihood,
            log_likelihood_jit,
        ), "JIT and non-JIT results should match"

        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
        )
        log_likelihood_diff_fmin = likelihood.evaluate(params, {})
        assert jnp.isfinite(log_likelihood_diff_fmin), (
            "Log likelihood with different f_min should be finite"
        )

        assert jnp.allclose(
            log_likelihood,
            log_likelihood_diff_fmin,
            atol=1e-2,
        ), "Log likelihoods should be close with small differences"


class TestHeterodynedTransientLikelihoodFD:
    # ------------------------------------------------------------------
    # Happy-path initialisation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        # First create base likelihood for comparison
        base_likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )

        # Create heterodyned likelihood with reference parameters
        ref_params = example_params()
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)

        # Test evaluation at reference parameters
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), "Heterodyned likelihood should be finite"

        # Test that heterodyned likelihood matches base likelihood at reference parameters
        base_result = base_likelihood.evaluate(params, {})
        assert jnp.allclose(
            result,
            base_result,
        ), (
            f"Heterodyned likelihood ({result}) should match base likelihood ({base_result}) at reference parameters"
        )

    def test_initialization_stores_attributes(self, detectors_and_waveform):
        """Coefficient arrays and grid arrays are populated after init."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        assert hasattr(likelihood, "freq_grid_low")
        assert hasattr(likelihood, "freq_grid_center")
        for det in ifos:
            assert det.name in likelihood.A0_array
            assert det.name in likelihood.A1_array
            assert det.name in likelihood.B0_array
            assert det.name in likelihood.B1_array
            assert det.name in likelihood.waveform_low_ref
            assert det.name in likelihood.waveform_center_ref

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_no_reference_params_and_no_prior_raises(self, detectors_and_waveform):
        """Omitting both reference_parameters and prior must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Heterodyned likelihood must accept per-detector f_min and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Heterodyned likelihood should be finite with different f_min"
        )

    # ------------------------------------------------------------------
    # maximize_likelihood tests
    # ------------------------------------------------------------------

    def test_maximize_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_params = example_params()

        # Inject zero-noise signal
        for ifo in ifos:
            ifo.inject_signal(
                duration=4.0,
                sampling_frequency=fmax * 2,
                trigger_time=gps,
                waveform_model=waveform,
                parameters=true_params,
                f_min=fmin,
                f_max=fmax,
                zero_noise=True,
            )

        base_likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )

        # Reference: MF log-likelihood at injected params.
        ll_injected = float(base_likelihood.evaluate(true_params, {}))

        # Fix all nuisance parameters to truth; DE searches only (M_c, q).
        fixed_parameters = {
            k: v for k, v in true_params.items() if k not in ("M_c", "eta")
        }

        prior = CombinePrior(
            [
                UniformPrior(25.0, 35.0, parameter_names=["M_c"]),
                UniformPrior(0.125, 1.0, parameter_names=["q"]),
            ]
        )
        likelihood_transforms = [MassRatioToSymmetricMassRatioTransform]

        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters=fixed_parameters,
            prior=prior,
            likelihood_transforms=likelihood_transforms,
            optimizer_popsize=10,
            optimizer_n_steps=50,
        )

        result = likelihood.reference_parameters.copy()

        # 1. Result must contain all expected keys (q → eta via transform).
        #    reference_parameters also includes trigger_time and gmst injected by __init__.
        expected_keys = {
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            "d_L",
            "t_c",
            "phase_c",
            "iota",
            "psi",
            "ra",
            "dec",
            "trigger_time",
            "gmst",
        }
        assert set(result.keys()) == expected_keys, (
            f"Unexpected keys: got {set(result.keys())}, expected {expected_keys}"
        )
        # 2. All returned values must be finite.
        for key, val in result.items():
            assert jnp.isfinite(val), (
                f"maximize_likelihood returned non-finite value for '{key}': {val}"
            )
        # 3. The heterodyned likelihood must be finite at the result.
        assert jnp.isfinite(likelihood.evaluate(result, {})), (
            "Heterodyned likelihood at maximized parameters must be finite"
        )
        # 4. The MF log-likelihood at the result should be close to the injected value
        result_ll = float(base_likelihood.evaluate(result, {}))
        assert jnp.isclose(result_ll, ll_injected), (
            f"Log-likelihood at maximize_likelihood result ({result_ll:.2f}) should be close to injected value ({ll_injected:.2f})"
        )
        # 5. The result should be close to the injected parameters
        common_keys_allclose(result, true_params)


class TestTimeMarginalizedTransientLikelihoodFD:
    """Tests for TransientLikelihoodFD with time_marginalization."""

    # ------------------------------------------------------------------
    # Happy-path initialisation tests
    # ------------------------------------------------------------------

    def test_initialization(self, detectors_and_waveform):
        """tc_range, tc_array, pad_low and pad_high are stored after init."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        assert isinstance(likelihood, TransientLikelihoodFD)
        assert hasattr(likelihood, "tc_range")
        assert hasattr(likelihood, "tc_array")
        assert hasattr(likelihood, "pad_low")
        assert hasattr(likelihood, "pad_high")
        assert likelihood.tc_range == (-0.1, 0.1)

    def test_custom_tc_range(self, detectors_and_waveform):
        """A custom tc_range is stored and reflected in the likelihood."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        custom_range = (-0.05, 0.05)
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={"tc_range": custom_range},
        )
        assert likelihood.tc_range == custom_range

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_fixed_t_c_raises(self, detectors_and_waveform):
        """Passing t_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have t_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={},
                fixed_parameters={"t_c": 0.0},
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        assert isinstance(likelihood, TransientLikelihoodFD)
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), "Time-marginalized likelihood should be finite"

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min must be accepted and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Time-marginalized likelihood should be finite with different f_min"
        )

    def test_geq_base_likelihood(self, detectors_and_waveform):
        """Time-marginalized likelihood must be >= the base likelihood.

        The time marginalization uses logsumexp over an FFT, which is
        always >= the value at any single t_c (including the t_c=0 used
        by the base likelihood).  In other words, the marginalized
        likelihood can find a better coalescence time within the range.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        marg_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        base_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )

        params_marg = example_params()
        params_base = example_params()

        marg_result = marg_likelihood.evaluate(params_marg, {})
        base_result = base_likelihood.evaluate(params_base, {})

        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result, (
            f"Time-marginalized ({marg_result:.4f}) should be >= base "
            f"({base_result:.4f}) because logsumexp >= any single element"
        )


class TestPhaseMarginalizedTransientLikelihoodFD:
    """Tests for TransientLikelihoodFD with phase_marginalization."""

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_fixed_phase_c_raises(self, detectors_and_waveform):
        """Passing phase_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), "Phase-marginalized likelihood should be finite"

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min must be accepted and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Phase-marginalized likelihood should be finite with different f_min"
        )

    def test_geq_base_likelihood(self, detectors_and_waveform):
        """Phase-marginalized likelihood must be >= the base likelihood.

        The phase marginalization replaces Re(<d|h>) with log I_0(|<d|h>|).
        Since I_0(x) >= 1 for all x >= 0, the marginalized value is always
        at least as large as the base value at any fixed phase.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        marg_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        base_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )

        params_marg = example_params()
        params_base = example_params()

        marg_result = marg_likelihood.evaluate(params_marg, {})
        base_result = base_likelihood.evaluate(params_base, {})

        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result, (
            f"Phase-marginalized ({marg_result:.4f}) should be >= base "
            f"({base_result:.4f}) because I_0(x) >= 1"
        )


class TestPhaseTimeMarginalizedTransientLikelihoodFD:
    """Tests for TransientLikelihoodFD with time_marginalization and phase_marginalization."""

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_fixed_phase_c_raises(self, detectors_and_waveform):
        """Passing phase_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={},
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_fixed_t_c_raises(self, detectors_and_waveform):
        """Passing t_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have t_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={},
                phase_marginalization=True,
                fixed_parameters={"t_c": 0.0},
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Phase-time-marginalized likelihood should be finite"
        )

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min must be accepted and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Phase-time-marginalized likelihood should be finite with different f_min"
        )

    def test_geq_base_likelihood(self, detectors_and_waveform):
        """Phase-time-marginalized likelihood must be >= the base likelihood.

        Marginalizing over both phase (via I_0) and time (via logsumexp) can
        only increase or preserve the log-likelihood relative to a single
        point evaluation.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        marg_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        base_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )

        params_marg = example_params()
        params_base = example_params()

        marg_result = marg_likelihood.evaluate(params_marg, {})
        base_result = base_likelihood.evaluate(params_base, {})

        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result, (
            f"Phase-time-marginalized ({marg_result:.4f}) should be >= base "
            f"({base_result:.4f})"
        )

    def test_geq_phase_only_marginalized(self, detectors_and_waveform):
        """Phase-time-marginalized must be >= phase-only-marginalized.

        The phase-only-marginalized evaluates at a single t_c=0, while
        the phase-time-marginalized searches over a range of t_c values
        via logsumexp, so it can only be equal or larger.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        phase_time_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        phase_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )

        params = example_params()

        pt_result = phase_time_likelihood.evaluate(params, {})
        p_result = phase_likelihood.evaluate(params, {})

        assert jnp.isfinite(pt_result)
        assert pt_result >= p_result, (
            f"Phase-time-marginalized ({pt_result:.4f}) should be >= "
            f"phase-only-marginalized ({p_result:.4f})"
        )


class TestHeterodynedPhaseMarginalizedTransientLikelihoodFD:
    """Tests for HeterodynedTransientLikelihoodFD with phase_marginalization."""

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_fixed_phase_c_raises(self, detectors_and_waveform):
        """Passing phase_c in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_parameters=ref_params,
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_no_reference_params_and_no_prior_raises(self, detectors_and_waveform):
        """Omitting both reference_parameters and prior must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                phase_marginalization=True,
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
            phase_marginalization=True,
        )
        assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Heterodyned phase-marginalized likelihood should be finite"
        )

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
            phase_marginalization=True,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), "JIT and eager results should match"

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min must be accepted and produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
            phase_marginalization=True,
        )
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Heterodyned phase-marginalized likelihood should be finite with different f_min"
        )

    def test_matches_phase_marginalized_at_ref_params(self, detectors_and_waveform):
        """At the reference parameters the heterodyned phase-marginalized
        likelihood should closely match the non-heterodyned phase-marginalized
        likelihood, since the heterodyne approximation is exact at the
        reference point.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        phase_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        ref_params = example_params()

        het_phase_likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
            phase_marginalization=True,
        )

        params = example_params()

        het_result = het_phase_likelihood.evaluate(params, {})
        phase_result = phase_likelihood.evaluate(params, {})

        assert jnp.isfinite(het_result)
        assert jnp.allclose(het_result, phase_result, atol=1e-1), (
            f"Heterodyned phase-marg ({het_result:.4f}) should match "
            f"phase-marg ({phase_result:.4f}) at reference parameters"
        )


class TestDistanceMarginalizedTransientLikelihoodFD:
    """Tests for TransientLikelihoodFD with distance_marginalization."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_d_L_prior(xmin: float = 100.0, xmax: float = 5000.0) -> PowerLawPrior:
        """Convenience factory for a d^2 (power-law alpha=2) distance prior."""
        return PowerLawPrior(xmin=xmin, xmax=xmax, alpha=2.0, parameter_names=["d_L"])

    @staticmethod
    def params_without_d_L() -> dict:
        """Parameter dict with d_L omitted (the likelihood injects its own value)."""
        return {
            "M_c": 30.0,
            "eta": 0.249,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "phase_c": 0.0,
            "t_c": 0.0,
            "iota": 0.0,
            "ra": 1.375,
            "dec": -1.2108,
            "psi": 0.0,
        }

    # ------------------------------------------------------------------
    # Validation tests (no waveform evaluation needed)
    # ------------------------------------------------------------------

    def test_init_no_distance_prior_raises(self):
        """Constructing DistanceMargConfig() without distance_prior must raise ValidationError."""
        from pydantic import ValidationError
        from jimgw.core.single_event.likelihood import DistanceMargConfig

        with pytest.raises(ValidationError):
            DistanceMargConfig()  # type: ignore[call-arg]

    def test_init_fixed_d_L_raises(self, detectors_and_waveform):
        """Passing d_L in fixed_parameters must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have d_L fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                distance_marginalization={"distance_prior": self.make_d_L_prior()},
                fixed_parameters={"d_L": 400.0},
            )

    def test_init_prior_missing_d_L_raises(self, detectors_and_waveform):
        """A prior that contains no d_L sub-prior must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        prior_no_d_L = UniformPrior(xmin=10.0, xmax=100.0, parameter_names=["M_c"])
        with pytest.raises(
            ValueError, match="must be a 1D prior with parameter_names="
        ):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                distance_marginalization={"distance_prior": prior_no_d_L},
            )

    def test_init_n_dist_points_too_small_raises(self, detectors_and_waveform):
        """n_dist_points < 2 must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="n_dist_points must be at least 2"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                distance_marginalization={
                    "distance_prior": self.make_d_L_prior(),
                    "n_dist_points": 1,
                },
            )

    def test_init_negative_ref_dist_raises(self, detectors_and_waveform):
        """ref_dist <= 0 must raise ValueError."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="ref_dist must be > 0"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                distance_marginalization={
                    "distance_prior": self.make_d_L_prior(),
                    "ref_dist": -100.0,
                },
            )

    # ------------------------------------------------------------------
    # Happy-path initialisation tests
    # ------------------------------------------------------------------

    def test_init_single_d_L_prior(self, detectors_and_waveform):
        """A PowerLawPrior directly for d_L must be accepted."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert isinstance(likelihood, TransientLikelihoodFD)

    def test_init_uniform_prior(self, detectors_and_waveform):
        """A UniformPrior for d_L must also be accepted."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        uniform_d_L = UniformPrior(xmin=100.0, xmax=5000.0, parameter_names=["d_L"])
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": uniform_d_L},
        )
        assert isinstance(likelihood, TransientLikelihoodFD)

    def test_default_ref_dist(self, detectors_and_waveform):
        """When ref_dist is None the default is the midpoint of [xmin, xmax]."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        d_L_prior = self.make_d_L_prior(xmin=200.0, xmax=1000.0)
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": d_L_prior},
        )
        assert jnp.isclose(likelihood.ref_dist, (200.0 + 1000.0) / 2.0)

    def test_custom_ref_dist(self, detectors_and_waveform):
        """An explicit ref_dist is stored unchanged."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": self.make_d_L_prior(),
                "ref_dist": 500.0,
            },
        )
        assert jnp.isclose(likelihood.ref_dist, 500.0)

    def test_log_weights_normalized(self, detectors_and_waveform):
        """log_weights must sum to 1 in probability space, i.e. logsumexp ≈ 0."""
        from jax.scipy.special import logsumexp

        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert jnp.isclose(logsumexp(likelihood.log_weights), 0.0, atol=1e-5)

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_evaluate_is_finite(self, detectors_and_waveform):
        """evaluate() must return a finite scalar."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), f"Expected finite log-likelihood, got {result}"

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        """jax.jit(evaluate) must agree with the eager result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L()
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), (
            f"JIT result {result_jit} does not match eager result {result}"
        )

    def test_matches_base_likelihood_near_true_distance(self, detectors_and_waveform):
        """With a very narrow prior tightly centered on the true d_L, the
        marginalized value should be close to the base (non-marginalized)
        likelihood evaluated at that distance.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_d_L = 400.0

        # Narrow uniform window around the true distance
        narrow_prior = UniformPrior(
            xmin=true_d_L - 1.0, xmax=true_d_L + 1.0, parameter_names=["d_L"]
        )
        marg_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": narrow_prior,
                "ref_dist": true_d_L,
            },
        )

        base_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )

        params_marg = self.params_without_d_L()
        params_base = example_params()  # includes d_L=400

        marg_result = marg_likelihood.evaluate(params_marg, {})
        base_result = base_likelihood.evaluate(params_base, {})

        assert jnp.isfinite(marg_result)
        # With a ±1 Mpc window around the true distance the marginalized value
        # should be within ~1 nat of the point-estimate log-likelihood.  The
        # small residual arises from the integration grid not landing exactly on
        # the likelihood peak and from quadrature curvature.
        assert jnp.abs(marg_result - base_result) < 1.0, (
            f"Marginalized ({marg_result:.4f}) should match base ({base_result:.4f}) "
            "within 1 nat when prior is a narrow window around the true distance"
        )

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        """Per-detector f_min dict (triggers the frequency_masks code path) must
        produce a finite result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), f"Expected finite log-likelihood, got {result}"


class TestPhaseDistanceMarginalizedTransientLikelihoodFD:
    """Tests for TransientLikelihoodFD with phase_marginalization and distance_marginalization."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_d_L_prior(xmin: float = 100.0, xmax: float = 5000.0) -> PowerLawPrior:
        return PowerLawPrior(xmin=xmin, xmax=xmax, alpha=2.0, parameter_names=["d_L"])

    @staticmethod
    def params_without_d_L_phase() -> dict:
        return {
            "M_c": 30.0,
            "eta": 0.249,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "t_c": 0.0,
            "iota": 0.0,
            "ra": 1.375,
            "dec": -1.2108,
            "psi": 0.0,
        }

    # ------------------------------------------------------------------
    # Validation tests
    # ------------------------------------------------------------------

    def test_init_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                phase_marginalization=True,
                distance_marginalization={"distance_prior": self.make_d_L_prior()},
                fixed_parameters={"phase_c": 0.0},
            )

    def test_init_fixed_d_L_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have d_L fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                phase_marginalization=True,
                distance_marginalization={"distance_prior": self.make_d_L_prior()},
                fixed_parameters={"d_L": 400.0},
            )

    # ------------------------------------------------------------------
    # Numerical evaluation tests
    # ------------------------------------------------------------------

    def test_evaluate_is_finite(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L_phase()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            f"Expected finite phase+distance-marginalized log-likelihood, got {result}"
        )

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L_phase()
        result = likelihood.evaluate(params, {})
        result_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.allclose(result, result_jit), (
            f"JIT result {result_jit} does not match eager result {result}"
        )

    def test_geq_distance_marginalized(self, detectors_and_waveform):
        """Phase+distance marginalization should be >= distance-only marginalization.

        For each distance grid point, phase marginalization computes log I_0(|kappa|),
        which is at least as large as evaluating the fixed phase used by the
        distance-only likelihood, so the final logsumexp is also >=.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        d_prior = self.make_d_L_prior()
        phase_distance_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": d_prior},
        )
        distance_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": d_prior,
                "ref_dist": phase_distance_likelihood.ref_dist,
            },
        )

        params = self.params_without_d_L_phase()
        pd_result = phase_distance_likelihood.evaluate(params, {})
        d_result = distance_likelihood.evaluate({**params, "phase_c": 0.0}, {})

        assert jnp.isfinite(pd_result)
        assert pd_result >= d_result, (
            f"Phase+distance-marginalized ({pd_result:.4f}) should be >= "
            f"distance-marginalized ({d_result:.4f})"
        )

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L_phase()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result), (
            "Phase+distance-marginalized likelihood should be finite with different f_min"
        )


class TestCallableFixedParameters:
    """Tests for the callable-value support in fixed_parameters.

    A value in ``fixed_parameters`` may be a callable
    ``(params: dict) -> Float`` instead of a plain constant.  The callable
    is invoked at each ``evaluate`` call with the *current* parameter dict
    (before the override is applied), which allows derived quantities that
    depend on other sampled parameters to be fixed.
    """

    def test_constant_callable_matches_constant(self, detectors_and_waveform):
        """A callable that ignores params and returns a constant must give the
        same result as passing the constant directly."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        const_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": 0.0, "s2_z": 0.0},
        )
        callable_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={
                "s1_z": lambda p: 0.0,
                "s2_z": lambda p: 0.0,
            },
        )

        params = example_params()
        # Remove the keys that are being fixed so both paths exercise the fix
        params_without = {k: v for k, v in params.items() if k not in ("s1_z", "s2_z")}

        result_const = const_likelihood.evaluate(dict(params_without), {})
        result_callable = callable_likelihood.evaluate(dict(params_without), {})

        assert jnp.allclose(result_const, result_callable), (
            f"Constant ({result_const}) and callable ({result_callable}) fixed_parameters "
            "should give the same result"
        )

    def test_callable_reads_sampled_params(self, detectors_and_waveform):
        """A callable fixed parameter can compute its value from the sampled dict.

        We use an identity-like callable for ``s1_z`` that reads a helper
        key ``_s1_z_raw`` from the param dict.  The result should equal
        evaluating with ``s1_z`` set explicitly.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            # fix s1_z by reading a helper key from the params dict
            fixed_parameters={"s1_z": lambda p: p["_s1_z_raw"]},
        )

        params = example_params()
        params["_s1_z_raw"] = 0.0  # helper value injected into the dict

        # Reference: simply pass s1_z = 0.0 directly
        ref_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        ref_params = example_params()

        result = likelihood.evaluate(dict(params), {})
        ref_result = ref_likelihood.evaluate(dict(ref_params), {})

        assert_all_finite(result)
        assert jnp.allclose(result, ref_result), (
            f"Callable ({result}) should match constant ({ref_result})"
        )

    def test_callable_fixed_parameter_does_not_mutate_input(
        self, detectors_and_waveform
    ):
        """evaluate() must not mutate the caller's params dict.

        TransientLikelihoodFD.evaluate() copies the dict internally, so
        passing params directly should leave it unchanged.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": lambda p: 0.5},
        )

        params = example_params()
        keys_before = set(params.keys())
        values_before = {k: float(v) for k, v in params.items()}

        # Pass the original dict directly — evaluate() must NOT mutate it
        likelihood.evaluate(params, {})

        assert set(params.keys()) == keys_before, (
            "evaluate() must not add or remove keys from the caller's dict"
        )
        for k, v in values_before.items():
            assert float(params[k]) == v, (
                f"evaluate() must not modify params['{k}'] (was {v}, now {float(params[k])})"
            )

    def test_callable_jit_compatible(self, detectors_and_waveform):
        """Callable fixed_parameters — including dict-returning transforms — must
        work under jax.jit for both scalar-lambda and transform.backward forms."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        # --- scalar-lambda form ---
        lambda_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        params = example_params()
        result_lambda = lambda_likelihood.evaluate(dict(params), {})
        result_lambda_jit = jax.jit(lambda_likelihood.evaluate)(dict(params), {})
        assert jnp.isfinite(result_lambda_jit), (
            "JIT scalar-lambda result must be finite"
        )
        assert jnp.allclose(result_lambda, result_lambda_jit), (
            "JIT and eager scalar-lambda results must match"
        )

        # --- dict-returning transform form ---
        transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time=gps, ifo=ifos[0]
        )
        transform_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"t_c": transform.backward},
        )
        params_with_tdet = dict(example_params())
        params_with_tdet["t_det"] = 0.0
        result_transform = transform_likelihood.evaluate(dict(params_with_tdet), {})
        result_transform_jit = jax.jit(transform_likelihood.evaluate)(
            dict(params_with_tdet), {}
        )
        assert jnp.isfinite(result_transform_jit), (
            "JIT transform.backward result must be finite"
        )
        assert jnp.allclose(result_transform, result_transform_jit), (
            "JIT and eager transform.backward results must match"
        )

    def test_callables_evaluated_in_insertion_order_with_chaining(
        self, detectors_and_waveform
    ):
        """Later callables in fixed_parameters see values set by earlier ones.

        The loop applies overrides in insertion order and mutates the working
        copy as it goes, so the second callable can read what the first one wrote.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        seen_s1_z_in_s2_callable = []

        def set_nonzero_s1_z(p):
            return 0.5  # override to a non-default value

        def read_s1_z_for_s2(p):
            # By the time this callable runs, s1_z should already be 0.5
            seen_s1_z_in_s2_callable.append(float(p["s1_z"]))
            return float(p["s1_z"])  # set s2_z = s1_z (chained)

        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": set_nonzero_s1_z, "s2_z": read_s1_z_for_s2},
        )

        params = example_params()  # s1_z=0.0, s2_z=0.0 originally
        likelihood.evaluate(dict(params), {})

        assert len(seen_s1_z_in_s2_callable) == 1, (
            "s2_z callable must have been invoked once"
        )
        assert seen_s1_z_in_s2_callable[0] == 0.5, (
            f"s2_z callable should see s1_z=0.5 (set by first callable), "
            f"but saw {seen_s1_z_in_s2_callable[0]}"
        )

    def test_heterodyned_callable_fixed_parameter(self, detectors_and_waveform):
        """Callable fixed_parameters must work in HeterodynedTransientLikelihoodFD."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()

        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )

        params = example_params()
        result = likelihood.evaluate(params, {})

        assert jnp.isfinite(result), (
            "Heterodyned likelihood with callable fixed_parameters should be finite"
        )

    def test_transform_forward_as_callable(self, detectors_and_waveform):
        """transform.backward can be passed directly as a callable value.

        ``GeocentricArrivalTimeToDetectorArrivalTimeTransform.backward`` maps
        ``t_det -> t_c`` (given sampled ra, dec).  Passing it as the callable
        for ``"t_c"`` should produce the same result as computing ``t_c``
        explicitly via an equivalent lambda.
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time=gps, ifo=ifos[0]
        )

        gmst = compute_gmst(gps)
        t_det_fixed = 0.0

        lambda_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={
                "t_c": lambda p: (
                    t_det_fixed - ifos[0].delay_from_geocenter(p["ra"], p["dec"], gmst)
                ),
            },
        )

        # Subject: use transform.backward directly — returns a dict, key "t_c" extracted
        transform_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"t_c": transform.backward},
        )

        # Params include t_det so the transform can invert it, plus all other required keys
        params = example_params()
        params["t_det"] = t_det_fixed

        result_lambda = lambda_likelihood.evaluate(dict(params), {})
        result_transform = transform_likelihood.evaluate(dict(params), {})

        assert jnp.isfinite(result_lambda), "Lambda fixed t_c result should be finite"
        assert jnp.isfinite(result_transform), (
            "Transform.backward fixed t_c result should be finite"
        )
        assert jnp.allclose(result_lambda, result_transform, atol=1e-6), (
            f"Lambda ({result_lambda}) and transform.backward ({result_transform}) "
            "approaches must give the same likelihood"
        )
