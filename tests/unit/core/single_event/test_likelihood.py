import jax
import jax.numpy as jnp
import pytest
from pathlib import Path
from jimgw.core.single_event.likelihood import (
    ZeroLikelihood,
    TransientLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
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
        likelihood = ZeroLikelihood()
        assert isinstance(likelihood, ZeroLikelihood)
        assert likelihood.evaluate(example_params(), {}) == 0.0


class TestTransientLikelihoodFD:
    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def make_d_L_prior(xmin: float = 100.0, xmax: float = 5000.0) -> PowerLawPrior:
        return PowerLawPrior(xmin=xmin, xmax=xmax, alpha=2.0, parameter_names=["d_L"])

    @staticmethod
    def params_without_d_L() -> dict:
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

    # ── Initialization ────────────────────────────────────────────────────────

    def test_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        assert isinstance(likelihood, TransientLikelihoodFD)
        assert likelihood.frequencies[0] == fmin
        assert likelihood.frequencies[-1] == fmax
        assert likelihood.trigger_time == gps
        assert hasattr(likelihood, "gmst")

    def test_uninitialized_data_raises(self):
        gps = 1126259462.4
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_psd(
                PowerSpectrum.from_file(
                    str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
                )
            )
        with pytest.raises(ValueError, match="does not have initialized data"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
            )

    def test_partially_initialized_data_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        extra = get_H1()
        extra.set_psd(
            PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{extra.name}.npz")
            )
        )
        with pytest.raises(ValueError, match=r"H1.*does not have initialized data"):
            TransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    def test_uninitialized_psd_raises(self):
        gps = 1126259462.4
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_data(
                Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
            )
        with pytest.raises(ValueError, match="does not have initialized PSD"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
            )

    def test_partially_initialized_psd_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        extra = get_H1()
        extra.set_data(
            Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{extra.name}.npz"))
        )
        with pytest.raises(ValueError, match="H1.*does not have initialized PSD"):
            TransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params()
        ll = likelihood.evaluate(params, {})
        assert jnp.isfinite(ll)
        ll_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.isfinite(ll_jit)
        assert jnp.allclose(ll, ll_jit)
        ll_diff = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
        ).evaluate(params, {})
        assert jnp.isfinite(ll_diff)
        assert jnp.allclose(ll, ll_diff, atol=1e-2)

    # ── Time marginalization ───────────────────────────────────────────────────

    def test_time_marg_initialization(self, detectors_and_waveform):
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

    def test_time_marg_custom_tc_range(self, detectors_and_waveform):
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

    def test_time_marg_fixed_t_c_raises(self, detectors_and_waveform):
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

    def test_time_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))

    def test_time_marg_jit_matches(self, detectors_and_waveform):
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
        assert jnp.allclose(
            likelihood.evaluate(params, {}), jax.jit(likelihood.evaluate)(params, {})
        )

    def test_time_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))

    def test_time_marg_geq_base(self, detectors_and_waveform):
        """Time-marginalized log-likelihood is >= base - log(N_total).

        _reduce_time returns logsumexp(tc_range terms) - log(N_total), where
        N_total = len(tc_array) is the full FFT size.  Since t_c=0 lies within
        the default tc_range (-0.1, 0.1), the t=0 term is always included in
        the logsumexp, giving the tight lower bound marg ≥ base - log(N_total).
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        base = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        marg_result = marg.evaluate(example_params(), {})
        base_result = base.evaluate(example_params(), {})
        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result - jnp.log(len(marg.tc_array))

    # ── Phase marginalization ──────────────────────────────────────────────────

    def test_phase_marg_fixed_phase_c_raises(self, detectors_and_waveform):
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

    def test_phase_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))

    def test_phase_marg_jit_matches(self, detectors_and_waveform):
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
        assert jnp.allclose(
            likelihood.evaluate(params, {}), jax.jit(likelihood.evaluate)(params, {})
        )

    def test_phase_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))

    def test_phase_marg_geq_base(self, detectors_and_waveform):
        """Phase-marginalized likelihood must be >= base (I_0(x) >= 1)."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        base = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        marg_result = marg.evaluate(example_params(), {})
        base_result = base.evaluate(example_params(), {})
        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result

    # ── Phase + time marginalization ───────────────────────────────────────────

    def test_phase_time_marg_fixed_phase_c_raises(self, detectors_and_waveform):
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

    def test_phase_time_marg_fixed_t_c_raises(self, detectors_and_waveform):
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

    def test_phase_time_marg_evaluation(self, detectors_and_waveform):
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
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))

    def test_phase_time_marg_jit_matches(self, detectors_and_waveform):
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
        assert jnp.allclose(
            likelihood.evaluate(params, {}), jax.jit(likelihood.evaluate)(params, {})
        )

    def test_phase_time_marg_different_fmin(self, detectors_and_waveform):
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
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))

    def test_phase_time_marg_geq_base(self, detectors_and_waveform):
        """Phase+time marg log-likelihood is >= base - log(N_total).

        Same reasoning as test_time_marg_geq_base: the time marginalisation
        normalises by len(tc_array), so the result is offset by -log(N_total)
        relative to the point estimate.  The t=0 term is always included,
        giving marg ≥ base - log(N_total).
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        base = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        marg_result = marg.evaluate(example_params(), {})
        base_result = base.evaluate(example_params(), {})
        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result - jnp.log(len(marg.tc_array))

    def test_phase_time_marg_geq_phase_only(self, detectors_and_waveform):
        """Phase+time marg log-likelihood is >= phase-only - log(N_total).

        The time marginalisation normalises by len(tc_array), so the result
        is offset by -log(N_total).  The t=0 term is always included in the
        logsumexp, giving pt ≥ p - log(N_total).
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        pt = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        p = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        pt_result = pt.evaluate(example_params(), {})
        p_result = p.evaluate(example_params(), {})
        assert jnp.isfinite(pt_result)
        assert pt_result >= p_result - jnp.log(len(pt.tc_array))

    # ── Distance marginalization ───────────────────────────────────────────────

    def test_dist_marg_no_prior_raises(self):
        from pydantic import ValidationError
        from jimgw.core.single_event.likelihood import DistanceMargConfig

        with pytest.raises(ValidationError):
            DistanceMargConfig()  # type: ignore[call-arg]

    def test_dist_marg_fixed_d_L_raises(self, detectors_and_waveform):
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

    def test_dist_marg_prior_missing_d_L_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(
            ValueError, match="must be a 1D prior with parameter_names="
        ):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                distance_marginalization={
                    "distance_prior": UniformPrior(
                        xmin=10.0, xmax=100.0, parameter_names=["M_c"]
                    )
                },
            )

    def test_dist_marg_n_dist_points_raises(self, detectors_and_waveform):
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

    def test_dist_marg_negative_ref_dist_raises(self, detectors_and_waveform):
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

    def test_dist_marg_power_law_prior(self, detectors_and_waveform):
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

    def test_dist_marg_uniform_prior(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": UniformPrior(
                    xmin=100.0, xmax=5000.0, parameter_names=["d_L"]
                )
            },
        )
        assert isinstance(likelihood, TransientLikelihoodFD)

    def test_dist_marg_default_ref_dist(self, detectors_and_waveform):
        """Default ref_dist is the midpoint of [xmin, xmax]."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": self.make_d_L_prior(xmin=200.0, xmax=1000.0)
            },
        )
        assert jnp.isclose(likelihood.ref_dist, (200.0 + 1000.0) / 2.0)

    def test_dist_marg_custom_ref_dist(self, detectors_and_waveform):
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

    def test_dist_marg_log_weights_normalized(self, detectors_and_waveform):
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

    def test_dist_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        result = likelihood.evaluate(self.params_without_d_L(), {})
        assert jnp.isfinite(result)

    def test_dist_marg_jit_matches(self, detectors_and_waveform):
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
        assert jnp.allclose(result, result_jit)

    def test_dist_marg_matches_base_near_true_distance(self, detectors_and_waveform):
        """With a narrow prior around d_L=400 the marginalized value ≈ point estimate."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_d_L = 400.0
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": UniformPrior(
                    xmin=true_d_L - 1.0, xmax=true_d_L + 1.0, parameter_names=["d_L"]
                ),
                "ref_dist": true_d_L,
            },
        )
        base = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        marg_result = marg.evaluate(self.params_without_d_L(), {})
        base_result = base.evaluate(example_params(), {})
        assert jnp.isfinite(marg_result)
        assert jnp.abs(marg_result - base_result) < 1.0

    def test_dist_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert jnp.isfinite(likelihood.evaluate(self.params_without_d_L(), {}))

    # ── Phase + distance marginalization ──────────────────────────────────────

    def test_phase_dist_marg_fixed_phase_c_raises(self, detectors_and_waveform):
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

    def test_phase_dist_marg_fixed_d_L_raises(self, detectors_and_waveform):
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

    def test_phase_dist_marg_evaluation(self, detectors_and_waveform):
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
        result = likelihood.evaluate(self.params_without_d_L_phase(), {})
        assert jnp.isfinite(result)

    def test_phase_dist_marg_jit_matches(self, detectors_and_waveform):
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
        assert jnp.allclose(result, result_jit)

    def test_phase_dist_marg_geq_distance_marginalized(self, detectors_and_waveform):
        """Phase+distance marginalization must be >= distance-only."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        d_prior = self.make_d_L_prior()
        pd = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": d_prior},
        )
        d = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": d_prior,
                "ref_dist": pd.ref_dist,
            },
        )
        params = self.params_without_d_L_phase()
        pd_result = pd.evaluate(params, {})
        d_result = d.evaluate({**params, "phase_c": 0.0}, {})
        assert jnp.isfinite(pd_result)
        assert pd_result >= d_result

    def test_phase_dist_marg_different_fmin(self, detectors_and_waveform):
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
        assert jnp.isfinite(likelihood.evaluate(self.params_without_d_L_phase(), {}))

    # ── Callable fixed parameters ──────────────────────────────────────────────

    def test_callable_constant_matches_constant(self, detectors_and_waveform):
        """A callable returning a constant gives the same result as passing the constant."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        const = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": 0.0, "s2_z": 0.0},
        )
        callable_ = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        params = {
            k: v for k, v in example_params().items() if k not in ("s1_z", "s2_z")
        }
        assert jnp.allclose(
            const.evaluate(dict(params), {}), callable_.evaluate(dict(params), {})
        )

    def test_callable_reads_sampled_params(self, detectors_and_waveform):
        """A callable can derive its value from other sampled parameters."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": lambda p: p["_s1_z_raw"]},
        )
        params = example_params()
        params["_s1_z_raw"] = 0.0
        ref = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        result = likelihood.evaluate(dict(params), {})
        assert_all_finite(result)
        assert jnp.allclose(result, ref.evaluate(example_params(), {}))

    def test_callable_does_not_mutate_input(self, detectors_and_waveform):
        """evaluate() must not mutate the caller's params dict."""
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
        likelihood.evaluate(params, {})
        assert set(params.keys()) == keys_before
        for k, v in values_before.items():
            assert float(params[k]) == v

    def test_callable_jit_compatible(self, detectors_and_waveform):
        """Callable fixed_parameters must work under jax.jit."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        lambda_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        params = example_params()
        result = lambda_likelihood.evaluate(dict(params), {})
        result_jit = jax.jit(lambda_likelihood.evaluate)(dict(params), {})
        assert jnp.isfinite(result_jit)
        assert jnp.allclose(result, result_jit)

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
        params_with_tdet = {**example_params(), "t_det": 0.0}
        result_tr = transform_likelihood.evaluate(dict(params_with_tdet), {})
        result_tr_jit = jax.jit(transform_likelihood.evaluate)(
            dict(params_with_tdet), {}
        )
        assert jnp.isfinite(result_tr_jit)
        assert jnp.allclose(result_tr, result_tr_jit)

    def test_callable_insertion_order_chaining(self, detectors_and_waveform):
        """Later callables in fixed_parameters see values written by earlier ones."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        seen_s1_z = []

        def set_nonzero_s1_z(p):
            return 0.5

        def read_s1_z_for_s2(p):
            seen_s1_z.append(float(p["s1_z"]))
            return float(p["s1_z"])

        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": set_nonzero_s1_z, "s2_z": read_s1_z_for_s2},
        )
        likelihood.evaluate(dict(example_params()), {})
        assert len(seen_s1_z) == 1
        assert seen_s1_z[0] == 0.5

    def test_callable_transform_backward(self, detectors_and_waveform):
        """transform.backward and an equivalent lambda must give the same result."""
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
                )
            },
        )
        transform_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"t_c": transform.backward},
        )
        params = {**example_params(), "t_det": t_det_fixed}
        result_lambda = lambda_likelihood.evaluate(dict(params), {})
        result_transform = transform_likelihood.evaluate(dict(params), {})
        assert jnp.isfinite(result_lambda)
        assert jnp.isfinite(result_transform)
        assert jnp.allclose(result_lambda, result_transform, atol=1e-6)


class TestHeterodynedTransientLikelihoodFD:
    # ── Initialization ────────────────────────────────────────────────────────

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
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
        params = example_params()
        result = likelihood.evaluate(params, {})
        assert jnp.isfinite(result)
        assert jnp.allclose(result, base.evaluate(params, {}))

    def test_initialization_stores_attributes(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
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

    def test_no_reference_params_and_no_prior_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params, {}), jax.jit(likelihood.evaluate)(params, {})
        )

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
        )
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))

    def test_maximize_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_params = example_params()
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
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ll_injected = float(base.evaluate(true_params, {}))
        fixed_parameters = {
            k: v for k, v in true_params.items() if k not in ("M_c", "eta")
        }
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters=fixed_parameters,
            prior=CombinePrior(
                [
                    UniformPrior(25.0, 35.0, parameter_names=["M_c"]),
                    UniformPrior(0.125, 1.0, parameter_names=["q"]),
                ]
            ),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
            optimizer_popsize=10,
            optimizer_n_steps=50,
        )
        result = likelihood.reference_parameters.copy()
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
        assert set(result.keys()) == expected_keys
        for val in result.values():
            assert jnp.isfinite(val)
        assert jnp.isfinite(likelihood.evaluate(result, {}))
        assert jnp.isclose(float(base.evaluate(result, {})), ll_injected)
        common_keys_allclose(result, true_params)

    # ── Phase marginalization ──────────────────────────────────────────────────

    def test_phase_marg_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_parameters=example_params(),
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_phase_marg_no_reference_raises(self, detectors_and_waveform):
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

    def test_phase_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))

    def test_phase_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params, {}), jax.jit(likelihood.evaluate)(params, {})
        )

    def test_phase_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))

    def test_phase_marg_matches_base_at_ref_params(self, detectors_and_waveform):
        """At reference params the het phase-marg should closely match non-het phase-marg."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()
        phase_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
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
        assert jnp.allclose(het_result, phase_result, atol=1e-1)

    # ── Callable fixed parameters ──────────────────────────────────────────────

    def test_callable_fixed_parameter(self, detectors_and_waveform):
        """Callable fixed_parameters must work in HeterodynedTransientLikelihoodFD."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        assert jnp.isfinite(likelihood.evaluate(example_params(), {}))


class TestMultibandedTransientLikelihoodFD:
    # ── Initialization ────────────────────────────────────────────────────────

    def test_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        assert isinstance(likelihood, MultibandedTransientLikelihoodFD)
        assert likelihood.minimum_frequency == fmin
        assert likelihood.maximum_frequency == fmax
        assert likelihood.trigger_time == gps
        assert hasattr(likelihood, "gmst")
        assert likelihood.reference_chirp_mass == 20.0
        assert likelihood.n_bands > 0

    def test_band_setup(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        assert len(likelihood.unique_frequencies) > 0
        assert len(likelihood.unique_to_original) > 0
        assert len(likelihood.unique_frequencies) <= len(likelihood.unique_to_original)
        for ifo in ifos:
            assert ifo.name in likelihood.linear_coeffs
            assert ifo.name in likelihood.quadratic_coeffs
            assert len(likelihood.linear_coeffs[ifo.name]) == len(
                likelihood.unique_to_original
            )

    def test_uninitialized_data_raises(self):
        gps = 1126259462.4
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_psd(
                PowerSpectrum.from_file(
                    str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
                )
            )
        with pytest.raises(ValueError, match="does not have initialized data"):
            MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
                reference_chirp_mass=20.0,
            )

    def test_partially_initialized_data_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        extra = get_H1()
        extra.set_psd(
            PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{extra.name}.npz")
            )
        )
        with pytest.raises(ValueError, match=r"H1.*does not have initialized data"):
            MultibandedTransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_chirp_mass=20.0,
            )

    def test_uninitialized_psd_raises(self):
        gps = 1126259462.4
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_data(
                Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
            )
        with pytest.raises(ValueError, match="does not have initialized PSD"):
            MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
                reference_chirp_mass=20.0,
            )

    def test_partially_initialized_psd_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        extra = get_H1()
        extra.set_data(
            Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{extra.name}.npz"))
        )
        with pytest.raises(ValueError, match=r"H1.*does not have initialized PSD"):
            MultibandedTransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_chirp_mass=20.0,
            )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        params = example_params()
        ll = likelihood.evaluate(params, {})
        assert jnp.isfinite(ll)
        ll_jit = jax.jit(likelihood.evaluate)(params, {})
        assert jnp.isfinite(ll_jit)
        assert jnp.allclose(ll, ll_jit)
        ll_diff = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        ).evaluate(params, {})
        assert jnp.isfinite(ll_diff)

    def test_evaluate_does_not_mutate_params(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        params = example_params()
        keys_before = set(params.keys())
        values_before = {k: float(v) for k, v in params.items()}
        likelihood.evaluate(params, {})
        assert set(params.keys()) == keys_before
        for k, v in values_before.items():
            assert float(params[k]) == v

    @pytest.mark.slow
    def test_accuracy_factor_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        params = example_params()
        for acc in [1.0, 5.0, 10.0]:
            likelihood = MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                accuracy_factor=acc,
                reference_chirp_mass=20.0,
            )
            assert jnp.isfinite(likelihood.evaluate(params, {})), (
                f"Not finite for accuracy_factor={acc}"
            )
