import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from pydantic import ValidationError

from jimgw.core.constants import EARTH_RADIUS_LIGHT_S
from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    GaussianPrior,
    PowerLawPrior,
    Prior,
    UniformPrior,
)
from jimgw.core.single_event import likelihood as likelihood_module
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import GroundBased3G, get_ET, get_H1, get_L1
from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
    TransientLikelihoodFD,
    ZeroLikelihood,
)
from jimgw.core.single_event.marginalization_config import (
    DistanceMargConfig,
    TimeMargConfig,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
)
from jimgw.core.single_event.waveform import (
    RippleIMRPhenomD,
    RippleIMRPhenomHM,
    RippleIMRPhenomPv2,
    RippleIMRPhenomXPHM,
    RippleSineGaussian,
)
from jimgw.core.utils import log_i0
from jimgw.samplers.config import BlackJAXSwiGConfig
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
        data.trigger_time = gps
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


def _waveform_cache_prior() -> CombinePrior:
    bounds = {
        "M_c": (20.0, 40.0),
        "q": (0.5, 1.0),
        "s1_z": (-0.05, 0.05),
        "s2_z": (-0.05, 0.05),
        "iota": (0.1, 3.0),
        "ra": (0.0, 2.0 * jnp.pi),
        "dec": (-1.5, 1.5),
        "psi": (0.0, jnp.pi),
        "t_c": (-0.05, 0.05),
    }
    return CombinePrior(
        [
            UniformPrior(lo, hi, parameter_names=[name])
            for name, (lo, hi) in bounds.items()
        ]
    )


class TestZeroLikelihood:
    def test_initialization_and_evaluation(self, detectors_and_waveform):
        likelihood = ZeroLikelihood()
        assert isinstance(likelihood, ZeroLikelihood)
        assert likelihood.evaluate(example_params()) == 0.0


class TestTransientLikelihoodFD:
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        assert isinstance(likelihood, TransientLikelihoodFD)
        assert likelihood.frequencies[0] == fmin
        assert likelihood.frequencies[-1] == fmax
        assert likelihood.trigger_time == gps

    def test_the_event_constants_stay_out_of_the_parameters(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        params = example_params()
        assert set(likelihood._prepare_parameters(params)) == set(params)

    def test_cached_waveform_matches_full_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        assert jnp.allclose(
            likelihood.evaluate(params),
            likelihood.evaluate_from_waveform(params, cache),
        )

    def test_cached_waveform_restores_physical_distance_scaling(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        physical_waveform = waveform(
            likelihood.frequencies, likelihood._prepare_parameters(params)
        )

        for polarization, strain in physical_waveform.items():
            assert jnp.allclose(cache[polarization] / params["d_L"], strain)

    def test_cached_waveform_supports_cache_reusing_changes(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        moved = {
            **params,
            "d_L": params["d_L"] * 1.2,
            "ra": params["ra"] + 0.1,
            "psi": 0.2,
            "t_c": 0.01,
        }
        assert jnp.allclose(
            likelihood.evaluate(moved),
            likelihood.evaluate_from_waveform(moved, cache),
        )

    def test_evaluate_bypasses_waveform_cache_machinery(
        self, detectors_and_waveform, monkeypatch
    ):
        # evaluate() must generate the waveform directly at the true d_L; the
        # waveform-cache machinery (unit-distance normalization when supported,
        # rescaling on the way out) is reserved for the SwiG cache path
        # (generate_waveform()/evaluate_from_waveform()) and must not add
        # overhead to every other sampler's likelihood calls.
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )

        def _unexpected(*args, **kwargs):
            raise AssertionError("evaluate() must not use the waveform-cache machinery")

        monkeypatch.setattr(likelihood, "_waveform_sky_for_cache", _unexpected)
        monkeypatch.setattr(likelihood, "_waveform_sky_from_cache", _unexpected)

        assert jnp.isfinite(likelihood.evaluate(example_params()))

    @pytest.mark.slow
    def test_waveform_cache_infers_blocks_after_transform_and_marginalization(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["ra", "dec"],
            ["psi"],
            ["t_c"],
        ]
        jim = Jim(
            likelihood,
            _waveform_cache_prior(),
            BlackJAXSwiGConfig(blocks=blocks, n_live=8, n_delete_frac=0.25),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
            periodic=[],
        )
        assert jim.sampler._rebuild_required_by_block == {
            (0, 1): True,
            (2, 3): True,
            (4,): True,
            (5, 6): False,
            (7,): False,
            (8,): False,
        }

    def test_waveform_cache_distance_block_reuses_cache_with_time_marginalization(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
            time_marginalization={"tc_range": (-0.03, 0.03)},
        )
        prior = CombinePrior(
            [
                base_prior
                for base_prior in _waveform_cache_prior().base_prior
                if base_prior.parameter_names != ("t_c",)
            ]
            + [UniformPrior(100.0, 1000.0, parameter_names=["d_L"])]
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["d_L"],
            ["ra", "dec"],
            ["psi"],
        ]
        jim = Jim(
            likelihood,
            prior,
            BlackJAXSwiGConfig(blocks=blocks, n_live=8, n_delete_frac=0.25),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
        )
        assert jim.sampler._rebuild_required_by_block == {
            (0, 1): True,
            (2, 3): True,
            (4,): True,
            (8,): False,
            (5, 6): False,
            (7,): False,
        }

    def test_waveform_cache_rejects_non_sampling_parameter_in_blocks(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["ra", "dec"],
            ["psi"],
            ["t_c"],
            ["d_L"],
        ]
        with pytest.raises(ValueError, match="d_L.*not sampling parameters"):
            Jim(
                likelihood,
                _waveform_cache_prior(),
                BlackJAXSwiGConfig(blocks=blocks, n_live=8, n_delete_frac=0.25),
                likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
            )

    def test_uninitialized_data_raises(self):
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
            )

    def test_partially_initialized_data_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        extra = get_H1()
        extra.set_psd(
            PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{extra.name}.npz")
            )
        )
        with pytest.raises(ValueError, match=r"H1.*does not have initialized data"):
            TransientLikelihoodFD(
                detectors=ifos + [extra], waveform=waveform, f_min=fmin, f_max=fmax
            )

    def test_uninitialized_psd_raises(self):
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
            )

    def test_partially_initialized_psd_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        extra = get_H1()
        extra.set_data(
            Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{extra.name}.npz"))
        )
        with pytest.raises(ValueError, match="H1.*does not have initialized PSD"):
            TransientLikelihoodFD(
                detectors=ifos + [extra], waveform=waveform, f_min=fmin, f_max=fmax
            )

    @pytest.mark.parametrize(
        "likelihood_class",
        [
            TransientLikelihoodFD,
            HeterodynedTransientLikelihoodFD,
            MultibandedTransientLikelihoodFD,
        ],
    )
    def test_time_domain_waveform_raises(
        self, detectors_and_waveform, likelihood_class
    ):
        # On the frequencies, taken as times, such a model is zero and so is the likelihood.
        ifos, _waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="frequency-domain waveform model"):
            likelihood_class(ifos, RippleSineGaussian(), f_min=fmin, f_max=fmax)
        assert not any(ifo.is_frozen for ifo in ifos)

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        params = example_params()
        ll = likelihood.evaluate(params)
        assert jnp.isfinite(ll)
        ll_jit = jax.jit(likelihood.evaluate)(params)
        assert jnp.isfinite(ll_jit)
        assert jnp.allclose(ll, ll_jit)
        ll_diff = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
        ).evaluate(params)
        assert jnp.isfinite(ll_diff)
        assert jnp.allclose(ll, ll_diff, atol=1e-2)

    # ------------------------------------------------------------------
    # Time marginalization
    # ------------------------------------------------------------------

    def test_time_marg_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={},
        )
        assert isinstance(likelihood, TransientLikelihoodFD)
        assert hasattr(likelihood, "tc_range")
        assert hasattr(likelihood, "tc_array")
        assert hasattr(likelihood, "time_frequency_indices")
        assert likelihood.tc_range == (-0.1, 0.1)

    def test_time_marg_custom_tc_range(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        custom_range = (-0.05, 0.05)
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={"tc_range": custom_range},
        )
        assert likelihood.tc_range == custom_range

    @pytest.mark.parametrize("phase", [False, True])
    @pytest.mark.parametrize("tc_range", [(-0.1, 0.1), (0.02, 0.08), (-0.08, -0.02)])
    def test_time_reduction_matches_masked_fft_values_and_gradients(
        self, detectors_and_waveform, phase, tc_range
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            ifos,
            waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=phase,
            time_marginalization={"tc_range": tc_range},
        )
        values = jax.random.normal(jax.random.key(42), (2, len(likelihood.frequencies)))
        correlation = (values[0] + 1j * values[1]) / len(likelihood.frequencies)
        indices = jnp.rint(likelihood.frequencies * likelihood.duration).astype(
            jnp.int64
        )
        selected = (likelihood.tc_array > tc_range[0]) & (
            likelihood.tc_array < tc_range[1]
        )

        def reference(correlation):
            dense = jnp.zeros(len(likelihood.tc_array), dtype=correlation.dtype)
            spectrum = jnp.fft.fft(dense.at[indices].add(correlation))
            integrand = log_i0(jnp.abs(spectrum)) if phase else spectrum.real
            # Uniform prior on t_c over tc_range: every sample weighs spacing / width.
            spacing = likelihood.tc_array[1] - likelihood.tc_array[0]
            return logsumexp(jnp.where(selected, integrand, -jnp.inf)) + jnp.log(
                spacing / (tc_range[1] - tc_range[0])
            )

        reduce = likelihood._reduce_phase_time if phase else likelihood._reduce_time
        actual_value, actual_gradient = jax.jit(jax.value_and_grad(reduce))(correlation)
        expected_value, expected_gradient = jax.jit(jax.value_and_grad(reference))(
            correlation
        )
        assert actual_value == pytest.approx(float(expected_value), abs=1e-12)
        assert jnp.allclose(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-12)

    @pytest.mark.slow
    def test_close_frequency_grids_preserve_duplicate_fft_bin_accumulation(self):
        ifos = [get_H1(), get_L1()]
        for ifo, delta_t in zip(ifos, [4.0 / 128, (4.0 / 128) * (1 + 1e-13)]):
            data = Data.from_td(
                jnp.zeros(128), delta_t, start_time=100.0, trigger_time=102.0
            )
            ifo.set_data(data)
            ifo.set_psd(PowerSpectrum(jnp.ones(data.n_freq), data.frequencies))
        likelihood = TransientLikelihoodFD(
            ifos,
            lambda frequencies, params: {},
            f_min=1.0,
            f_max=8.0,
            time_marginalization=True,
        )
        indices = jnp.rint(likelihood.frequencies * likelihood.duration).astype(
            jnp.int64
        )
        assert len(jnp.unique(indices)) < len(indices)
        assert not isinstance(likelihood.time_frequency_indices, slice)
        correlation = jnp.arange(len(indices)) + 0.1j
        expected = jnp.fft.fft(
            jnp.zeros(len(likelihood.tc_array), dtype=correlation.dtype)
            .at[indices]
            .add(correlation)
        )
        assert jnp.allclose(
            jax.jit(likelihood._time_fft)(correlation), expected, rtol=1e-12, atol=1e-12
        )

    def test_time_marg_fixed_t_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have t_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                time_marginalization={},
                fixed_parameters={"t_c": 0.0},
            )

    def test_time_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={},
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_time_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={},
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_time_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            time_marginalization={},
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    @pytest.mark.parametrize(
        "n_samples", [(128, 128), (129, 129), (128, 256), (129, 257)]
    )
    @pytest.mark.parametrize("include_highest_frequency", [False, True])
    def test_time_marg_disjoint_ranges_match_direct_time_quadrature(
        self, n_samples, include_highest_frequency
    ):
        # Unequal sampling rates, odd sample counts and included Nyquist bins all
        # need a dense FFT grid whose unused interior frequencies are zero.
        ifos = [get_H1(), get_L1()]
        for ifo, n_time in zip(ifos, n_samples):
            time = jnp.arange(n_time) * (4.0 / n_time)
            data = Data.from_td(
                0.1 * jnp.sin(2 * jnp.pi * 3.0 * time)
                + 0.05 * jnp.cos(2 * jnp.pi * (n_time // 2) / 4.0 * time),
                4.0 / n_time,
                start_time=100.0,
                trigger_time=102.0,
                window=jnp.ones(n_time),
            )
            ifo.set_data(data)
            ifo.set_psd(PowerSpectrum(jnp.ones(data.n_freq), data.frequencies))

        def waveform(frequencies, params):
            amplitude = jnp.exp(-1j * params["phase_c"]) / params["d_L"]
            return {
                "p": jnp.full(frequencies.shape, 0.3 + 0.2j) * amplitude,
                "c": jnp.full(frequencies.shape, 0.1 - 0.4j) * amplitude,
            }

        f_min = {"H1": 1.0, "L1": 8.0}
        f_max = {
            "H1": 4.0,
            "L1": float(ifos[1].data.frequencies[-1])
            if include_highest_frequency
            else 12.0,
        }
        base = TransientLikelihoodFD(
            ifos,
            waveform,
            f_min=f_min,
            f_max=f_max,
        )
        marginalized = TransientLikelihoodFD(
            ifos,
            waveform,
            f_min=f_min,
            f_max=f_max,
            time_marginalization={"tc_range": (-0.5, 0.5)},
        )
        params = {**example_params(), "d_L": 1.0}
        times = marginalized.tc_array
        direct = jax.vmap(lambda tc: base.evaluate({**params, "t_c": tc}))(times)
        selected = (times > -0.5) & (times < 0.5)
        spacing = times[1] - times[0]
        expected = logsumexp(jnp.where(selected, direct, -jnp.inf)) + jnp.log(
            spacing / 1.0
        )
        assert float(marginalized.evaluate(params)) == pytest.approx(
            float(expected), abs=1e-10
        )
        cache = marginalized.generate_waveform(params)
        assert float(
            marginalized.evaluate_from_waveform(params, cache)
        ) == pytest.approx(float(expected), abs=1e-10)

    @pytest.mark.parametrize(
        "tc_range",
        [(0.1, -0.1), (0.1, 0.1), (jnp.nan, 0.1), (-0.1, jnp.inf), (10.0, 11.0)],
    )
    def test_invalid_time_marginalization_range_raises(
        self, detectors_and_waveform, tc_range
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="tc_range"):
            TransientLikelihoodFD(
                ifos,
                waveform,
                f_min=fmin,
                f_max=fmax,
                time_marginalization={"tc_range": tc_range},
            )

    def test_time_marg_geq_base(self, detectors_and_waveform):
        """Time-marginalized log-likelihood is >= base + log(spacing / width).

        _reduce_time returns logsumexp(tc_range terms) + log(spacing / width) for a
        uniform prior on t_c over tc_range, as in bilby.  Since t_c=0 lies within
        the default tc_range (-0.1, 0.1), the t=0 term is always included in the
        logsumexp, giving the lower bound marg >= base + log(spacing / width).
        """
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={},
        )
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        marg_result = marg.evaluate(example_params())
        base_result = base.evaluate(example_params())
        assert jnp.isfinite(marg_result)
        weight = (marg.tc_array[1] - marg.tc_array[0]) / (
            marg.tc_range[1] - marg.tc_range[0]
        )
        assert marg_result >= base_result + jnp.log(weight)

    # ------------------------------------------------------------------
    # Phase marginalization
    # ------------------------------------------------------------------

    def test_phase_marg_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_phase_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_phase_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_marg_geq_base(self, detectors_and_waveform):
        """Phase-marginalized likelihood must be >= base (I_0(x) >= 1)."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
        )
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        marg_result = marg.evaluate(example_params())
        base_result = base.evaluate(example_params())
        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result

    # ------------------------------------------------------------------
    # Phase + time marginalization
    # ------------------------------------------------------------------

    def test_phase_time_marg_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                time_marginalization={},
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_phase_time_marg_fixed_t_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have t_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                time_marginalization={},
                phase_marginalization=True,
                fixed_parameters={"t_c": 0.0},
            )

    def test_phase_time_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={},
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_time_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={},
            phase_marginalization=True,
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_phase_time_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            time_marginalization={},
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_time_marg_geq_base(self, detectors_and_waveform):
        """Phase+time marg log-likelihood is >= base + log(spacing / width).

        Same reasoning as test_time_marg_geq_base: the t=0 term is always
        included, giving marg >= base + log(spacing / width).
        """
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={},
            phase_marginalization=True,
        )
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        marg_result = marg.evaluate(example_params())
        base_result = base.evaluate(example_params())
        assert jnp.isfinite(marg_result)
        weight = (marg.tc_array[1] - marg.tc_array[0]) / (
            marg.tc_range[1] - marg.tc_range[0]
        )
        assert marg_result >= base_result + jnp.log(weight)

    def test_phase_time_marg_geq_phase_only(self, detectors_and_waveform):
        """Phase+time marg log-likelihood is >= phase-only + log(spacing / width).

        The t=0 term is always included in the logsumexp, giving
        pt >= p + log(spacing / width).
        """
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        pt = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={},
            phase_marginalization=True,
        )
        p = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
        )
        pt_result = pt.evaluate(example_params())
        p_result = p.evaluate(example_params())
        assert jnp.isfinite(pt_result)
        weight = (pt.tc_array[1] - pt.tc_array[0]) / (pt.tc_range[1] - pt.tc_range[0])
        assert pt_result >= p_result + jnp.log(weight)

    # ------------------------------------------------------------------
    # Distance marginalization
    # ------------------------------------------------------------------

    def test_dist_marg_no_prior_raises(self):
        from pydantic import ValidationError

        from jimgw.core.single_event.likelihood import DistanceMargConfig

        with pytest.raises(ValidationError):
            DistanceMargConfig()  # type: ignore[call-arg]

    def test_dist_marg_fixed_d_L_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have d_L fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                distance_marginalization={"distance_prior": self.make_d_L_prior()},
                fixed_parameters={"d_L": 400.0},
            )

    def test_dist_marg_prior_missing_d_L_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(
            ValueError, match="must be a 1D prior with parameter_names="
        ):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                distance_marginalization={
                    "distance_prior": UniformPrior(
                        xmin=10.0, xmax=100.0, parameter_names=["M_c"]
                    )
                },
            )

    def test_dist_marg_n_dist_points_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="greater than or equal to 2"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                distance_marginalization={
                    "distance_prior": self.make_d_L_prior(),
                    "n_dist_points": 1,
                },
            )

    def test_dist_marg_negative_ref_dist_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="greater than 0"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                distance_marginalization={
                    "distance_prior": self.make_d_L_prior(),
                    "ref_dist": -100.0,
                },
            )

    def test_dist_marg_power_law_prior(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert isinstance(likelihood, TransientLikelihoodFD)

    def test_dist_marg_uniform_prior(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            distance_marginalization={
                "distance_prior": UniformPrior(
                    xmin=100.0, xmax=5000.0, parameter_names=["d_L"]
                )
            },
        )
        assert isinstance(likelihood, TransientLikelihoodFD)

    def test_dist_marg_default_ref_dist(self, detectors_and_waveform):
        """Default ref_dist is the midpoint of [xmin, xmax]."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            distance_marginalization={
                "distance_prior": self.make_d_L_prior(xmin=200.0, xmax=1000.0)
            },
        )
        assert jnp.isclose(likelihood.ref_dist, (200.0 + 1000.0) / 2.0)

    def test_dist_marg_custom_ref_dist(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            distance_marginalization={
                "distance_prior": self.make_d_L_prior(),
                "ref_dist": 500.0,
            },
        )
        assert jnp.isclose(likelihood.ref_dist, 500.0)

    def test_dist_marg_log_weights_normalized(self, detectors_and_waveform):
        from jax.scipy.special import logsumexp

        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert jnp.isclose(logsumexp(likelihood.log_weights), 0.0, atol=1e-5)

    def test_dist_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        result = likelihood.evaluate(self.params_without_d_L())
        assert jnp.isfinite(result)

    def test_dist_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L()
        result = likelihood.evaluate(params)
        result_jit = jax.jit(likelihood.evaluate)(params)
        assert jnp.allclose(result, result_jit)

    def test_dist_marg_matches_base_near_true_distance(self, detectors_and_waveform):
        """With a narrow prior around d_L=400 the marginalized value ≈ point estimate."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        true_d_L = 400.0
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            distance_marginalization={
                "distance_prior": UniformPrior(
                    xmin=true_d_L - 1.0, xmax=true_d_L + 1.0, parameter_names=["d_L"]
                ),
                "ref_dist": true_d_L,
            },
        )
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        marg_result = marg.evaluate(self.params_without_d_L())
        base_result = base.evaluate(example_params())
        assert jnp.isfinite(marg_result)
        assert jnp.abs(marg_result - base_result) < 1.0

    def test_dist_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert jnp.isfinite(likelihood.evaluate(self.params_without_d_L()))

    # ------------------------------------------------------------------
    # Phase + distance marginalization
    # ------------------------------------------------------------------

    def test_phase_dist_marg_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                phase_marginalization=True,
                distance_marginalization={"distance_prior": self.make_d_L_prior()},
                fixed_parameters={"phase_c": 0.0},
            )

    def test_phase_dist_marg_fixed_d_L_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have d_L fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                phase_marginalization=True,
                distance_marginalization={"distance_prior": self.make_d_L_prior()},
                fixed_parameters={"d_L": 400.0},
            )

    def test_phase_dist_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        result = likelihood.evaluate(self.params_without_d_L_phase())
        assert jnp.isfinite(result)

    def test_phase_dist_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L_phase()
        result = likelihood.evaluate(params)
        result_jit = jax.jit(likelihood.evaluate)(params)
        assert jnp.allclose(result, result_jit)

    def test_phase_dist_marg_geq_distance_marginalized(self, detectors_and_waveform):
        """Phase+distance marginalization must be >= distance-only."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        d_prior = self.make_d_L_prior()
        pd = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": d_prior},
        )
        d = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            distance_marginalization={
                "distance_prior": d_prior,
                "ref_dist": pd.ref_dist,
            },
        )
        params = self.params_without_d_L_phase()
        pd_result = pd.evaluate(params)
        d_result = d.evaluate({**params, "phase_c": 0.0})
        assert jnp.isfinite(pd_result)
        assert pd_result >= d_result

    def test_phase_dist_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert jnp.isfinite(likelihood.evaluate(self.params_without_d_L_phase()))

    # ------------------------------------------------------------------
    # Callable fixed parameters
    # ------------------------------------------------------------------

    def test_callable_constant_matches_constant(self, detectors_and_waveform):
        """A callable returning a constant gives the same result as passing the constant."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        const = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            fixed_parameters={"s1_z": 0.0, "s2_z": 0.0},
        )
        callable_ = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        params = {
            k: v for k, v in example_params().items() if k not in ("s1_z", "s2_z")
        }
        assert jnp.allclose(
            const.evaluate(dict(params)), callable_.evaluate(dict(params))
        )

    def test_callable_reads_sampled_params(self, detectors_and_waveform):
        """A callable can derive its value from other sampled parameters."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            fixed_parameters={"s1_z": lambda p: p["_s1_z_raw"]},
        )
        params = example_params()
        params["_s1_z_raw"] = 0.0
        ref = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        result = likelihood.evaluate(dict(params))
        assert_all_finite(result)
        assert jnp.allclose(result, ref.evaluate(example_params()))

    def test_callable_does_not_mutate_input(self, detectors_and_waveform):
        """evaluate() must not mutate the caller's params dict."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            fixed_parameters={"s1_z": lambda p: 0.5},
        )
        params = example_params()
        keys_before = set(params.keys())
        values_before = {k: float(v) for k, v in params.items()}
        likelihood.evaluate(params)
        assert set(params.keys()) == keys_before
        for k, v in values_before.items():
            assert float(params[k]) == v

    def test_callable_jit_compatible(self, detectors_and_waveform):
        """Callable fixed_parameters must work under jax.jit."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        lambda_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        params = example_params()
        result = lambda_likelihood.evaluate(dict(params))
        result_jit = jax.jit(lambda_likelihood.evaluate)(dict(params))
        assert jnp.isfinite(result_jit)
        assert jnp.allclose(result, result_jit)

        transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(ifo=ifos[0])
        transform_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            fixed_parameters={"t_c": transform.backward},
        )
        params_with_tdet = {**example_params(), "t_det": 0.0}
        result_tr = transform_likelihood.evaluate(dict(params_with_tdet))
        result_tr_jit = jax.jit(transform_likelihood.evaluate)(dict(params_with_tdet))
        assert jnp.isfinite(result_tr_jit)
        assert jnp.allclose(result_tr, result_tr_jit)

    def test_callable_insertion_order_chaining(self, detectors_and_waveform):
        """Later callables in fixed_parameters see values written by earlier ones."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
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
            fixed_parameters={"s1_z": set_nonzero_s1_z, "s2_z": read_s1_z_for_s2},
        )
        likelihood.evaluate(dict(example_params()))
        assert len(seen_s1_z) == 1
        assert seen_s1_z[0] == 0.5

    def test_callable_transform_backward(self, detectors_and_waveform):
        """transform.backward and an equivalent lambda must give the same result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(ifo=ifos[0])
        gmst = compute_gmst(gps)
        t_det_fixed = 0.0

        lambda_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
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
            fixed_parameters={"t_c": transform.backward},
        )
        params = {**example_params(), "t_det": t_det_fixed}
        result_lambda = lambda_likelihood.evaluate(dict(params))
        result_transform = transform_likelihood.evaluate(dict(params))
        assert jnp.isfinite(result_lambda)
        assert jnp.isfinite(result_transform)
        assert jnp.allclose(result_lambda, result_transform, atol=1e-6)


class TestHeterodynedTransientLikelihoodFD:
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        ref_params = example_params()
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_parameters=ref_params,
        )
        assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)
        params = example_params()
        result = likelihood.evaluate(params)
        assert jnp.isfinite(result)
        assert jnp.allclose(result, base.evaluate(params))

    def test_initialization_stores_attributes(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_parameters=example_params(),
        )
        assert hasattr(likelihood, "freq_grid_low")
        assert hasattr(likelihood, "freq_grid_high")
        assert hasattr(likelihood, "bin_widths")
        assert len(likelihood.freq_grid) == likelihood.n_bins + 1
        for attr in ("freq_grid_low", "freq_grid_high", "bin_widths"):
            assert attr not in vars(likelihood)
        for det in ifos:
            for attr in ("summary_data", "_waveform_ref"):
                obj = getattr(likelihood, attr)
                assert det.name in obj
                assert jnp.isfinite(obj[det.name]).all()
            assert likelihood._waveform_ref[det.name].shape == (2 * likelihood.n_bins,)

    @pytest.mark.slow
    def test_preserves_waveform_grids_with_one_projection(
        self, detectors_and_waveform, monkeypatch
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            ifos,
            waveform,
            f_min=fmin,
            f_max=fmax,
            n_bins=32,
            reference_parameters=example_params(),
        )
        waveform_grids = []
        response_grids = []
        original_waveform = type(waveform).__call__
        original_response = type(ifos[0]).fd_response

        def record_waveform(model, frequencies, params):
            waveform_grids.append(frequencies)
            return original_waveform(model, frequencies, params)

        def record_response(detector, frequencies, h_sky, params):
            response_grids.append((detector.name, frequencies))
            return original_response(detector, frequencies, h_sky, params)

        monkeypatch.setattr(type(waveform), "__call__", record_waveform)
        monkeypatch.setattr(type(ifos[0]), "fd_response", record_response)
        params = example_params()
        direct = likelihood.evaluate(params)
        assert len(waveform_grids) == 2
        assert jnp.array_equal(waveform_grids[0], likelihood.freq_grid_low)
        assert jnp.array_equal(waveform_grids[1], likelihood.freq_grid_high)
        assert [name for name, _ in response_grids] == likelihood.detector_names
        assert all(
            jnp.array_equal(
                grid,
                jnp.concatenate((likelihood.freq_grid_low, likelihood.freq_grid_high)),
            )
            for _, grid in response_grids
        )

        waveform_grids.clear()
        response_grids.clear()
        cache = likelihood.generate_waveform(params)
        assert len(waveform_grids) == 2
        assert jnp.array_equal(waveform_grids[0], likelihood.freq_grid_low)
        assert jnp.array_equal(waveform_grids[1], likelihood.freq_grid_high)
        assert all(
            strain.shape == (likelihood.n_bins,)
            for sky in cache.values()
            for strain in sky.values()
        )
        waveform_grids.clear()
        cached = likelihood.evaluate_from_waveform(params, cache)
        assert not waveform_grids
        assert len(response_grids) == len(ifos)
        assert jnp.allclose(cached, direct)

    @pytest.mark.slow
    def test_projection_preserves_grid_dependent_evaluation_and_gradients(
        self, detectors_and_waveform
    ):
        ifos, waveform, _fmin, _fmax, _gps = detectors_and_waveform

        class GridDependentWaveform(type(waveform)):
            def __call__(self, frequencies, params):
                # Waveforms such as IMRPhenomD depend on the grid's first spacing.
                strain = super().__call__(frequencies, params)
                scale = 1.0 + params["M_c"] * (frequencies[1] - frequencies[0]) * 1e-3
                return {key: scale * value for key, value in strain.items()}

        waveform = GridDependentWaveform(f_ref=20.0)
        f_min = {"H1": 30.0, "L1": 40.0}
        f_max = {"H1": 900.0, "L1": 800.0}
        for detector in ifos:
            mask = (detector.psd.frequencies >= 20.0) & (
                detector.psd.frequencies <= 1000.0
            )
            detector.set_psd(
                PowerSpectrum(detector.psd.values[mask], detector.psd.frequencies[mask])
            )

        def reference_waveform(frequencies, params):
            strain = waveform(frequencies, params)
            mask = (frequencies >= 80.0) & (frequencies <= 600.0)
            return {key: jnp.where(mask, value, 0.0) for key, value in strain.items()}

        likelihood = HeterodynedTransientLikelihoodFD(
            ifos,
            waveform,
            f_min=f_min,
            f_max=f_max,
            n_bins=32,
            reference_parameters=example_params(),
            reference_waveform=reference_waveform,
            phase_marginalization=True,
        )
        assert likelihood.freq_grid_low[0] >= 80.0
        assert likelihood.freq_grid_high[-1] <= 600.0
        low_reference = reference_waveform(
            likelihood.freq_grid_low, likelihood.reference_parameters
        )
        high_reference = reference_waveform(
            likelihood.freq_grid_high, likelihood.reference_parameters
        )
        reference_responses = [
            (
                detector.fd_response(
                    likelihood.freq_grid_low,
                    low_reference,
                    likelihood.reference_parameters,
                ),
                detector.fd_response(
                    likelihood.freq_grid_high,
                    high_reference,
                    likelihood.reference_parameters,
                ),
            )
            for detector in ifos
        ]

        def split_evaluation(params):
            prepared = likelihood._prepare_parameters(params)
            low = waveform(likelihood.freq_grid_low, prepared)
            high = waveform(likelihood.freq_grid_high, prepared)
            match = jnp.zeros((), dtype=jnp.complex128)
            optimal = jnp.zeros(())
            for detector, (low_ref, high_ref) in zip(ifos, reference_responses):
                r_low = (
                    detector.fd_response(likelihood.freq_grid_low, low, prepared)
                    / low_ref
                )
                r_high = (
                    detector.fd_response(likelihood.freq_grid_high, high, prepared)
                    / high_ref
                )
                r0 = (r_low + r_high) / 2
                r1 = (r_high - r_low) / likelihood.bin_widths
                A0, A1, B0, B1 = likelihood.summary_data[detector.name]
                match += jnp.sum(A0 * r0.conj() + A1 * r1.conj())
                optimal += jnp.sum(
                    B0 * jnp.abs(r0) ** 2 + 2 * B1 * (r0 * r1.conj()).real
                ).real
            return log_i0(jnp.abs(match)) - optimal / 2

        params = {**example_params(), "M_c": 30.2, "psi": 0.1, "t_c": 0.002}
        names = ("M_c", "d_L", "psi", "t_c")
        point = jnp.array([params[name] for name in names])

        def with_parameters(evaluate):
            return lambda values: evaluate({**params, **dict(zip(names, values))})

        expected_value, expected_gradient = jax.jit(
            jax.value_and_grad(with_parameters(split_evaluation))
        )(point)
        for evaluate in (
            likelihood.evaluate,
            lambda p: likelihood.evaluate_from_waveform(
                p, likelihood.generate_waveform(p)
            ),
        ):
            value, gradient = jax.jit(jax.value_and_grad(with_parameters(evaluate)))(
                point
            )
            assert jnp.allclose(value, expected_value, rtol=1e-10, atol=1e-8)
            assert jnp.allclose(gradient, expected_gradient, rtol=1e-9, atol=1e-7)

    @pytest.mark.parametrize(
        "bad, message",
        [
            ("empty", "contain polarizations"),
            ("zero", "no support"),
            ("nan", "one finite value per frequency"),
            ("inf", "one finite value per frequency"),
            ("wrong_shape", "one finite value per frequency"),
            ("narrow_support", "fewer than two bin edges"),
            ("internal_hole", "nonzero inside its supported range"),
        ],
    )
    def test_rejects_invalid_reference_waveforms(
        self, detectors_and_waveform, bad, message
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform

        def reference(frequencies, params):
            if bad == "empty":
                return {}
            values = jnp.ones_like(frequencies, dtype=jnp.complex128)
            if bad == "zero":
                values = jnp.zeros_like(values)
            elif bad in ("nan", "inf"):
                values = values.at[0].set(jnp.nan if bad == "nan" else jnp.inf)
            elif bad == "wrong_shape":
                values = values[:-1]
            elif bad == "narrow_support":
                values = jnp.where(
                    frequencies == frequencies[len(frequencies) // 2], values, 0
                )
            elif bad == "internal_hole":
                values = values.at[len(values) // 2].set(0)
            return {"p": values, "c": 1j * values}

        with pytest.raises(ValueError, match=message):
            HeterodynedTransientLikelihoodFD(
                ifos,
                waveform,
                f_min=fmin,
                f_max=fmax,
                reference_parameters=example_params(),
                reference_waveform=reference,
            )
        assert all(detector.is_frozen for detector in ifos)

    @pytest.mark.parametrize("edge_value", [0.0, jnp.nan])
    def test_rejects_invalid_reference_bin_edges(
        self, detectors_and_waveform, edge_value
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        frequencies = ifos[0].data.frequencies
        full_size = int(jnp.sum((frequencies >= fmin) & (frequencies <= fmax)))

        def reference(frequencies, params):
            values = jnp.ones_like(frequencies, dtype=jnp.complex128)
            if len(frequencies) != full_size:
                values = values.at[len(values) // 2].set(edge_value)
            return {"p": values, "c": 1j * values}

        with pytest.raises(ValueError, match="finite|nonzero"):
            HeterodynedTransientLikelihoodFD(
                ifos,
                waveform,
                f_min=fmin,
                f_max=fmax,
                reference_parameters=example_params(),
                reference_waveform=reference,
            )

    @pytest.mark.parametrize("different_reference", [False, True])
    def test_rejects_unsupported_singleton_waveform_grid(
        self, detectors_and_waveform, different_reference
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform

        def reference(frequencies, params):
            values = jnp.ones_like(frequencies, dtype=jnp.complex128)
            return {"p": values, "c": 1j * values}

        with pytest.raises(ValueError, match="increase n_bins|increasing n_bins"):
            HeterodynedTransientLikelihoodFD(
                ifos,
                waveform,
                f_min=fmin,
                f_max=fmax,
                n_bins=1,
                reference_parameters=example_params(),
                reference_waveform=reference if different_reference else None,
            )

    @pytest.mark.slow
    def test_trimming_lower_bins_regenerates_grid_dependent_references(
        self, detectors_and_waveform, monkeypatch
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        monkeypatch.setattr(
            HeterodynedTransientLikelihoodFD,
            "_make_binning_scheme",
            lambda self, frequencies, n_bins: jnp.array(
                [20.0, 30.0, 50.0, 80.0, 120.0, 160.0]
            ),
        )

        def reference(frequencies, params):
            cutoff = 20.0 + 1.1 * (frequencies[1] - frequencies[0])
            values = jnp.where(frequencies >= cutoff, 1.0 + 0j, 0j)
            return {"p": values, "c": 1j * values}

        likelihood = HeterodynedTransientLikelihoodFD(
            ifos,
            waveform,
            f_min=fmin,
            f_max=fmax,
            n_bins=5,
            reference_parameters=example_params(),
            reference_waveform=reference,
        )
        assert jnp.array_equal(likelihood.freq_grid, jnp.array([80.0, 120.0, 160.0]))
        assert jnp.isfinite(likelihood.evaluate(example_params()))
        assert all(
            jnp.all(jnp.isfinite(response) & (response != 0))
            for response in likelihood._waveform_ref.values()
        )

    def test_rejects_nonfinite_derived_coefficients(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        for detector in ifos:
            detector.set_psd(
                PowerSpectrum(
                    jnp.full_like(detector.psd.values, 1e-300),
                    detector.psd.frequencies,
                )
            )

        def reference(frequencies, params):
            values = jnp.full_like(frequencies, 1e100, dtype=jnp.complex128)
            return {"p": values, "c": 1j * values}

        with pytest.raises(ValueError, match="coefficients.*finite"):
            HeterodynedTransientLikelihoodFD(
                ifos,
                waveform,
                f_min=fmin,
                f_max=fmax,
                reference_parameters=example_params(),
                reference_waveform=reference,
            )

    def test_no_reference_params_and_no_prior_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
            )

    def test_uninitialized_data_raises(self):
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_psd(
                PowerSpectrum.from_file(
                    str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
                )
            )
        with pytest.raises(ValueError, match="does not have initialized data"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                reference_parameters=example_params(),
            )

    def test_partially_initialized_data_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        extra = get_H1()
        extra.set_psd(
            PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{extra.name}.npz")
            )
        )
        with pytest.raises(ValueError, match=r"H1.*does not have initialized data"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                reference_parameters=example_params(),
            )

    def test_uninitialized_psd_raises(self):
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_data(
                Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
            )
        with pytest.raises(ValueError, match="does not have initialized PSD"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                reference_parameters=example_params(),
            )

    def test_partially_initialized_psd_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        extra = get_H1()
        extra.set_data(
            Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{extra.name}.npz"))
        )
        with pytest.raises(ValueError, match=r"H1.*does not have initialized PSD"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                reference_parameters=example_params(),
            )

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_parameters=example_params(),
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_cached_waveform_matches_full_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            n_bins=32,
            reference_parameters=example_params(),
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        assert jnp.allclose(
            likelihood.evaluate_from_waveform(params, cache),
            likelihood.evaluate(params),
        )

    def test_cached_waveform_supports_cache_reusing_changes(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            n_bins=32,
            reference_parameters=example_params(),
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        projected = {
            **params,
            "d_L": params["d_L"] * 1.2,
            "psi": params["psi"] + 0.1,
            "t_c": 0.002,
        }
        assert jnp.allclose(
            likelihood.evaluate_from_waveform(projected, cache),
            likelihood.evaluate(projected),
        )

    def test_evaluate_bypasses_waveform_cache_machinery(
        self, detectors_and_waveform, monkeypatch
    ):
        # evaluate() must generate the waveform directly at the true d_L; the
        # waveform-cache machinery (unit-distance normalization when supported,
        # rescaling on the way out) is reserved for the SwiG cache path
        # (generate_waveform()/evaluate_from_waveform()) and must not add
        # overhead to every other sampler's likelihood calls.
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            n_bins=32,
            reference_parameters=example_params(),
        )

        def _unexpected(*args, **kwargs):
            raise AssertionError("evaluate() must not use the waveform-cache machinery")

        monkeypatch.setattr(likelihood, "_waveform_sky_for_cache", _unexpected)
        monkeypatch.setattr(likelihood, "_waveform_sky_from_cache", _unexpected)

        assert jnp.isfinite(likelihood.evaluate(example_params()))

    @pytest.mark.slow
    def test_waveform_cache_uses_heterodyned_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            fixed_parameters={"d_L": 400.0},
            f_min=fmin,
            f_max=fmax,
            n_bins=32,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["ra", "dec"],
            ["psi"],
            ["t_c"],
        ]
        jim = Jim(
            likelihood,
            _waveform_cache_prior(),
            BlackJAXSwiGConfig(blocks=blocks, n_live=8, n_delete_frac=0.25),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
        )
        assert jim.sampler._rebuild_required_by_block == {
            (0, 1): True,
            (2, 3): True,
            (4,): True,
            (5, 6): False,
            (7,): False,
            (8,): False,
        }

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            reference_parameters=example_params(),
        )
        # Per-detector frequency bounds do not change the reference waveform
        # support, so the relative-binning grid still starts at the global f_min.
        assert jnp.isclose(likelihood.freq_grid_low[0], fmin)
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_evaluate_does_not_mutate_params(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_parameters=example_params(),
        )
        params = example_params()
        keys_before = set(params.keys())
        values_before = {k: float(v) for k, v in params.items()}
        likelihood.evaluate(params)
        assert set(params.keys()) == keys_before
        for k, v in values_before.items():
            assert float(params[k]) == v

    @pytest.mark.parametrize(
        ("n_bins", "epsilon", "match"),
        [
            (32, 0.5, "mutually exclusive"),
            (0, None, "positive integer"),
            (None, 0.0, "positive number"),
        ],
    )
    def test_invalid_binning_parameters_raise(
        self, detectors_and_waveform, n_bins, epsilon, match
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match=match):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                n_bins=n_bins,
                epsilon=epsilon,
                reference_parameters=example_params(),
            )

    @pytest.mark.slow
    def test_maximize_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_params = example_params()
        for ifo in ifos:
            ifo.set_zero_noise(
                duration=4.0,
                sampling_frequency=fmax * 2,
                trigger_time=gps,
                f_min=fmin,
                f_max=fmax,
            )
            ifo.inject_signal(waveform, true_params)
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax
        )
        ll_injected = float(base.evaluate(true_params))
        fixed_parameters = {
            k: v for k, v in true_params.items() if k not in ("M_c", "eta")
        }
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
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
        }
        assert set(result.keys()) == expected_keys
        for val in result.values():
            assert jnp.isfinite(val)
        assert jnp.isfinite(likelihood.evaluate(result))
        assert jnp.isclose(float(base.evaluate(result)), ll_injected)
        common_keys_allclose(result, true_params)

    @pytest.mark.parametrize(
        "bounds",
        [((20.0, 1024.0), (20.0, 1024.0)), ((20.1, 900.1), (30.1, 512.1))],
    )
    def test_maximize_likelihood_stops_early_at_optimizer_target(
        self, detectors_and_waveform, caplog, monkeypatch, bounds
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_params = example_params()
        for ifo in ifos:
            ifo.set_zero_noise(
                duration=4.0,
                sampling_frequency=fmax * 2,
                trigger_time=gps,
                f_min=fmin,
                f_max=fmax,
            )
            ifo.inject_signal(waveform, true_params)
        fixed_parameters = {
            k: v for k, v in true_params.items() if k not in ("M_c", "eta")
        }
        objectives = []

        def record_objective(*args, **kwargs):
            objective = TransientLikelihoodFD(*args, **kwargs)
            objectives.append(objective)
            return objective

        monkeypatch.setattr(
            likelihood_module, "TransientLikelihoodFD", record_objective
        )
        # A trivially low target is satisfied by the first generation's best
        # fitness, so the loop must exit long before optimizer_n_steps.
        with caplog.at_level(
            logging.DEBUG, logger="jimgw.core.single_event.likelihood"
        ):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min={ifo.name: low for ifo, (low, _) in zip(ifos, bounds)},
                f_max={ifo.name: high for ifo, (_, high) in zip(ifos, bounds)},
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
                optimizer_target=-1e10,
            )
        [finished_message] = [
            record.message
            for record in caplog.records
            if "CMA-ES finished after" in record.message
        ]
        generations = int(finished_message.split("after ")[1].split(" generations")[0])
        assert generations < 50
        [objective] = objectives
        for ifo, frequencies, (low, high) in zip(
            ifos, objective.detector_frequencies, bounds
        ):
            # Off-grid f_min and f_max must keep exactly the data bins inside the
            # frequency range of the likelihood, not expand to that of the data.
            available = ifo.data.frequencies
            expected = available[(available >= low) & (available <= high)]
            assert jnp.array_equal(frequencies, expected)

    @pytest.mark.slow
    def test_maximize_likelihood_stops_early_on_cma_stall(
        self, detectors_and_waveform, monkeypatch, caplog
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_params = example_params()
        for ifo in ifos:
            ifo.set_zero_noise(
                duration=4.0,
                sampling_frequency=fmax * 2,
                trigger_time=gps,
                f_min=fmin,
                f_max=fmax,
            )
            ifo.inject_signal(waveform, true_params)
        fixed_parameters = {
            k: v for k, v in true_params.items() if k not in ("M_c", "eta")
        }

        # Force the CMA-ES stall/convergence branch deterministically,
        # mirroring how test_maximize_likelihood_stops_early_at_optimizer_target
        # forces the target-reached branch with an extreme target rather than
        # relying on real numerical convergence timing (slow/flaky to tune).
        # Must not fire at generation 0: state.best_solution is still the NaN
        # fill from es.init before the first real _step runs, so exiting
        # before any generation completes would surface as a spurious
        # failure elsewhere rather than testing the stall path.
        def fake_cma_cond(
            population, fitness, state, params, restart_state, restart_params
        ):
            return state.generation_counter >= 1

        monkeypatch.setattr(likelihood_module, "cma_cond", fake_cma_cond)

        with caplog.at_level(
            logging.DEBUG, logger="jimgw.core.single_event.likelihood"
        ):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
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
        [finished_message] = [
            record.message
            for record in caplog.records
            if "CMA-ES finished after" in record.message
        ]
        generations = int(finished_message.split("after ")[1].split(" generations")[0])
        assert generations == 1
        assert "stop_reason=stalled" in finished_message

    @pytest.mark.slow
    def test_low_frequency_reference_cutoff_does_not_reindex_summary_data(
        self, detectors_and_waveform, monkeypatch
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        reference_fmin = 80.0
        requested_n_bins = 32

        def fake_compute_coefficients(likelihood, detector, h_ref, f_bins):
            # The frequency range of the likelihood, not that of the data (which
            # here still starts at 0 Hz).
            freqs = likelihood.detector_frequencies[
                list(likelihood.detectors).index(detector)
            ]
            freqs_broadcast = freqs[None, :]
            left_bounds = f_bins[:-1][:, None]
            right_bounds = f_bins[1:][:, None]

            mask = (freqs_broadcast >= left_bounds) & (freqs_broadcast < right_bounds)

            n_freqs = len(freqs)
            n_bins = len(f_bins) - 1
            assert n_freqs > len(f_bins), f"{n_freqs = }, {len(f_bins) = }"
            assert likelihood.n_bins == n_bins, f"{likelihood.n_bins = }, {n_bins = }"
            assert n_bins <= requested_n_bins
            assert freqs[0] == fmin
            assert f_bins[0] >= reference_fmin
            assert mask.shape == (n_bins, n_freqs), (
                f"{mask.shape = }, expected: ({n_bins}, {n_freqs})"
            )
            coeffs = jnp.arange(n_bins, dtype=jnp.float64)
            return jnp.array([coeffs + nn * 100 for nn in range(4)])

        monkeypatch.setattr(
            HeterodynedTransientLikelihoodFD,
            "_compute_coefficients",
            fake_compute_coefficients,
        )

        def reference_waveform(frequencies, params):
            waveform_sky = waveform(frequencies, params)
            mask = frequencies >= reference_fmin
            return {
                polarization: jnp.where(mask, strain, jnp.zeros_like(strain))
                for polarization, strain in waveform_sky.items()
            }

        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            reference_waveform=reference_waveform,
            f_min=fmin,
            f_max=fmax,
            n_bins=requested_n_bins,
            reference_parameters=example_params(),
        )
        assert likelihood.n_bins < requested_n_bins
        assert likelihood.freq_grid_low[0] >= reference_fmin

        expected = jnp.arange(likelihood.n_bins, dtype=jnp.float64)
        expected_arr = jnp.array([expected + nn * 100 for nn in range(4)])
        for detector in ifos:
            assert jnp.array_equal(likelihood.summary_data[detector.name], expected_arr)

    # ------------------------------------------------------------------
    # Coefficient computation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "bin_edges",
        [
            pytest.param([1.03, 3.17, 6.81, 14.07], id="off-grid-edges"),
            pytest.param([2.0, 3.125, 7.5, 14.0], id="exact-edges"),
            pytest.param([2.0, 2.0, 4.0, 4.0, 9.0, 14.0, 14.0], id="duplicate-edges"),
            pytest.param([2.0, 14.0], id="single-bin"),
            pytest.param([17.0, 18.0], id="all-samples-outside"),
            pytest.param([4.0, 4.0], id="zero-width-final-bin"),
        ],
    )
    def test_compute_coefficients_matches_dense_reference(self, bin_edges):
        rng = np.random.default_rng(140)
        freqs = np.linspace(0.0, 16.0, 129)
        detector = get_H1()
        detector.set_data(
            Data.from_fd(
                jnp.array(
                    rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
                ),
                jnp.array(freqs),
            )
        )
        detector.set_psd(
            PowerSpectrum(
                jnp.array(rng.uniform(0.1, 10.0, size=freqs.size)), jnp.array(freqs)
            )
        )
        # The coefficients only need the detector and the selections of its spectra.
        likelihood = object.__new__(HeterodynedTransientLikelihoodFD)
        likelihood.detectors = [detector]
        likelihood._detector_data_selections = (slice(None),)
        likelihood._detector_psd_selections = (slice(None),)

        h_ref = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
        data = np.asarray(detector.data.fd)
        psd = np.asarray(detector.psd.values)
        edges = np.array(bin_edges)
        # Preserve the original dense definition independently of segment indices.
        mask = (freqs[None, :] >= edges[:-1, None]) & (freqs[None, :] < edges[1:, None])
        mask[-1] |= freqs == edges[-1]
        centers = (edges[:-1] + edges[1:]) / 2
        shifted_mask = (freqs[None, :] - centers[:, None]) * mask
        data_prod = data * h_ref.conj() / psd
        self_prod = h_ref * h_ref.conj() / psd
        expected = (
            4
            / float(detector.data.duration)
            * np.array(
                [
                    np.sum(data_prod[None, :] * mask, axis=1),
                    np.sum(data_prod[None, :] * shifted_mask, axis=1),
                    np.sum(self_prod[None, :] * mask, axis=1),
                    np.sum(self_prod[None, :] * shifted_mask, axis=1),
                ]
            )
        )

        actual = likelihood._compute_coefficients(
            detector, jnp.array(h_ref), jnp.array(edges)
        )
        assert actual.shape == (4, len(edges) - 1)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    # ------------------------------------------------------------------
    # Phase marginalization
    # ------------------------------------------------------------------

    def test_phase_marg_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                reference_parameters=example_params(),
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_phase_marg_no_reference_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                phase_marginalization=True,
            )

    def test_phase_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_phase_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_marg_matches_base_at_ref_params(self, detectors_and_waveform):
        """At reference params the het phase-marg should closely match non-het phase-marg."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        ref_params = example_params()
        phase_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
        )
        het_phase_likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_parameters=ref_params,
            phase_marginalization=True,
        )
        params = example_params()
        het_result = het_phase_likelihood.evaluate(params)
        phase_result = phase_likelihood.evaluate(params)
        assert jnp.isfinite(het_result)
        assert jnp.allclose(het_result, phase_result, atol=1e-1)

    # ------------------------------------------------------------------
    # Callable fixed parameters
    # ------------------------------------------------------------------

    def test_callable_fixed_parameter(self, detectors_and_waveform):
        """Callable fixed_parameters must work in HeterodynedTransientLikelihoodFD."""
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_parameters=example_params(),
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))


class TestMultibandedTransientLikelihoodFD:
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_infers_banding_parameters_from_prior(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        mc_prior = UniformPrior(15.0, 30.0, parameter_names=["M_c"])
        tc_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
        prior = CombinePrior([CombinePrior([mc_prior]), tc_prior])

        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, prior=prior
        )

        t_end = min(
            float(ifo.data.start_time) + float(ifo.data.duration) - gps for ifo in ifos
        )
        expected_time_offset = t_end - tc_prior.xmin + EARTH_RADIUS_LIGHT_S
        expected_delta_f_end = 100.0 / (t_end - tc_prior.xmax - EARTH_RADIUS_LIGHT_S)

        assert likelihood.reference_chirp_mass == mc_prior.xmin
        assert jnp.isclose(likelihood.time_offset, expected_time_offset)
        assert jnp.isclose(likelihood.delta_f_end, expected_delta_f_end)

    @pytest.mark.slow
    def test_unbounded_time_prior_uses_defaults(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        prior = CombinePrior([GaussianPrior(0.0, 1.0, parameter_names=["t_c"])])

        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            prior=prior,
            reference_chirp_mass=20.0,
        )

        assert likelihood.time_offset == 2.12
        assert likelihood.delta_f_end == 53.0

    @pytest.mark.parametrize(
        ("prior", "match"),
        [
            (None, "Either reference_chirp_mass or a prior"),
            (
                CombinePrior([UniformPrior(-0.1, 0.1, parameter_names=["t_c"])]),
                "no M_c prior found",
            ),
            (
                CombinePrior([GaussianPrior(20.0, 1.0, parameter_names=["M_c"])]),
                "no M_c prior found",
            ),
        ],
    )
    def test_missing_reference_chirp_mass_raises(
        self, detectors_and_waveform, prior, match
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match=match):
            MultibandedTransientLikelihoodFD(
                detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, prior=prior
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"reference_chirp_mass": 0.0}, "reference_chirp_mass"),
            ({"highest_mode": 0}, "highest_mode"),
            ({"accuracy_factor": 0.0}, "accuracy_factor"),
            ({"time_offset": -1.0}, "time_offset"),
            ({"delta_f_end": 0.0}, "delta_f_end"),
            ({"min_banding_duration": -1.0}, "min_banding_duration"),
            ({"max_banding_frequency": 0.0}, "max_banding_frequency"),
        ],
    )
    def test_invalid_banding_parameters_raise(
        self, detectors_and_waveform, kwargs, match
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        params = {"reference_chirp_mass": 20.0, **kwargs}
        with pytest.raises(ValueError, match=match):
            MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                **params,
            )

    @pytest.mark.slow
    def test_frequency_band_helpers_create_multiple_bands(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_chirp_mass=20.0,
            time_offset=0.0,
        )

        tau, dtaudf = likelihood._compute_tau_dtaudf(100.0)
        fnext, dfnext = likelihood._find_starting_frequency(4.0, fmin)
        no_fnext, no_dfnext = likelihood._find_starting_frequency(
            4.0, likelihood.max_banding_frequency
        )

        assert tau > 0
        assert dtaudf < 0
        assert fnext is not None
        assert dfnext is not None
        assert fmin < fnext < likelihood.max_banding_frequency
        assert dfnext > 0
        assert (no_fnext, no_dfnext) == (None, None)
        assert likelihood.n_bands > 1

    def test_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_chirp_mass=20.0,
        )
        assert isinstance(likelihood, MultibandedTransientLikelihoodFD)
        assert likelihood.minimum_frequency == fmin
        assert likelihood.maximum_frequency == fmax
        assert likelihood.trigger_time == gps
        assert likelihood.reference_chirp_mass == 20.0
        assert likelihood.time_offset == 2.12
        assert likelihood.delta_f_end == 53.0
        assert likelihood.n_bands > 0

    @pytest.mark.slow
    def test_band_setup(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_chirp_mass=20.0,
        )
        assert len(likelihood.unique_frequencies) > 0
        assert len(likelihood.unique_to_original) > 0
        assert len(likelihood.unique_frequencies) <= len(likelihood.unique_to_original)
        for ifo in ifos:
            assert ifo.name in likelihood.linear_coeffs
            assert ifo.name in likelihood.quadratic_coeffs
            assert len(likelihood.linear_coeffs[ifo.name]) == len(
                likelihood.unique_frequencies
            )
            assert len(likelihood.quadratic_coeffs[ifo.name]) == len(
                likelihood.unique_frequencies
            )

    def test_uninitialized_data_raises(self):
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
                reference_chirp_mass=20.0,
            )

    def test_partially_initialized_data_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
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
                reference_chirp_mass=20.0,
            )

    def test_uninitialized_psd_raises(self):
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
                reference_chirp_mass=20.0,
            )

    def test_partially_initialized_psd_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
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
                reference_chirp_mass=20.0,
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_cropped_psds_and_differing_frequency_ranges_match_cached_and_jitted_gradients(
        self, detectors_and_waveform
    ):
        ifos, waveform, _fmin, _fmax, _gps = detectors_and_waveform
        f_min = {"H1": 30.0, "L1": 40.0}
        f_max = {"H1": 900.0, "L1": 800.0}
        for detector in ifos:
            frequencies = detector.psd.frequencies
            mask = (frequencies >= 20.0) & (frequencies <= 1000.0)
            values = jnp.where(
                frequencies < f_min[detector.name],
                jnp.nan,
                jnp.where(
                    frequencies > f_max[detector.name], jnp.inf, detector.psd.values
                ),
            )
            detector.set_psd(PowerSpectrum(values[mask], frequencies[mask]))

        likelihood = MultibandedTransientLikelihoodFD(
            ifos,
            waveform,
            f_min=f_min,
            f_max=f_max,
            reference_chirp_mass=20.0,
            time_offset=0.25,
            min_banding_duration=0.5,
        )
        assert likelihood.n_bands > 1
        assert len(likelihood.unique_frequencies) < len(likelihood.unique_to_original)
        params = {**example_params(), "M_c": 30.2, "psi": 0.1, "t_c": 0.002}
        names = ("M_c", "d_L", "psi", "t_c")
        point = jnp.array([params[name] for name in names])

        def with_parameters(evaluate):
            return lambda values: evaluate({**params, **dict(zip(names, values))})

        expected_value, expected_gradient = jax.value_and_grad(
            with_parameters(likelihood.evaluate)
        )(point)
        for evaluate in (
            likelihood.evaluate,
            lambda p: likelihood.evaluate_from_waveform(
                p, likelihood.generate_waveform(p)
            ),
        ):
            value, gradient = jax.jit(jax.value_and_grad(with_parameters(evaluate)))(
                point
            )
            assert jnp.allclose(value, expected_value, rtol=1e-10, atol=1e-8)
            assert jnp.allclose(gradient, expected_gradient, rtol=1e-9, atol=1e-7)

    @pytest.mark.slow
    def test_cached_waveform_matches_direct_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_chirp_mass=20.0,
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)

        assert jnp.allclose(
            likelihood.evaluate(params),
            likelihood.evaluate_from_waveform(params, cache),
        )
        assert jnp.allclose(
            likelihood.evaluate_from_waveform(params, cache),
            jax.jit(likelihood.evaluate_from_waveform)(params, cache),
        )

        projected = {
            **params,
            "d_L": params["d_L"] * 1.2,
            "psi": params["psi"] + 0.1,
            "t_c": 0.002,
        }
        assert jnp.allclose(
            likelihood.evaluate(projected),
            likelihood.evaluate_from_waveform(projected, cache),
        )

    @pytest.mark.slow
    def test_evaluate_bypasses_waveform_cache_machinery(
        self, detectors_and_waveform, monkeypatch
    ):
        # evaluate() must generate the waveform directly at the true d_L; the
        # waveform-cache machinery (unit-distance normalization when supported,
        # rescaling on the way out) is reserved for the SwiG cache path
        # (generate_waveform()/evaluate_from_waveform()) and must not add
        # overhead to every other sampler's likelihood calls.
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_chirp_mass=20.0,
        )

        def _unexpected(*args, **kwargs):
            raise AssertionError("evaluate() must not use the waveform-cache machinery")

        monkeypatch.setattr(likelihood, "_waveform_sky_for_cache", _unexpected)
        monkeypatch.setattr(likelihood, "_waveform_sky_from_cache", _unexpected)

        assert jnp.isfinite(likelihood.evaluate(example_params()))

    @pytest.mark.slow
    def test_waveform_cache_uses_multibanded_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            fixed_parameters={"d_L": 400.0, "phase_c": 0.0},
            f_min=fmin,
            f_max=fmax,
            reference_chirp_mass=20.0,
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["ra", "dec"],
            ["psi"],
            ["t_c"],
        ]
        jim = Jim(
            likelihood,
            _waveform_cache_prior(),
            BlackJAXSwiGConfig(blocks=blocks, n_live=8, n_delete_frac=0.25),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
        )
        assert jim.sampler._rebuild_required_by_block == {
            (0, 1): True,
            (2, 3): True,
            (4,): True,
            (5, 6): False,
            (7,): False,
            (8,): False,
        }

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_chirp_mass=20.0,
        )
        params = example_params()
        ll = likelihood.evaluate(params)
        assert jnp.isfinite(ll)
        ll_jit = jax.jit(likelihood.evaluate)(params)
        assert jnp.isfinite(ll_jit)
        assert jnp.allclose(ll, ll_jit)
        ll_diff = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            reference_chirp_mass=20.0,
        ).evaluate(params)
        assert jnp.isfinite(ll_diff)

    @pytest.mark.slow
    def test_evaluate_does_not_mutate_params(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            reference_chirp_mass=20.0,
        )
        params = example_params()
        keys_before = set(params.keys())
        values_before = {k: float(v) for k, v in params.items()}
        likelihood.evaluate(params)
        assert set(params.keys()) == keys_before
        for k, v in values_before.items():
            assert float(params[k]) == v

    @pytest.mark.slow
    def test_accuracy_factor_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        params = example_params()
        for acc in [1.0, 5.0, 10.0]:
            likelihood = MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                accuracy_factor=acc,
                reference_chirp_mass=20.0,
            )
            assert jnp.isfinite(likelihood.evaluate(params)), (
                f"Not finite for accuracy_factor={acc}"
            )

    @pytest.mark.slow
    def test_minimum_and_maximum_frequency_are_the_range_of_the_likelihood(
        self, detectors_and_waveform
    ):
        # As in bilby: the band construction starts at f_min and ends at f_max, not at
        # the nearest frequencies of the data.
        ifos, waveform, _fmin, _fmax, _gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            ifos,
            waveform,
            f_min={"H1": 23.3, "L1": 30.0},
            f_max={"H1": 700.7, "L1": 900.1},
            reference_chirp_mass=20.0,
        )

        assert likelihood.minimum_frequency == 23.3
        assert likelihood.maximum_frequency == 900.1
        assert float(likelihood.fb_dfb[0][0]) == 23.3

    @pytest.mark.slow
    def test_a_range_start_between_bin_multiples_keeps_the_first_bin(
        self, detectors_and_waveform
    ):
        _ifos, waveform, _fmin, _fmax, gps = detectors_and_waveform
        params = example_params()
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            psd = PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
            )
            ifo.set_psd(psd)
            # 5 s: the first frequency, 23.4 Hz, times 5 s is 117.00000000000001.
            ifo.set_zero_noise(
                duration=5.0,
                sampling_frequency=2048.0,
                trigger_time=gps,
                f_min=23.3,
                f_max=1000.0,
            )
            ifo.inject_signal(waveform, params)

        one_band = MultibandedTransientLikelihoodFD(
            ifos, waveform, reference_chirp_mass=20.0, time_offset=10.0
        )

        assert one_band.n_bands == 1
        np.testing.assert_allclose(
            one_band.evaluate(params),
            TransientLikelihoodFD(ifos, waveform).evaluate(params),
            rtol=1e-9,
        )

    @pytest.mark.slow
    def test_the_default_banding_times_need_data_ending_2_s_after_the_trigger(
        self, detectors_and_waveform
    ):
        _ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        params = example_params()
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            psd = PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
            )
            ifo.set_psd(psd)
            ifo.set_zero_noise(
                duration=8.0,
                sampling_frequency=2048.0,
                trigger_time=gps,
                f_min=fmin,
                f_max=fmax,
                post_trigger_duration=4.0,
            )
            ifo.inject_signal(waveform, params)

        with pytest.raises(ValueError, match="end 2 s after the trigger time"):
            MultibandedTransientLikelihoodFD(ifos, waveform, reference_chirp_mass=20.0)
        banded = MultibandedTransientLikelihoodFD(
            ifos,
            waveform,
            reference_chirp_mass=20.0,
            time_offset=4.12,
            delta_f_end=53.0,
        )
        np.testing.assert_allclose(
            banded.evaluate(params),
            TransientLikelihoodFD(ifos, waveform).evaluate(params),
            rtol=1e-6,
        )


class TestLikelihoodFrequencyRange:
    """A likelihood has its own frequency range, inside that of the data.

    The frequency range of the data (``Data.frequency_bounds``) is what a
    detector's data hold. The frequency range of the likelihood (``f_min``, ``f_max``)
    is what the likelihood evaluates. Likelihoods share frozen detectors and keep
    their own frequency selections.
    """

    def test_building_a_likelihood_leaves_the_frequency_bounds_alone(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)
        assert all(ifo.data.frequency_bounds == (0.0, jnp.inf) for ifo in ifos)

    def test_two_likelihoods_with_different_frequency_ranges_share_detectors(
        self, detectors_and_waveform
    ):
        # The old likelihood wrote its frequency range into the detector, so building
        # a second one broke the first with a shape error.
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        params = example_params()
        full = TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)
        before = float(full.evaluate(params))
        narrow = TransientLikelihoodFD(ifos, waveform, f_min=30.0, f_max=512.0)
        assert float(full.evaluate(params)) == before
        assert jnp.isfinite(narrow.evaluate(params))
        assert float(narrow.evaluate(params)) != before

    @pytest.mark.parametrize(
        "kind",
        [
            "transient",
            "transient_time",
            "transient_phase",
            pytest.param("heterodyne", marks=pytest.mark.slow),
            pytest.param("multiband", marks=pytest.mark.slow),
        ],
    )
    def test_a_likelihood_shares_and_freezes_the_detectors(
        self, detectors_and_waveform, kind
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        data = tuple(ifo.data for ifo in ifos)
        psds = tuple(ifo.psd for ifo in ifos)
        params = example_params()
        prior = CombinePrior(
            [
                CombinePrior([UniformPrior(15.0, 30.0, parameter_names=["M_c"])]),
                UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
            ]
        )
        likelihood = {
            "transient": lambda: TransientLikelihoodFD(
                ifos, waveform, f_min=fmin, f_max=fmax
            ),
            "transient_time": lambda: TransientLikelihoodFD(
                ifos, waveform, f_min=fmin, f_max=fmax, time_marginalization=True
            ),
            "transient_phase": lambda: TransientLikelihoodFD(
                ifos, waveform, f_min=fmin, f_max=fmax, phase_marginalization=True
            ),
            "heterodyne": lambda: HeterodynedTransientLikelihoodFD(
                ifos, waveform, f_min=fmin, f_max=fmax, reference_parameters=params
            ),
            "multiband": lambda: MultibandedTransientLikelihoodFD(
                ifos, waveform, f_min=fmin, f_max=fmax, prior=prior
            ),
        }[kind]()
        for attribute in (
            "detector_frequencies",
            "detector_frequency_ranges",
            "detector_fd_data",
            "detector_psds",
            "detector_start_times",
            "detector_durations",
            "detector_n_time",
            "trigger_time",
            "duration",
            "df",
            "minimum_frequency",
            "maximum_frequency",
        ):
            assert attribute not in vars(likelihood)
        assert likelihood.duration == ifos[0].data.duration
        assert likelihood.trigger_time == ifos[0].data.trigger_time
        assert likelihood.df == 1 / ifos[0].data.duration
        for i, ifo in enumerate(ifos):
            frequencies, _, _ = ifo.slice_to_frequency_range(fmin, fmax)
            assert jnp.array_equal(likelihood.detector_frequencies[i], frequencies)
        before = float(likelihood.evaluate(params))
        duration = float(likelihood.duration)
        cache = likelihood.generate_waveform(params)
        cached_before = float(likelihood.evaluate_from_waveform(params, cache))
        for i, ifo in enumerate(ifos):
            assert likelihood.detectors[i] is ifo
            assert ifo.data is data[i]
            assert ifo.psd is psds[i]
            assert ifo.is_frozen and data[i].is_frozen and psds[i].is_frozen
            for target, attribute, value in (
                (ifo, "name", "changed"),
                (ifo, "latitude", ifo.latitude + 0.01),
                (data[i], "start_time", data[i].start_time + 0.01),
                (data[i], "frequency_bounds", (40.0, 300.0)),
                (data[i], "window", jnp.ones_like(data[i].window)),
                (psds[i], "values", psds[i].values * 9.0),
            ):
                with pytest.raises(RuntimeError, match="frozen"):
                    setattr(target, attribute, value)
            with pytest.raises(RuntimeError, match="frozen"):
                ifo.set_data(Data())
            with pytest.raises(RuntimeError, match="frozen"):
                ifo.set_psd(PowerSpectrum())
        assert float(likelihood.evaluate(params)) == before
        assert float(likelihood.evaluate_from_waveform(params, cache)) == cached_before
        assert float(jax.jit(likelihood.evaluate)(params)) == pytest.approx(before)
        assert float(jax.jit(likelihood.evaluate_from_waveform)(params, cache)) == (
            pytest.approx(cached_before)
        )
        assert float(likelihood.duration) == duration
        # Reordering and replacing the caller's list cannot change detector order.
        ifos.reverse()
        assert float(likelihood.evaluate(params)) == before
        ifos[:] = [get_H1()]
        assert float(likelihood.evaluate(params)) == before
        ifos.clear()
        assert float(likelihood.evaluate_from_waveform(params, cache)) == cached_before

    def test_the_frequency_range_of_the_data_is_the_default(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        for ifo in ifos:
            ifo.data.frequency_bounds = (fmin, fmax)
        default = TransientLikelihoodFD(ifos, waveform)
        explicit = TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)
        assert all(
            jnp.array_equal(default_freqs, explicit_freqs)
            for default_freqs, explicit_freqs in zip(
                default.detector_frequencies, explicit.detector_frequencies
            )
        )
        params = example_params()
        assert float(default.evaluate(params)) == float(explicit.evaluate(params))

    @pytest.mark.parametrize("frequency_range", [(10.0, 1024.0), (20.0, 2000.0)])
    def test_a_range_wider_than_the_data_raises(
        self, detectors_and_waveform, frequency_range
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        for ifo in ifos:
            ifo.data.frequency_bounds = (fmin, fmax)
        f_min, f_max = frequency_range
        with pytest.raises(
            ValueError, match="wider than the frequency range of the data"
        ):
            TransientLikelihoodFD(ifos, waveform, f_min=f_min, f_max=f_max)

    def test_a_range_starting_at_zero_raises(self, detectors_and_waveform):
        ifos, waveform, _fmin, fmax, _gps = detectors_and_waveform
        # Real data: the frequency range of the data starts at 0 Hz, where waveforms
        # are not defined.
        with pytest.raises(ValueError, match="positive f_min"):
            TransientLikelihoodFD(ifos, waveform)
        with pytest.raises(ValueError, match="positive f_min"):
            TransientLikelihoodFD(ifos, waveform, f_min=0.0, f_max=fmax)

    def test_f_max_above_nyquist_raises_and_infinity_means_nyquist(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, _fmax, _gps = detectors_and_waveform
        nyquist = float(ifos[0].data.sampling_frequency) / 2
        with pytest.raises(ValueError, match="Nyquist"):
            TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=nyquist + 100.0)
        for f_max in (None, jnp.inf):
            likelihood = TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=f_max)
            assert all(
                float(frequencies[0]) == fmin and float(frequencies[-1]) == nyquist
                for frequencies in likelihood.detector_frequencies
            )

    def test_the_nyquist_frequency_of_a_non_round_sampling_rate_is_accepted(
        self, detectors_and_waveform
    ):
        _ifos, waveform, fmin, _fmax, gps = detectors_and_waveform
        # 1 / (1 / 744) is not exactly 744, so half of it lies just below the top bin.
        data = Data.from_td(
            jnp.zeros(2976), 1 / 744.0, start_time=gps - 2.0, trigger_time=gps
        )
        ifo = get_H1()
        ifo.set_data(data)
        ifo.set_psd(PowerSpectrum.from_file(str(FIXTURES_DIR / "GW150914_psd_H1.npz")))
        likelihood = TransientLikelihoodFD([ifo], waveform, f_min=fmin)
        assert float(likelihood.detector_frequencies[0][-1]) == float(
            data.frequencies[-1]
        )
        TransientLikelihoodFD([ifo], waveform, f_min=fmin, f_max=744.0 / 2)

    def test_an_empty_range_raises(self, detectors_and_waveform):
        ifos, waveform, _fmin, _fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="empty"):
            TransientLikelihoodFD(ifos, waveform, f_min=100.0, f_max=100.0)

    def test_frequency_ranges_can_differ_between_detectors(
        self, detectors_and_waveform
    ):
        ifos, waveform, _fmin, _fmax, _gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            ifos,
            waveform,
            f_min={"H1": 20.0, "L1": 30.0},
            f_max={"H1": 1024.0, "L1": 512.0},
        )
        assert [
            (float(frequencies[0]), float(frequencies[-1]))
            for frequencies in likelihood.detector_frequencies
        ] == [(20.0, 1024.0), (30.0, 512.0)]

    def test_a_range_error_leaves_the_data_of_every_detector_unfixed(
        self, detectors_and_waveform
    ):
        ifos, waveform, _fmin, _fmax, _gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Nyquist"):
            TransientLikelihoodFD(
                ifos,
                waveform,
                f_min={"H1": 20.0, "L1": 20.0},
                f_max={"H1": 1024.0, "L1": 5000.0},
            )
        for ifo in ifos:
            assert not ifo.data.has_fd
            ifo.data.set_tukey_window(alpha=0.1)

    @pytest.mark.parametrize(
        "kind",
        [
            "transient",
            pytest.param("heterodyne", marks=pytest.mark.slow),
            pytest.param("multiband", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.parametrize("noncontiguous", [False, True])
    def test_a_cropped_psd_uses_its_own_selection_under_jit(
        self, detectors_and_waveform, noncontiguous, kind
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        original = ifos[0]
        cropped = deepcopy(original)
        mask = (cropped.psd.frequencies >= fmin) & (cropped.psd.frequencies <= fmax)
        frequencies = cropped.psd.frequencies[mask]
        values = cropped.psd.values[mask]
        if noncontiguous:
            # An unused entry between selected frequencies requires integer indexing.
            midpoint = len(frequencies) // 2
            frequencies = jnp.concatenate(
                [frequencies[:midpoint], jnp.array([0.0]), frequencies[midpoint:]]
            )
            values = jnp.concatenate(
                [values[:midpoint], jnp.array([jnp.nan]), values[midpoint:]]
            )
        # set_psd() interpolates onto the full data grid; attach the cropped PSD
        # directly to exercise the selection of its frequencies.
        cropped.psd = PowerSpectrum(values, frequencies)
        psd = cropped.psd
        likelihood_class, options = {
            "transient": (TransientLikelihoodFD, {}),
            "heterodyne": (
                HeterodynedTransientLikelihoodFD,
                {"reference_parameters": example_params()},
            ),
            "multiband": (
                MultibandedTransientLikelihoodFD,
                {"reference_chirp_mass": 20.0},
            ),
        }[kind]
        full = likelihood_class([original], waveform, f_min=fmin, f_max=fmax, **options)
        selected = likelihood_class(
            [cropped], waveform, f_min=fmin, f_max=fmax, **options
        )
        assert selected.detectors[0].psd is psd
        params = example_params()
        expected = float(full.evaluate(params))
        assert float(selected.evaluate(params)) == expected
        assert float(jax.jit(selected.evaluate)(params)) == pytest.approx(expected)
        cache = selected.generate_waveform(params)
        assert float(jax.jit(selected.evaluate_from_waveform)(params, cache)) == (
            pytest.approx(expected)
        )

    @pytest.mark.slow
    def test_heterodyne_and_multiband_use_their_own_frequency_ranges(self):
        # 16 s of zero noise with a low-mass signal are long enough for several bands.
        # The likelihood ranges differ between the detectors and lie inside the range of
        # the data, which stays as it was.
        waveform = RippleIMRPhenomD(f_ref=20.0)
        params = {
            **example_params(),
            "M_c": 6.0,
            "eta": 0.2,
            "d_L": 150.0,
            "iota": 0.5,
            "psi": 0.3,
        }
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_psd(
                PowerSpectrum.from_file(
                    str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
                )
            )
            ifo.set_zero_noise(
                duration=16.0,
                sampling_frequency=2048.0,
                trigger_time=1126259462.4,
                f_min=20.0,
                f_max=1000.0,
            )
            ifo.inject_signal(waveform, params)
        f_min = {"H1": 30.0, "L1": 40.0}
        f_max = {"H1": 900.0, "L1": 800.0}
        # Invalid values outside the selected likelihood ranges must not contribute
        # to either the precomputed multiband coefficients or the full integral.
        for ifo in ifos:
            frequencies = ifo.psd.frequencies
            values = jnp.where(
                frequencies < f_min[ifo.name],
                jnp.nan,
                jnp.where(frequencies > f_max[ifo.name], jnp.inf, ifo.psd.values),
            )
            ifo.set_psd(PowerSpectrum(values, frequencies))
        full = TransientLikelihoodFD(ifos, waveform, f_min=f_min, f_max=f_max)
        heterodyned = HeterodynedTransientLikelihoodFD(
            ifos, waveform, f_min=f_min, f_max=f_max, reference_parameters=params
        )
        prior = CombinePrior(
            [
                CombinePrior([UniformPrior(5.0, 8.0, parameter_names=["M_c"])]),
                UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
            ]
        )
        multibanded = MultibandedTransientLikelihoodFD(
            ifos, waveform, f_min=f_min, f_max=f_max, prior=prior
        )
        expected = [(30.0, 900.0), (40.0, 800.0)]
        for likelihood in (heterodyned, multibanded):
            assert [
                (float(frequencies[0]), float(frequencies[-1]))
                for frequencies in likelihood.detector_frequencies
            ] == expected
        assert all(ifo.data.frequency_bounds == (20.0, 1000.0) for ifo in ifos)
        assert multibanded.n_bands > 1
        # The heterodyne likelihood is exact at its reference point only.
        assert float(heterodyned.evaluate(params)) == pytest.approx(
            float(full.evaluate(params)), abs=1e-3
        )
        for update in ({}, {"M_c": 6.01}, {"t_c": 0.001, "d_L": 160.0}):
            point = {**params, **update}
            assert float(multibanded.evaluate(point)) == pytest.approx(
                float(full.evaluate(point)), abs=5e-3
            )
        before = float(full.evaluate(params))
        with pytest.raises(ValueError, match="PSD.*not finite and positive"):
            TransientLikelihoodFD(ifos, waveform, f_min=20.0, f_max=1000.0)
        assert all(ifo.is_frozen for ifo in ifos)
        assert float(full.evaluate(params)) == before


class TestLikelihoodTriggerTime:
    """A likelihood reads its trigger time from the data of its detectors.

    The data of every detector must have the same trigger time, inside them. Unless
    it is set, the trigger time of data is 2 s before their end, so detectors that
    analyse the same segment agree.
    """

    CLASSES = (
        TransientLikelihoodFD,
        HeterodynedTransientLikelihoodFD,
        MultibandedTransientLikelihoodFD,
    )

    @staticmethod
    def inject(ifos, waveform, fmin, fmax, trigger_times):
        for ifo, trigger_time in zip(ifos, trigger_times):
            ifo.set_zero_noise(
                duration=4.0,
                sampling_frequency=fmax * 2,
                trigger_time=trigger_time,
                f_min=fmin,
                f_max=fmax,
            )
            ifo.inject_signal(waveform, example_params())

    def test_the_likelihood_reads_the_trigger_time_of_the_data(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)
        assert likelihood.trigger_time == gps

    @pytest.mark.slow
    def test_every_likelihood_class_reads_it(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        self.inject(ifos, waveform, fmin, fmax, (gps, gps))
        prior = CombinePrior(
            [
                CombinePrior([UniformPrior(15.0, 30.0, parameter_names=["M_c"])]),
                UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
            ]
        )
        heterodyned = HeterodynedTransientLikelihoodFD(
            ifos,
            waveform,
            f_min=fmin,
            f_max=fmax,
            reference_parameters=example_params(),
        )
        multibanded = MultibandedTransientLikelihoodFD(
            ifos, waveform, f_min=fmin, f_max=fmax, prior=prior
        )
        assert heterodyned.trigger_time == multibanded.trigger_time == gps

    def test_the_default_is_two_seconds_before_the_end_of_the_data(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        for ifo in ifos:
            ifo.data.trigger_time = None
        likelihood = TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)
        data = ifos[0].data
        assert likelihood.trigger_time == data.start_time + data.duration - 2.0

    @pytest.mark.parametrize("likelihood_class", CLASSES)
    def test_detectors_with_different_segments_raise(
        self, detectors_and_waveform, likelihood_class
    ):
        # With the default, the trigger time follows the end of the data.
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        for ifo in ifos:
            ifo.data.trigger_time = None
        later = ifos[1].data
        ifos[1].set_data(
            Data.from_td(
                td=later.td,
                delta_t=later.delta_t,
                start_time=float(later.start_time) + 1.0,
            )
        )
        with pytest.raises(ValueError, match="different trigger times"):
            likelihood_class(ifos, waveform, f_min=fmin, f_max=fmax)

    @pytest.mark.parametrize("likelihood_class", CLASSES)
    def test_detectors_with_different_trigger_times_raise(
        self, detectors_and_waveform, likelihood_class
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ifos[1].data.trigger_time = gps + 0.01
        with pytest.raises(ValueError, match="different trigger times"):
            likelihood_class(ifos, waveform, f_min=fmin, f_max=fmax)

    def test_detectors_made_with_different_trigger_times_cannot_be_combined(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        self.inject(ifos, waveform, fmin, fmax, (gps, gps + 0.01))
        with pytest.raises(ValueError, match="different trigger times"):
            TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)

    @pytest.mark.parametrize("offset", [-10.0, 10.0], ids=["before", "after"])
    def test_a_trigger_time_outside_the_data_raises(
        self, detectors_and_waveform, offset
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        for ifo in ifos:
            ifo.data.trigger_time = gps + offset
        with pytest.raises(ValueError, match="outside the data of detector 'H1'"):
            TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)

    @pytest.mark.parametrize("edge", ["start_time", "end_time"])
    def test_the_edges_of_the_data_are_inside(self, detectors_and_waveform, edge):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        trigger_time = float(getattr(ifos[0].data, edge))
        ifos[0].data.trigger_time = trigger_time
        likelihood = TransientLikelihoodFD(ifos[:1], waveform, f_min=fmin, f_max=fmax)
        assert likelihood.trigger_time == trigger_time

    def test_a_start_time_left_at_its_default_is_caught(self, detectors_and_waveform):
        # Data.from_td(td, delta_t) starts at 0.0 s, not at a GPS time.
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        for ifo in ifos:
            ifo.set_data(
                Data.from_td(td=ifo.data.td, delta_t=ifo.data.delta_t, trigger_time=gps)
            )
        with pytest.raises(ValueError, match="outside the data"):
            TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)

    def test_a_built_likelihood_prevents_changing_its_trigger_time(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        params = example_params()
        likelihood = TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)
        before = float(likelihood.evaluate(params))
        for ifo in ifos:
            with pytest.raises(RuntimeError, match="frozen"):
                ifo.data.trigger_time = gps + 5.0
        assert likelihood.trigger_time == gps
        assert float(likelihood.evaluate(params)) == before


class TestLikelihoodDetectors:
    """Which sets of detectors a likelihood accepts."""

    @pytest.mark.parametrize(
        "invalid", ["trigger_time", "psd", "frequency_range", "spacing", "duplicate"]
    )
    def test_common_validation_finishes_before_any_detector_is_frozen(
        self, detectors_and_waveform, invalid
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        f_max = fmax
        if invalid == "trigger_time":
            ifos[1].data.trigger_time = gps + 0.01
        elif invalid == "psd":
            ifos[1].set_psd(
                PowerSpectrum(jnp.zeros(ifos[1].psd.n_freq), ifos[1].psd.frequencies)
            )
        elif invalid == "frequency_range":
            f_max = {"H1": fmax, "L1": float(ifos[1].data.sampling_frequency)}
        elif invalid == "spacing":
            ifos[1].set_data(
                Data.from_td(
                    jnp.ones(4096),
                    delta_t=1 / 2048.0,
                    start_time=gps - 1.0,
                    trigger_time=gps,
                )
            )
        else:
            ifos.append(ifos[0])
        with pytest.raises(ValueError):
            TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=f_max)
        for ifo in ifos:
            assert not ifo.is_frozen
            assert not ifo.data.is_frozen
            assert not ifo.psd.is_frozen
            ifo.name = ifo.name
            ifo.data.trigger_time = gps
            ifo.psd.name = "still mutable"

    @pytest.mark.parametrize(
        "kind", ["marginalization", "reference", "binning", "multiband"]
    )
    def test_later_setup_failures_leave_validated_detectors_frozen(
        self, detectors_and_waveform, kind
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        constructors = {
            "marginalization": lambda: TransientLikelihoodFD(
                ifos,
                waveform,
                f_min=fmin,
                f_max=fmax,
                time_marginalization={"tc_range": (0.0, 0.0)},
            ),
            "reference": lambda: HeterodynedTransientLikelihoodFD(
                ifos, waveform, f_min=fmin, f_max=fmax
            ),
            "binning": lambda: HeterodynedTransientLikelihoodFD(
                ifos,
                waveform,
                f_min=fmin,
                f_max=fmax,
                reference_parameters=example_params(),
                n_bins=0,
            ),
            "multiband": lambda: MultibandedTransientLikelihoodFD(
                ifos,
                waveform,
                f_min=fmin,
                f_max=fmax,
                reference_chirp_mass=-1.0,
            ),
        }
        with pytest.raises(ValueError):
            constructors[kind]()
        assert all(
            ifo.is_frozen and ifo.data.is_frozen and ifo.psd.is_frozen for ifo in ifos
        )
        # The validated detectors are reusable after a failed subtype setup.
        likelihood = TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_optimizer_failure_keeps_shared_detectors_frozen(
        self, detectors_and_waveform, monkeypatch
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform

        def fail_optimizer(likelihood, **kwargs):
            objective = TransientLikelihoodFD(
                likelihood.detectors, waveform, f_min=fmin, f_max=fmax
            )
            assert all(ifo.is_frozen for ifo in objective.detectors)
            assert objective.detectors == tuple(ifos)
            raise RuntimeError("optimizer failed")

        monkeypatch.setattr(
            HeterodynedTransientLikelihoodFD, "maximize_likelihood", fail_optimizer
        )
        prior = UniformPrior(20.0, 40.0, parameter_names=["M_c"])
        with pytest.raises(RuntimeError, match="optimizer failed"):
            HeterodynedTransientLikelihoodFD(
                ifos, waveform, f_min=fmin, f_max=fmax, prior=prior
            )
        assert all(ifo.is_frozen for ifo in ifos)

    def test_a_likelihood_needs_a_detector(self, detectors_and_waveform):
        _, waveform, fmin, fmax, _ = detectors_and_waveform
        with pytest.raises(ValueError, match="at least one detector"):
            TransientLikelihoodFD([], waveform, f_min=fmin, f_max=fmax)

    def test_detectors_need_different_names(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, _ = detectors_and_waveform
        with pytest.raises(ValueError, match="different names"):
            TransientLikelihoodFD(
                [ifos[0], deepcopy(ifos[0])], waveform, f_min=fmin, f_max=fmax
            )

    def test_detectors_with_different_frequency_spacing_raise(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        for ifo, duration in zip(ifos, (4.0, 8.0)):
            ifo.set_zero_noise(
                duration=duration,
                sampling_frequency=2 * fmax,
                trigger_time=gps,
                f_min=fmin,
                f_max=fmax,
            )
        with pytest.raises(ValueError, match="same frequency spacing"):
            TransientLikelihoodFD(ifos, waveform, f_min=fmin, f_max=fmax)

    def test_a_range_with_fewer_than_two_frequencies_raises(
        self, detectors_and_waveform
    ):
        ifos, waveform, _, _, _ = detectors_and_waveform
        with pytest.raises(ValueError, match="at least two"):
            TransientLikelihoodFD(ifos, waveform, f_min=20.0, f_max=20.1)


@pytest.fixture
def likelihood_inputs():
    detectors = [get_H1(), get_L1()]
    for detector in detectors:
        data = Data.from_td(jnp.zeros(1024), 1 / 256, trigger_time=2.0)
        data.frequency_bounds = (20.0, 100.0)
        detector.set_data(data)
        detector.set_psd(
            PowerSpectrum(jnp.ones_like(data.frequencies), data.frequencies)
        )
    return detectors, RippleIMRPhenomD(f_ref=20.0)


@pytest.mark.parametrize("bound", ["f_min", "f_max"])
@pytest.mark.parametrize(
    "mapping", [{"H1": 30.0}, {"H1": 30.0, "L1": 40.0, "L2": 50.0}]
)
def test_frequency_maps_reject_missing_and_unknown_detectors(
    likelihood_inputs, bound, mapping
):
    detectors, waveform = likelihood_inputs
    with pytest.raises(ValueError, match=f"{bound}.*detector names"):
        TransientLikelihoodFD(detectors, waveform, **{bound: mapping})
    assert all(not detector.is_frozen for detector in detectors)


@pytest.mark.parametrize("bad", ["false", "true", 0, 1, [], ()])
def test_phase_marginalization_rejects_wrong_types(likelihood_inputs, bad):
    detectors, waveform = likelihood_inputs
    with pytest.raises(ValueError, match="phase_marginalization must be"):
        TransientLikelihoodFD(detectors, waveform, phase_marginalization=bad)
    assert all(detector.is_frozen for detector in detectors)


@pytest.mark.parametrize("bad", ["false", 0, 1, [], TimeMargConfig()])
def test_heterodyne_phase_marginalization_rejects_wrong_types(likelihood_inputs, bad):
    detectors, waveform = likelihood_inputs
    with pytest.raises(ValueError, match="phase_marginalization must be"):
        HeterodynedTransientLikelihoodFD(detectors, waveform, phase_marginalization=bad)


@pytest.mark.parametrize("option", ["time_marginalization", "phase_marginalization"])
@pytest.mark.parametrize(
    "value, enabled", [(None, False), (False, False), (True, True), ({}, True)]
)
def test_marginalization_valid_sentinels_remain_supported(
    likelihood_inputs, option, value, enabled
):
    detectors, waveform = likelihood_inputs
    likelihood = TransientLikelihoodFD(detectors, waveform, **{option: value})
    assert getattr(likelihood, option) is enabled


@pytest.mark.parametrize("bad", [True, 2.0, 2.5, "2", 0, -1])
def test_heterodyne_rejects_invalid_bin_counts_before_optimizing(
    likelihood_inputs, monkeypatch, bad
):
    def unexpected_optimization(*args, **kwargs):
        pytest.fail("Invalid binning must fail before reference optimization")

    monkeypatch.setattr(
        HeterodynedTransientLikelihoodFD, "maximize_likelihood", unexpected_optimization
    )
    detectors, waveform = likelihood_inputs
    with pytest.raises(ValueError, match="positive integer"):
        HeterodynedTransientLikelihoodFD(
            detectors, waveform, n_bins=bad, prior=UniformPrior(1.0, 2.0, ["M_c"])
        )


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf, True, [1.0], "1.0", 1j])
def test_heterodyne_rejects_invalid_epsilon_before_optimizing(
    likelihood_inputs, monkeypatch, bad
):
    def unexpected_optimization(*args, **kwargs):
        pytest.fail("Invalid binning must fail before reference optimization")

    monkeypatch.setattr(
        HeterodynedTransientLikelihoodFD, "maximize_likelihood", unexpected_optimization
    )
    detectors, waveform = likelihood_inputs
    with pytest.raises(ValueError, match="finite positive number"):
        HeterodynedTransientLikelihoodFD(
            detectors, waveform, epsilon=bad, prior=UniformPrior(1.0, 2.0, ["M_c"])
        )


@pytest.mark.parametrize(
    "field",
    [
        "reference_chirp_mass",
        "accuracy_factor",
        "time_offset",
        "delta_f_end",
        "min_banding_duration",
        "max_banding_frequency",
    ],
)
@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf, True, [1.0], 1j])
def test_multiband_rejects_nonfinite_or_nonscalar_parameters(
    likelihood_inputs, field, bad
):
    detectors, waveform = likelihood_inputs
    kwargs: dict[str, Any] = {
        "reference_chirp_mass": 20.0,
        "time_offset": 0.25,
        "delta_f_end": 53.0,
    }
    kwargs[field] = bad
    with pytest.raises(ValueError, match=field):
        MultibandedTransientLikelihoodFD(detectors, waveform, **kwargs)


@pytest.mark.parametrize("bad", [True, 2.0, 2.5, "2", 0, -1])
def test_multiband_highest_mode_requires_positive_integer(likelihood_inputs, bad):
    detectors, waveform = likelihood_inputs
    with pytest.raises(ValueError, match="highest_mode.*positive integer"):
        MultibandedTransientLikelihoodFD(
            detectors, waveform, reference_chirp_mass=20.0, highest_mode=bad
        )


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_distance_marginalization_reference_distance_must_be_finite(bad):
    with pytest.raises(ValidationError):
        DistanceMargConfig(distance_prior=UniformPrior(1.0, 2.0, ["d_L"]), ref_dist=bad)


@pytest.mark.parametrize(
    "bounds", [(0.1, -0.1), (0.0, 0.0), (jnp.nan, 0.1), (-0.1, jnp.inf)]
)
def test_time_marginalization_config_requires_finite_increasing_bounds(bounds):
    with pytest.raises(ValidationError):
        TimeMargConfig(tc_range=bounds)


def test_distance_marginalization_rejects_unbounded_distance_prior(likelihood_inputs):
    detectors, waveform = likelihood_inputs
    with pytest.raises(ValueError, match="bounds must be finite"):
        TransientLikelihoodFD(
            detectors,
            waveform,
            distance_marginalization={"distance_prior": _UnboundedDistancePrior()},
        )


class _GridDistancePrior(Prior):
    log_density: float
    use_vector: bool
    allow_holes: bool

    def __init__(self, log_density, *, use_vector=False, allow_holes=False):
        super().__init__(["d_L"])
        self.log_density = log_density
        self.use_vector = use_vector
        self.allow_holes = allow_holes

    def get_bounds(self):
        return 1.0, 2.0

    def log_prob(self, z):
        if self.use_vector:
            return jnp.full(2, self.log_density)
        if self.allow_holes:
            return jnp.where(z["d_L"] < 1.5, self.log_density, -jnp.inf)
        return jnp.asarray(self.log_density)

    def sample(self, rng_key, n_samples):
        raise NotImplementedError


class _UnboundedDistancePrior(_GridDistancePrior):
    def __init__(self):
        super().__init__(0.0)

    def get_bounds(self):
        return 1.0, jnp.inf


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_distance_marginalization_rejects_invalid_grid_normalization(
    likelihood_inputs, bad
):
    detectors, waveform = likelihood_inputs
    with pytest.raises(ValueError, match="finite, nonzero probability"):
        TransientLikelihoodFD(
            detectors,
            waveform,
            distance_marginalization={
                "distance_prior": _GridDistancePrior(bad),
                "n_dist_points": 8,
            },
        )


def test_distance_marginalization_requires_scalar_log_density(likelihood_inputs):
    detectors, waveform = likelihood_inputs
    with pytest.raises(ValueError, match="real scalar log probability"):
        TransientLikelihoodFD(
            detectors,
            waveform,
            distance_marginalization={
                "distance_prior": _GridDistancePrior(0.0, use_vector=True),
                "n_dist_points": 8,
            },
        )


def test_distance_marginalization_allows_zero_density_on_part_of_grid(
    likelihood_inputs,
):
    detectors, waveform = likelihood_inputs
    likelihood = TransientLikelihoodFD(
        detectors,
        waveform,
        distance_marginalization={
            "distance_prior": _GridDistancePrior(0.0, allow_holes=True),
            "n_dist_points": 8,
        },
    )
    assert jnp.isclose(jnp.exp(likelihood.log_weights).sum(), 1.0)
    assert jnp.any(jnp.isneginf(likelihood.log_weights))


@pytest.mark.parametrize("log_density", [-1e300, 1e300])
def test_distance_grid_normalization_handles_large_common_log_density(
    likelihood_inputs, log_density
):
    detectors, waveform = likelihood_inputs
    likelihood = TransientLikelihoodFD(
        detectors,
        waveform,
        distance_marginalization={
            "distance_prior": _GridDistancePrior(log_density),
            "n_dist_points": 8,
        },
    )
    assert jnp.allclose(jnp.exp(likelihood.log_weights), jnp.full(8, 1 / 8))
    assert jnp.isclose(
        likelihood._reduce_distance(jnp.asarray(0.0), jnp.asarray(0.0)), 0.0
    )


@pytest.mark.parametrize("field", ["optimizer_popsize", "optimizer_n_steps"])
@pytest.mark.parametrize("bad", [True, 2.0, 2.5, 0, -1, jnp.nan, jnp.inf])
def test_reference_optimizer_requires_positive_integer_counters(field, bad):
    likelihood = object.__new__(HeterodynedTransientLikelihoodFD)
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        likelihood.maximize_likelihood(
            UniformPrior(1.0, 2.0, ["M_c"]), [], **{field: bad}
        )


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf, True, [1.0], "1.0", 1j])
def test_reference_optimizer_requires_finite_scalar_target(bad):
    likelihood = object.__new__(HeterodynedTransientLikelihoodFD)
    with pytest.raises(
        ValueError, match="optimizer_target must be a finite real scalar"
    ):
        likelihood.maximize_likelihood(
            UniformPrior(1.0, 2.0, ["M_c"]), [], optimizer_target=bad
        )


@pytest.mark.slow
def test_heterodyne_accepts_numpy_integer_bin_count(likelihood_inputs):
    detectors, waveform = likelihood_inputs
    bin_count: Any = np.int64(16)
    likelihood = HeterodynedTransientLikelihoodFD(
        detectors,
        waveform,
        n_bins=bin_count,
        reference_parameters=example_params(),
    )
    assert isinstance(likelihood.n_bins, int)
    assert likelihood.n_bins > 0


@pytest.mark.parametrize(
    "likelihood_type",
    [
        TransientLikelihoodFD,
        HeterodynedTransientLikelihoodFD,
        MultibandedTransientLikelihoodFD,
    ],
)
@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, 1j, [0.0], "north", True])
def test_invalid_geometry_fails_before_any_detector_freezes(
    likelihood_inputs, likelihood_type, bad
):
    detectors, waveform = likelihood_inputs
    detectors[-1].longitude = bad
    with pytest.raises(ValueError, match="longitude must be a finite real scalar"):
        likelihood_type(detectors, waveform)
    for detector in detectors:
        assert not detector.is_frozen
        assert not detector.data.is_frozen
        assert not detector.psd.is_frozen


def _set_extreme_weighted_data(detector, frequency):
    frequencies = detector.data.frequencies
    index = int(frequency * detector.data.duration)
    fd = jnp.zeros_like(frequencies, dtype=jnp.complex128)
    fd = fd.at[index].set(1e200 + 1e200j)
    data = Data.from_fd(fd, frequencies, trigger_time=detector.data.trigger_time)
    data.frequency_bounds = (20.0, 100.0)
    detector.set_data(data)
    psd = jnp.ones_like(frequencies).at[index].set(1e-200)
    detector.set_psd(PowerSpectrum(psd, frequencies))


@pytest.mark.parametrize(
    "likelihood_type",
    [
        TransientLikelihoodFD,
        HeterodynedTransientLikelihoodFD,
        MultibandedTransientLikelihoodFD,
    ],
)
def test_psd_weighted_data_overflow_fails_before_any_detector_freezes(
    likelihood_inputs, likelihood_type
):
    detectors, waveform = likelihood_inputs
    _set_extreme_weighted_data(detectors[-1], 40.0)
    with pytest.raises(ValueError, match="PSD-weighted data.*must be finite"):
        likelihood_type(detectors, waveform)
    assert all(not detector.is_frozen for detector in detectors)


def test_psd_weighted_data_overflow_outside_the_frequency_range_is_ignored(
    likelihood_inputs,
):
    detectors, waveform = likelihood_inputs
    _set_extreme_weighted_data(detectors[-1], 120.0)
    likelihood = TransientLikelihoodFD(detectors, waveform)
    assert jnp.isfinite(likelihood.evaluate(example_params()))


# Long enough (about 100 s above 10 Hz) for the rotation to show in the likelihood.
LONG_SIGNAL_PARAMS = {
    "M_c": 5.0,
    "eta": 0.24,
    "s1_z": 0.1,
    "s2_z": -0.1,
    "d_L": 200.0,
    "phase_c": 0.4,
    "iota": 0.5,
    "ra": 1.1,
    "dec": -0.4,
    "psi": 0.7,
    "t_c": 0.0,
}


class TestGroundBased3GDetectors:
    """Likelihoods with GroundBased3G detectors, whose response follows the Earth's rotation."""

    duration = 128.0
    sampling_frequency = 512.0
    f_min, f_max = 10.0, 200.0
    trigger_time = 1400000000.0

    def _injected(self, det):
        """Inject the long signal into zero noise; return the detector and its optimal SNR."""
        frequencies = jnp.linspace(0.0, self.sampling_frequency / 2, 1025)
        det.set_psd(PowerSpectrum(jnp.full(1025, 1e-46), frequencies))
        det.set_zero_noise(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            trigger_time=self.trigger_time,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        optimal_snr, _ = det.inject_signal(
            RippleIMRPhenomD(f_ref=20.0), LONG_SIGNAL_PARAMS
        )
        return det, float(optimal_snr)

    @staticmethod
    def _static_copy(det):
        """A GroundBased3G without rotation, carrying the data and PSD of *det*."""
        static = GroundBased3G.from_detector(det, earth_rotation=False)
        static.set_psd(det.psd)
        static.set_data(det.data)
        return static

    def _likelihood(self, dets, kind="full"):
        waveform = RippleIMRPhenomD(f_ref=20.0)
        common = {"f_min": self.f_min, "f_max": self.f_max}
        if kind == "full":
            return TransientLikelihoodFD(dets, waveform, **common)
        if kind == "time":
            return TransientLikelihoodFD(
                dets, waveform, time_marginalization={"tc_range": (-0.1, 0.1)}, **common
            )
        if kind == "heterodyne":
            return HeterodynedTransientLikelihoodFD(
                dets,
                waveform,
                reference_parameters=dict(LONG_SIGNAL_PARAMS),
                n_bins=100,
                **common,
            )
        prior = CombinePrior([UniformPrior(-0.1, 0.1, parameter_names=["t_c"])])
        return MultibandedTransientLikelihoodFD(
            dets,
            waveform,
            prior=prior,
            reference_chirp_mass=LONG_SIGNAL_PARAMS["M_c"],
            **common,
        )

    def _log_likelihood(self, det, kind="full"):
        return float(self._likelihood([det], kind).evaluate(dict(LONG_SIGNAL_PARAMS)))

    @pytest.mark.slow
    def test_rotating_template_recovers_the_optimal_log_likelihood(self):
        """A rotating injection is recovered exactly by the rotating response only."""
        rotating, snr = self._injected(get_ET(earth_rotation=True)[0])
        static = self._static_copy(rotating)
        optimal = snr**2 / 2

        assert self._log_likelihood(rotating) == pytest.approx(optimal, rel=1e-10)
        assert optimal - self._log_likelihood(static) > 1e-3

    def test_et_triangle_recovers_the_optimal_log_likelihood(self):
        """All three ET detectors together, each with its own rotating response."""
        injected = [self._injected(det) for det in get_ET(earth_rotation=True)]
        et = [det for det, _ in injected]
        optimal = sum(snr**2 for _, snr in injected) / 2
        log_likelihood = float(self._likelihood(et).evaluate(dict(LONG_SIGNAL_PARAMS)))
        assert log_likelihood == pytest.approx(optimal, rel=1e-10)

    @pytest.mark.parametrize(
        "kind",
        [
            "time",
            pytest.param("heterodyne", marks=pytest.mark.slow),
            pytest.param("multiband", marks=pytest.mark.slow),
        ],
    )
    def test_other_likelihoods_follow_the_rotation(self, kind):
        """Switching the rotation off moves these likelihoods as it moves the full one."""
        rotating, _ = self._injected(get_ET(earth_rotation=True)[0])
        static = self._static_copy(rotating)

        def shift(kind):
            return self._log_likelihood(rotating, kind) - self._log_likelihood(
                static, kind
            )

        full = shift("full")
        assert full > 1e-3
        assert shift(kind) == pytest.approx(full, abs=1e-4)


class TestMarginalizationAssumptions:
    """The log-likelihood is wrong, silently, where the assumptions of a marginalization break."""

    @pytest.fixture
    def warnings(self, caplog, monkeypatch):
        # The "jimgw" logger does not propagate to the root logger, where caplog listens.
        monkeypatch.setattr(logging.getLogger("jimgw"), "propagate", True)
        with caplog.at_level("WARNING"):
            yield lambda: " ".join(record.message for record in caplog.records)

    @pytest.mark.parametrize(
        "make_waveform, reason",
        [
            (RippleIMRPhenomHM, "modes up to m = 4"),
            (RippleIMRPhenomPv2, "precession"),
            (RippleIMRPhenomXPHM, "precession and modes up to m = 4"),
        ],
    )
    def test_phase_marginalization_warns_if_the_signal_is_not_the_22_mode_alone(
        self, detectors_and_waveform, warnings, make_waveform, reason
    ):
        ifos, _waveform, fmin, fmax, _gps = detectors_and_waveform
        TransientLikelihoodFD(
            ifos,
            make_waveform(f_ref=20.0),
            f_min=fmin,
            f_max=fmax,
            phase_marginalization=True,
        )
        assert f"has {reason}, so the marginalized likelihood may be inaccurate" in (
            warnings()
        )

    def test_phase_marginalization_is_silent_for_the_22_mode_alone(
        self, detectors_and_waveform, warnings
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        TransientLikelihoodFD(
            ifos, waveform, f_min=fmin, f_max=fmax, phase_marginalization=True
        )
        assert "phase_marginalization assumes" not in warnings()

    def test_the_warning_needs_phase_marginalization(
        self, detectors_and_waveform, warnings
    ):
        ifos, _waveform, fmin, fmax, _gps = detectors_and_waveform
        TransientLikelihoodFD(
            ifos, RippleIMRPhenomHM(f_ref=20.0), f_min=fmin, f_max=fmax
        )
        assert "phase_marginalization assumes" not in warnings()

    @pytest.mark.slow
    def test_the_heterodyned_likelihood_warns_too(
        self, detectors_and_waveform, warnings
    ):
        ifos, _waveform, fmin, fmax, _gps = detectors_and_waveform
        HeterodynedTransientLikelihoodFD(
            ifos,
            RippleIMRPhenomHM(f_ref=20.0),
            f_min=fmin,
            f_max=fmax,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        assert "has modes up to m = 4" in warnings()

    def test_time_marginalization_warns_if_tc_range_exceeds_the_time_grid(
        self, detectors_and_waveform, warnings
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        duration = float(ifos[0].data.duration)
        # Twice as wide as the time grid, which spans the duration of the data.
        TransientLikelihoodFD(
            ifos,
            waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={"tc_range": (-duration, duration)},
        )
        assert "extends beyond the time grid of the data" in warnings()
        assert "may be inaccurate" in warnings()

    def test_time_marginalization_is_silent_inside_the_time_grid(
        self, detectors_and_waveform, warnings
    ):
        ifos, waveform, fmin, fmax, _gps = detectors_and_waveform
        TransientLikelihoodFD(
            ifos,
            waveform,
            f_min=fmin,
            f_max=fmax,
            time_marginalization={"tc_range": (-0.1, 0.1)},
        )
        assert "extends beyond the time grid" not in warnings()
