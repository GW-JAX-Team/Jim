"""Tests for the default sample/likelihood transforms, the periodic list and to_likelihood_space."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    GaussianPrior,
    LogisticDistribution,
    PowerLawPrior,
    RayleighPrior,
    SinePrior,
    UniformPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.detector import get_detector_preset
from jimgw.core.single_event.helpers import (
    get_default_likelihood_transforms,
    get_default_periodic,
    get_default_sample_transforms,
    to_likelihood_space,
)
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    SpinAnglesToCartesianSpinTransform,
)
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.transforms import BoundToBound
from tests.utils import attach_data

TRIGGER_TIME = 1126259462.4
PI = float(jnp.pi)

J_FRAME = ["theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2"]
GW150914 = [
    "M_c",
    "q",
    "s1_z",
    "s2_z",
    "iota",
    "d_L",
    "t_c",
    "phase_c",
    "psi",
    "ra",
    "dec",
]


@pytest.fixture(scope="module")
def ifos():
    preset = get_detector_preset()
    detectors = [preset["H1"], preset["L1"]]
    attach_data(TRIGGER_TIME, detectors)
    return detectors


@pytest.fixture(scope="module")
def waveform():
    return RippleIMRPhenomD(f_ref=20.0)


def prior_of(*names):
    """A prior with one unit-interval uniform per name; enough where only names matter."""
    return CombinePrior([UniformPrior(0.0, 1.0, [name]) for name in names])


def type_names(transforms):
    return [type(t).__name__ for t in transforms]


# ----------------------------------------------------------------------
# Sample transforms
# ----------------------------------------------------------------------


class TestSampleTransforms:
    def test_the_gw150914_default_is_a_time_and_a_sky_transform(self, ifos):
        transforms = get_default_sample_transforms(prior_of(*GW150914), ifos)
        assert type_names(transforms) == [
            "GeocentricArrivalTimeToDetectorArrivalTimeTransform",
            "SkyFrameToDetectorFrameSkyPositionTransform",
        ]

    def test_geocentric_sky_has_no_sky_transform(self, ifos):
        transforms = get_default_sample_transforms(
            prior_of(*GW150914), ifos, detector_frame_sky=False
        )
        assert type_names(transforms) == [
            "GeocentricArrivalTimeToDetectorArrivalTimeTransform"
        ]

    def test_geocentric_time_has_no_time_transform(self, ifos):
        transforms = get_default_sample_transforms(
            prior_of(*GW150914), ifos, detector_frame_time=False
        )
        assert type_names(transforms) == ["SkyFrameToDetectorFrameSkyPositionTransform"]

    def test_a_prior_already_in_the_detector_frame_needs_no_transform(self, ifos):
        names = [n for n in GW150914 if n not in ("t_c", "ra", "dec")]
        prior = prior_of(*names, "t_det", "azimuth", "zenith")
        assert get_default_sample_transforms(prior, ifos) == []

    def test_the_time_is_the_arrival_time_at_the_first_detector(self, ifos):
        prior = CombinePrior(
            [
                UniformPrior(-0.1, 0.1, ["t_c"]),
                UniformPrior(0.0, 2 * PI, ["ra"]),
                CosinePrior(["dec"]),
            ]
        )
        (time_transform, _) = get_default_sample_transforms(prior, ifos)
        point = {"t_c": 0.02, "ra": 1.3, "dec": 0.4}
        expected = GeocentricArrivalTimeToDetectorArrivalTimeTransform(ifos[0]).forward(
            point
        )
        assert float(time_transform.forward(point)["t_det"]) == pytest.approx(
            float(expected["t_det"])
        )

    def test_one_detector_samples_ra_dec_directly(self, ifos):
        prior = prior_of("ra", "dec")
        with pytest.raises(ValueError, match="two detectors"):
            get_default_sample_transforms(prior, ifos[:1])
        assert (
            get_default_sample_transforms(prior, ifos[:1], detector_frame_sky=False)
            == []
        )

    def test_the_sky_frame_is_built_from_the_given_detectors(self, ifos):
        transforms = get_default_sample_transforms(prior_of("ra", "dec"), ifos)
        point = {"ra": 1.3, "dec": 0.4}
        expected = SkyFrameToDetectorFrameSkyPositionTransform(ifos).forward(point)
        got = transforms[0].forward(point)
        assert float(got["azimuth"]) == pytest.approx(float(expected["azimuth"]))
        assert float(got["zenith"]) == pytest.approx(float(expected["zenith"]))


# ----------------------------------------------------------------------
# Unit cube (NS AW)
# ----------------------------------------------------------------------

PRIORS_WITH_A_UNIT_CUBE = {
    "uniform": UniformPrior(-3.0, 5.0, ["x"]),
    "sine": SinePrior(["x"]),
    "cosine": CosinePrior(["x"]),
    "power_law": PowerLawPrior(1.0, 2000.0, 2.0, ["x"]),
    "rayleigh": RayleighPrior(0.5, ["x"]),
    "gaussian": GaussianPrior(1.0, 2.0, ["x"]),
}


class TestUnitCube:
    @pytest.mark.parametrize("name", PRIORS_WITH_A_UNIT_CUBE)
    def test_every_prior_type_is_mapped_to_a_uniform_unit_interval(self, ifos, name):
        prior = PRIORS_WITH_A_UNIT_CUBE[name]
        transforms = get_default_sample_transforms(prior, ifos, unit_cube=True)
        samples = prior.sample(jax.random.key(0), 20000)
        for transform in transforms:
            samples = jax.vmap(transform.forward)(samples)
        (unit,) = [np.asarray(v) for v in samples.values()]
        assert unit.min() >= 0.0 and unit.max() <= 1.0
        assert unit.mean() == pytest.approx(0.5, abs=0.01)
        assert unit.var() == pytest.approx(1 / 12, abs=0.005)

    def test_sampling_parameters_of_the_ns_aw_example_prior(self, ifos):
        prior = CombinePrior(
            [
                UniformPrior(20.0, 40.0, ["M_c"]),
                UniformPrior(0.125, 1.0, ["q"]),
                UniformPrior(-0.99, 0.99, ["s1_z"]),
                UniformPrior(-0.99, 0.99, ["s2_z"]),
                SinePrior(["iota"]),
                PowerLawPrior(1.0, 2000.0, 2.0, ["d_L"]),
                UniformPrior(-0.1, 0.1, ["t_det"]),
                UniformPrior(0.0, PI, ["psi"]),
                UniformPrior(0.0, 2 * PI, ["ra"]),
                CosinePrior(["dec"]),
            ]
        )
        transforms = get_default_sample_transforms(prior, ifos, unit_cube=True)
        names = tuple(prior.parameter_names)
        for transform in transforms:
            names = transform.propagate_name(names)
        assert sorted(names) == sorted(
            [
                "M_c_unit",
                "q_unit",
                "s1_z_unit",
                "s2_z_unit",
                "cos_iota_unit",
                "d_L_unit",
                "t_det_unit",
                "psi_unit",
                "azimuth_unit",
                "cos_zenith_unit",
            ]
        )
        samples = prior.sample(jax.random.key(1), 1000)
        for transform in transforms:
            samples = jax.vmap(transform.forward)(samples)
        for value in samples.values():
            assert float(value.min()) >= 0.0 and float(value.max()) <= 1.0

    def test_a_single_component_prior_is_accepted(self, ifos):
        transforms = get_default_sample_transforms(
            SinePrior(["iota"]), ifos, unit_cube=True
        )
        assert type_names(transforms) == ["CosineTransform", "BoundToBound"]

    def test_the_helper_priors_inside_a_component_are_not_mapped(self, ifos):
        # SinePrior's own base_prior is a UniformPrior on "cos(iota)"; it must be left alone.
        prior = CombinePrior([UniformPrior(0.0, 1.0, ["x"]), SinePrior(["iota"])])
        transforms = get_default_sample_transforms(prior, ifos, unit_cube=True)
        assert type_names(transforms) == [
            "BoundToBound",
            "CosineTransform",
            "BoundToBound",
        ]

    def test_a_uniform_sphere_expands_to_magnitude_polar_and_azimuthal_angle(
        self, ifos
    ):
        prior = CombinePrior([UniformSpherePrior(["s1"])])
        transforms = get_default_sample_transforms(prior, ifos, unit_cube=True)
        names = tuple(prior.parameter_names)
        for transform in transforms:
            names = transform.propagate_name(names)
        assert sorted(names) == ["cos_s1_theta_unit", "s1_mag_unit", "s1_phi_unit"]

    def test_geocentric_ra_dec_get_their_own_unit_cube(self, ifos):
        prior = CombinePrior([UniformPrior(0.0, 2 * PI, ["ra"]), CosinePrior(["dec"])])
        transforms = get_default_sample_transforms(
            prior, ifos, detector_frame_sky=False, unit_cube=True
        )
        names = tuple(prior.parameter_names)
        for transform in transforms:
            names = transform.propagate_name(names)
        assert sorted(names) == ["ra_unit", "sin_dec_unit"]

    def test_a_detector_frame_t_c_cannot_have_exact_unit_cube_bounds(self, ifos):
        prior = CombinePrior(
            [
                UniformPrior(-0.1, 0.1, ["t_c"]),
                UniformPrior(0.0, 2 * PI, ["ra"]),
                CosinePrior(["dec"]),
            ]
        )
        with pytest.raises(ValueError, match="t_det"):
            get_default_sample_transforms(prior, ifos, unit_cube=True)
        # Sampling t_c itself is exact, so that is accepted.
        get_default_sample_transforms(
            prior, ifos, detector_frame_time=False, unit_cube=True
        )

    def test_a_prior_without_a_default_unit_cube_is_refused(self, ifos):
        with pytest.raises(TypeError, match="LogisticDistribution"):
            get_default_sample_transforms(
                CombinePrior([LogisticDistribution(["x"])]), ifos, unit_cube=True
            )

    @pytest.mark.parametrize(
        "ra, dec",
        [
            (UniformPrior(1.0, 2.0, ["ra"]), CosinePrior(["dec"])),
            (UniformPrior(0.0, 2 * PI, ["ra"]), UniformPrior(-0.5, 0.5, ["dec"])),
        ],
        ids=["restricted_ra", "uniform_dec"],
    )
    def test_a_non_isotropic_sky_prior_cannot_use_the_detector_frame_sky(
        self, ifos, ra, dec
    ):
        # azimuth and cos(zenith) would not be uniform on [0, 1] for such a prior.
        prior = CombinePrior([ra, dec])
        with pytest.raises(ValueError, match="isotropic"):
            get_default_sample_transforms(prior, ifos, unit_cube=True)
        # Mapping ra/dec directly works for any of the supported priors.
        get_default_sample_transforms(
            prior, ifos, detector_frame_sky=False, unit_cube=True
        )


# ----------------------------------------------------------------------
# Likelihood transforms
# ----------------------------------------------------------------------


class TestLikelihoodTransforms:
    def test_the_gw150914_default_is_only_q_to_eta(self, ifos, waveform):
        transforms = get_default_likelihood_transforms(
            prior_of(*GW150914), ifos, waveform
        )
        assert len(transforms) == 1
        assert transforms[0].name_mapping == (["q"], ["eta"])

    def test_a_detector_frame_sky_prior_is_taken_back_to_ra_dec(self, ifos, waveform):
        transforms = get_default_likelihood_transforms(
            prior_of("M_c", "q", "azimuth", "zenith"), ifos, waveform
        )
        mappings = [
            (set(t.name_mapping[0]), set(t.name_mapping[1])) for t in transforms
        ]
        assert ({"azimuth", "zenith"}, {"ra", "dec"}) in mappings

    def test_a_detector_frame_time_prior_is_taken_back_to_t_c(self, ifos, waveform):
        transforms = get_default_likelihood_transforms(
            prior_of("M_c", "q", "ra", "dec", "t_det"), ifos, waveform
        )
        mappings = [t.name_mapping for t in transforms]
        assert (["t_det"], ["t_c"]) in mappings

    def test_the_time_is_taken_back_at_the_first_detector(self, ifos, waveform):
        transforms = get_default_likelihood_transforms(
            prior_of("ra", "dec", "t_det"), ifos, waveform
        )
        point = {"ra": 1.3, "dec": 0.4, "t_det": 0.01}
        expected = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            ifos[0]
        ).backward(point)
        for transform in transforms:
            point = transform.forward(point)
        assert float(point["t_c"]) == pytest.approx(float(expected["t_c"]))

    def test_j_frame_angles_use_the_waveform_reference_frequency(self, ifos):
        transforms = get_default_likelihood_transforms(
            prior_of("M_c", "q", "phase_c", *J_FRAME),
            ifos,
            RippleIMRPhenomD(f_ref=30.0),
        )
        (spin,) = [
            t for t in transforms if isinstance(t, SpinAnglesToCartesianSpinTransform)
        ]
        assert spin.freq_ref == 30.0
        assert "phase_c" in spin.conditional_names

    def test_phase_marginalization_converts_the_spins_at_zero_phase(
        self, ifos, waveform
    ):
        transforms = get_default_likelihood_transforms(
            prior_of("M_c", "q", *J_FRAME), ifos, waveform, phase_marginalization=True
        )
        (spin,) = [
            t for t in transforms if isinstance(t, SpinAnglesToCartesianSpinTransform)
        ]
        assert "phase_c" not in spin.conditional_names

    def test_spherical_spins_get_one_transform_each(self, ifos, waveform):
        names = [f"s{i}_{s}" for i in (1, 2) for s in ("mag", "theta", "phi")]
        transforms = get_default_likelihood_transforms(
            prior_of("M_c", "q", *names), ifos, waveform
        )
        assert (
            sum(isinstance(t, SphereSpinToCartesianSpinTransform) for t in transforms)
            == 2
        )

    def test_cartesian_spins_need_no_transform(self, ifos, waveform):
        names = ["s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z"]
        transforms = get_default_likelihood_transforms(
            prior_of("M_c", "q", *names), ifos, waveform
        )
        assert [t.name_mapping for t in transforms] == [(["q"], ["eta"])]

    def test_a_partial_j_frame_set_gets_no_spin_transform(self, ifos, waveform):
        # Same rule as to_likelihood_space: all seven angles or none.
        transforms = get_default_likelihood_transforms(
            prior_of("M_c", "q", "theta_jn", "a_1"), ifos, waveform
        )
        assert [t.name_mapping for t in transforms] == [(["q"], ["eta"])]

    def test_the_spin_transform_still_finds_q(self, ifos, waveform):
        # q is replaced by eta, so that has to come after the transform that reads q.
        prior = CombinePrior(
            [
                UniformPrior(20.0, 40.0, ["M_c"]),
                UniformPrior(0.125, 1.0, ["q"]),
                UniformPrior(0.0, 2 * PI, ["phase_c"]),
                SinePrior(["theta_jn"]),
                UniformPrior(0.0, 2 * PI, ["phi_jl"]),
                SinePrior(["tilt_1"]),
                SinePrior(["tilt_2"]),
                UniformPrior(0.0, 2 * PI, ["phi_12"]),
                UniformPrior(0.0, 0.9, ["a_1"]),
                UniformPrior(0.0, 0.9, ["a_2"]),
            ]
        )
        transforms = get_default_likelihood_transforms(prior, ifos, waveform)
        point = {k: v[0] for k, v in prior.sample(jax.random.key(0), 1).items()}
        for transform in transforms:
            point = transform.forward(point)
        assert {"eta", "iota", "s1_x", "s2_z"} <= point.keys()
        assert not {"q", "theta_jn"} & point.keys()

    @pytest.mark.parametrize(
        "prior",
        [
            CombinePrior(
                [UniformPrior(20.0, 40.0, ["M_c"]), UniformPrior(0.125, 1.0, ["q"])]
                + [UniformPrior(0.0, 1.0, [n]) for n in ("t_det", "ra", "dec")]
            ),
            CombinePrior(
                [UniformPrior(20.0, 40.0, ["M_c"]), UniformPrior(0.125, 1.0, ["q"])]
                + [UniformPrior(0.0, 1.0, [n]) for n in ("azimuth", "zenith")]
            ),
            CombinePrior(
                [UniformPrior(20.0, 40.0, ["M_c"]), UniformPrior(0.125, 1.0, ["q"])]
                + [UniformSpherePrior(["s1"]), UniformSpherePrior(["s2"])]
            ),
        ],
        ids=["t_det", "azimuth_zenith", "sphere_spins"],
    )
    def test_the_transforms_agree_with_to_likelihood_space(self, ifos, waveform, prior):
        transforms = get_default_likelihood_transforms(prior, ifos, waveform)
        point = {k: v[0] for k, v in prior.sample(jax.random.key(3), 1).items()}
        expected = to_likelihood_space(
            {k: float(v) for k, v in point.items()}, ifos, waveform
        )
        for transform in transforms:
            point = transform.forward(point)
        assert point.keys() == expected.keys()
        for key, value in expected.items():
            assert float(point[key]) == pytest.approx(value)


# ----------------------------------------------------------------------
# Periodic parameters
# ----------------------------------------------------------------------


def gw_prior(**replace):
    components = {
        "M_c": UniformPrior(20.0, 40.0, ["M_c"]),
        "t_c": UniformPrior(-0.1, 0.1, ["t_c"]),
        "psi": UniformPrior(0.0, PI, ["psi"]),
        "ra": UniformPrior(0.0, 2 * PI, ["ra"]),
        "dec": CosinePrior(["dec"]),
    }
    components.update(replace)
    return CombinePrior([c for c in components.values() if c is not None])


class TestPeriodic:
    def test_psi_and_azimuth_wrap_in_the_detector_frame(self, ifos):
        prior = gw_prior()
        transforms = get_default_sample_transforms(prior, ifos)
        assert get_default_periodic(prior, transforms) == {
            "psi": (0.0, PI),
            "azimuth": (0.0, 2 * PI),
        }

    def test_ra_wraps_when_the_sky_is_sampled_in_ra_dec(self, ifos):
        prior = gw_prior()
        transforms = get_default_sample_transforms(
            prior, ifos, detector_frame_sky=False
        )
        assert get_default_periodic(prior, transforms) == {
            "psi": (0.0, PI),
            "ra": (0.0, 2 * PI),
        }

    def test_a_range_shorter_than_the_period_does_not_wrap(self, ifos):
        prior = gw_prior(ra=UniformPrior(1.0, 2.0, ["ra"]))
        transforms = get_default_sample_transforms(
            prior, ifos, detector_frame_sky=False
        )
        assert get_default_periodic(prior, transforms) == {"psi": (0.0, PI)}

    def test_a_range_of_one_period_anywhere_wraps_at_its_own_bounds(self, ifos):
        prior = gw_prior(ra=UniformPrior(-PI, PI, ["ra"]))
        transforms = get_default_sample_transforms(
            prior, ifos, detector_frame_sky=False
        )
        assert get_default_periodic(prior, transforms)["ra"] == (-PI, PI)

    def test_only_a_uniform_prior_counts(self, ifos):
        prior = gw_prior(psi=GaussianPrior(1.5, 0.3, ["psi"]))
        transforms = get_default_sample_transforms(prior, ifos)
        assert "psi" not in get_default_periodic(prior, transforms)

    def test_phase_and_spherical_spin_angles_wrap(self, ifos):
        prior = CombinePrior(
            [
                UniformPrior(0.0, 2 * PI, ["phase_c"]),
                UniformSpherePrior(["s1"]),
                UniformPrior(0.0, 1.0, ["x"]),
            ]
        )
        assert get_default_periodic(prior, []) == {
            "phase_c": (0.0, 2 * PI),
            "s1_phi": (0.0, 2 * PI),
        }

    def test_the_unit_cube_gives_the_names_of_the_unit_parameters(self, ifos):
        prior = gw_prior(t_c=None, t_det=UniformPrior(-0.1, 0.1, ["t_det"]))
        transforms = get_default_sample_transforms(prior, ifos, unit_cube=True)
        assert get_default_periodic(prior, transforms) == [
            "psi_unit",
            "azimuth_unit",
        ]

    def test_the_unit_cube_wraps_ra_when_the_sky_is_sampled_in_ra_dec(self, ifos):
        prior = gw_prior(t_c=None)
        transforms = get_default_sample_transforms(
            prior, ifos, detector_frame_sky=False, unit_cube=True
        )
        assert get_default_periodic(prior, transforms) == [
            "psi_unit",
            "ra_unit",
        ]

    def test_a_hand_written_unit_cube_list_is_read_as_given(self):
        transforms = [
            BoundToBound(
                name_mapping=(["psi"], ["psi_unit"]),
                original_lower_bound=0.0,
                original_upper_bound=PI,
                target_lower_bound=0.0,
                target_upper_bound=1.0,
            ),
            BoundToBound(
                name_mapping=(["ra"], ["ra_unit"]),
                original_lower_bound=1.0,
                original_upper_bound=2.0,
                target_lower_bound=0.0,
                target_upper_bound=1.0,
            ),
        ]
        prior = prior_of("psi", "ra")
        assert get_default_periodic(prior, transforms) == ["psi_unit"]

    def test_every_mapping_of_a_multi_parameter_bound_to_bound_is_read(self):
        transforms = [
            BoundToBound(
                name_mapping=(["psi", "ra"], ["psi_unit", "ra_unit"]),
                original_lower_bound=jnp.array([0.0, 0.0]),
                original_upper_bound=jnp.array([PI, 2 * PI]),
                target_lower_bound=jnp.array([0.0, 0.0]),
                target_upper_bound=jnp.array([1.0, 1.0]),
            )
        ]
        prior = prior_of("psi", "ra")
        assert get_default_periodic(prior, transforms) == [
            "psi_unit",
            "ra_unit",
        ]

    def test_a_parameter_made_by_a_transform_wraps_only_if_it_is_azimuth(self):
        # Nothing says the range of "psi" made by a transform is one period.
        transform = BoundToBound(
            name_mapping=(["x"], ["psi"]),
            original_lower_bound=0.0,
            original_upper_bound=1.0,
            target_lower_bound=0.0,
            target_upper_bound=0.5,
        )
        assert get_default_periodic(prior_of("x"), [transform]) == {}

    def test_the_bounds_are_plain_floats(self, ifos):
        prior = gw_prior()
        bounds = get_default_periodic(prior, get_default_sample_transforms(prior, ifos))
        assert all(type(v) is float for pair in bounds.values() for v in pair)


# ----------------------------------------------------------------------
# to_likelihood_space
# ----------------------------------------------------------------------


class TestToLikelihoodSpace:
    def test_q_becomes_eta(self, ifos, waveform):
        p = to_likelihood_space({"M_c": 30.0, "q": 0.5}, ifos, waveform)
        assert p["eta"] == pytest.approx(0.5 / 1.5**2)
        assert "q" not in p

    def test_eta_is_left_alone(self, ifos, waveform):
        p = to_likelihood_space({"M_c": 30.0, "eta": 0.2}, ifos, waveform)
        assert p == {"M_c": 30.0, "eta": 0.2}

    def test_azimuth_zenith_and_t_det_become_ra_dec_and_t_c(self, ifos, waveform):
        ra, dec, t_c = 1.3, 0.4, 0.02
        detector_frame = SkyFrameToDetectorFrameSkyPositionTransform(ifos).forward(
            {"ra": ra, "dec": dec}
        )
        t_det = GeocentricArrivalTimeToDetectorArrivalTimeTransform(ifos[0]).forward(
            {"t_c": t_c, "ra": ra, "dec": dec}
        )["t_det"]
        p = to_likelihood_space(
            {
                "azimuth": float(detector_frame["azimuth"]),
                "zenith": float(detector_frame["zenith"]),
                "t_det": float(t_det),
            },
            ifos,
            waveform,
        )
        assert p["ra"] == pytest.approx(ra)
        assert p["dec"] == pytest.approx(dec)
        assert p["t_c"] == pytest.approx(t_c)

    def test_j_frame_angles_become_cartesian_spins_at_the_waveform_f_ref(self, ifos):
        params = dict(
            zip(J_FRAME, [0.7, 1.1, 0.5, 1.2, 2.0, 0.6, 0.3]),
            M_c=30.0,
            q=0.8,
            phase_c=0.4,
        )
        waveform = RippleIMRPhenomD(f_ref=25.0)
        expected = SpinAnglesToCartesianSpinTransform(freq_ref=25.0).forward(params)
        p = to_likelihood_space(params, ifos, waveform)
        assert p["iota"] == pytest.approx(float(expected["iota"]))
        assert p["s1_x"] == pytest.approx(float(expected["s1_x"]))
        assert not set(J_FRAME) & p.keys()

    def test_spherical_spins_become_cartesian_spins(self, ifos, waveform):
        params = {"s1_mag": 0.5, "s1_theta": 1.0, "s1_phi": 2.0}
        p = to_likelihood_space(params, ifos, waveform)
        assert p["s1_z"] == pytest.approx(0.5 * np.cos(1.0))
        assert p["s1_x"] == pytest.approx(0.5 * np.sin(1.0) * np.cos(2.0))
