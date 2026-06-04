import jax
import jax.numpy as jnp
import numpy as np
from itertools import combinations

from jimgw.core.single_event.transforms import (
    DistanceToSNRWeightedDistanceTransform,
    SphereSpinToCartesianSpinTransform,
    SpinAnglesToCartesianSpinTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    ComponentMassesToChirpMassMassRatioTransform,
    ComponentMassesToChirpMassSymmetricMassRatioTransform,
    MassRatioToSymmetricMassRatioTransform,
    ChirpMassMassRatioToComponentMassesTransform,
    ChirpMassSymmetricMassRatioToComponentMassesTransform,
    SymmetricMassRatioToMassRatioTransform,
)
from jimgw.core.single_event.detector import get_detector_preset
from tests.utils import assert_all_finite, common_keys_allclose

detector_preset = get_detector_preset()
H1 = detector_preset["H1"]
L1 = detector_preset["L1"]
V1 = detector_preset["V1"]


class TestDistanceTransform:
    def test_forward_distance_transform(self):
        """
        Test transformation from distance to SNR-weighted distance (boundaries excluded)
        """
        output, jacobian = DistanceToSNRWeightedDistanceTransform(
            trigger_time=1126259462.4,
            ifos=[H1, L1],
        ).transform(
            {
                "d_L": 200.0,
                "M_c": 30.0,
                "ra": 1.0,
                "dec": 0.0,
                "psi": 0.5,
                "iota": 0.6,
            }
        )

        assert np.isfinite(output["d_hat"])
        assert_all_finite(jacobian)

    def test_backward_distance_transform(self):
        """
        Test transformation from SNR-weighted distance to distance (boundaries excluded)
        """
        output, jacobian = DistanceToSNRWeightedDistanceTransform(
            trigger_time=1126259462.4,
            ifos=[H1, L1],
        ).inverse(
            {
                "d_hat": 100.0,
                "M_c": 30.0,
                "ra": 1.0,
                "dec": 0.0,
                "psi": 0.5,
                "iota": 0.6,
            }
        )
        assert np.isfinite(output["d_L"])
        assert_all_finite(jacobian)

    def test_forward_backward_consistency(self):
        """
        Test that the forward and inverse transformations are consistent
        """

        key = jax.random.key(42)
        key, *subkeys = jax.random.split(key, 7)
        dL = jax.random.uniform(subkeys[0], (10,), minval=1, maxval=2000)
        M_c = jax.random.uniform(subkeys[1], (10,), minval=1, maxval=100)
        ra = jax.random.uniform(subkeys[2], (10,), minval=0, maxval=2 * jnp.pi)
        dec = jax.random.uniform(
            subkeys[3], (10,), minval=-jnp.pi / 2, maxval=jnp.pi / 2
        )
        psi = jax.random.uniform(subkeys[4], (10,), minval=0, maxval=jnp.pi)
        iota = jax.random.uniform(subkeys[5], (10,), minval=0, maxval=jnp.pi)

        inputs = jnp.stack([dL, M_c, ra, dec, psi, iota], axis=-1).T
        param_name = ["d_L", "M_c", "ra", "dec", "psi", "iota"]
        inputs = dict(zip(param_name, inputs))
        distance_transform = DistanceToSNRWeightedDistanceTransform(
            trigger_time=1126259462.4,
            ifos=[H1, L1],
        )
        forward_transform_output = jax.vmap(distance_transform.forward)(inputs)
        output = jax.vmap(distance_transform.backward)(forward_transform_output)
        assert np.allclose(output["d_L"], dL)

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """

        subkeys = jax.random.split(jax.random.key(12), 6)
        dL = jax.random.uniform(subkeys[0], (1,), minval=1, maxval=2000)
        M_c = jax.random.uniform(subkeys[1], (1,), minval=1, maxval=100)
        ra = jax.random.uniform(subkeys[2], (1,), minval=0, maxval=2 * jnp.pi)
        dec = jax.random.uniform(
            subkeys[3], (1,), minval=-jnp.pi / 2, maxval=jnp.pi / 2
        )
        psi = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=jnp.pi)
        iota = jax.random.uniform(subkeys[5], (1,), minval=0, maxval=jnp.pi)

        sample = [
            dL[0],
            M_c[0],
            ra[0],
            dec[0],
            psi[0],
            iota[0],
        ]
        sample_dict = dict(zip(["d_L", "M_c", "ra", "dec", "psi", "iota"], sample))

        jit_transform = jax.jit(
            DistanceToSNRWeightedDistanceTransform(
                trigger_time=1126259462.4,
                ifos=[H1, L1],
            ).transform
        )
        jitted_output, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_output = DistanceToSNRWeightedDistanceTransform(
            trigger_time=1126259462.4,
            ifos=[H1, L1],
        ).forward(sample_dict)

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """

        subkeys = jax.random.split(jax.random.key(123), 6)
        d_hat = jax.random.uniform(subkeys[0], (1,), minval=1, maxval=100000)
        M_c = jax.random.uniform(subkeys[1], (1,), minval=1, maxval=100)
        ra = jax.random.uniform(subkeys[2], (1,), minval=0, maxval=2 * jnp.pi)
        dec = jax.random.uniform(
            subkeys[3], (1,), minval=-jnp.pi / 2, maxval=jnp.pi / 2
        )
        psi = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=jnp.pi)
        iota = jax.random.uniform(subkeys[5], (1,), minval=0, maxval=jnp.pi)

        sample = [
            d_hat[0],
            M_c[0],
            ra[0],
            dec[0],
            psi[0],
            iota[0],
        ]
        sample_dict = dict(zip(["d_hat", "M_c", "ra", "dec", "psi", "iota"], sample))

        jit_inverse_transform = jax.jit(
            DistanceToSNRWeightedDistanceTransform(
                trigger_time=1126259462.4,
                ifos=[H1, L1],
            ).inverse
        )
        jitted_output, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_output = DistanceToSNRWeightedDistanceTransform(
            trigger_time=1126259462.4,
            ifos=[H1, L1],
        ).backward(sample_dict)

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)


class TestSphereSpinToCartesianSpinTransform:
    def test_forward_transform(self):
        """
        Test the forward transformation from spherical to Cartesian spin components
        """
        output, jacobian = SphereSpinToCartesianSpinTransform("s1").transform(
            {
                "s1_mag": 0.4,
                "s1_theta": jnp.pi,
                "s1_phi": 0.8,
            }
        )
        assert (
            np.isfinite(output["s1_x"])
            & np.isfinite(output["s1_y"])
            & np.isfinite(output["s1_z"])
        )
        assert_all_finite(jacobian)

    def test_backward_transform(self):
        """
        Test the backward transformation from Cartesian to spherical spin components
        """
        output, jacobian = SphereSpinToCartesianSpinTransform("s1").inverse(
            {
                "s1_x": 0.4,
                "s1_y": 0.5,
                "s1_z": 0.6,
            }
        )
        assert (
            np.isfinite(output["s1_mag"])
            & np.isfinite(output["s1_theta"])
            & np.isfinite(output["s1_phi"])
        )
        assert_all_finite(jacobian)

    def test_forward_backward_consistency(self):
        """
        Test that the forward and inverse transformations are consistent
        """

        key = jax.random.key(42)
        key, *subkeys = jax.random.split(key, 4)
        s1_mag = jax.random.uniform(subkeys[0], (10,), minval=1e-3, maxval=1.0)
        s1_theta = jax.random.uniform(subkeys[1], (10,), minval=0, maxval=jnp.pi)
        s1_phi = jax.random.uniform(subkeys[2], (10,), minval=0, maxval=2 * jnp.pi)

        inputs = {"s1_mag": s1_mag, "s1_theta": s1_theta, "s1_phi": s1_phi}
        transform = SphereSpinToCartesianSpinTransform("s1")
        forward_transform_output = jax.vmap(transform.forward)(inputs)
        output = jax.vmap(transform.backward)(forward_transform_output)

        assert common_keys_allclose(output, inputs)

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """

        subkeys = jax.random.split(jax.random.key(12), 3)
        s1_mag = jax.random.uniform(subkeys[0], (1,), minval=1e-3, maxval=1.0)
        s1_theta = jax.random.uniform(subkeys[1], (1,), minval=0, maxval=jnp.pi)
        s1_phi = jax.random.uniform(subkeys[2], (1,), minval=0, maxval=2 * jnp.pi)

        sample_dict = {
            "s1_mag": s1_mag[0],
            "s1_theta": s1_theta[0],
            "s1_phi": s1_phi[0],
        }

        jit_transform = jax.jit(
            lambda data: SphereSpinToCartesianSpinTransform("s1").transform(data)
        )
        jitted_output, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_output = SphereSpinToCartesianSpinTransform("s1").forward(
            sample_dict
        )

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """

        keys = jax.random.split(jax.random.key(123), 2)
        S1 = jax.random.uniform(keys[0], (3,), minval=-1, maxval=1)
        a1 = jax.random.uniform(keys[1], (1,), minval=1e-3, maxval=1.0)
        S1 *= a1 / jnp.linalg.norm(S1)

        sample_dict = dict(zip(["s1_x", "s1_y", "s1_z"], S1))

        jit_inverse_transform = jax.jit(
            lambda data: SphereSpinToCartesianSpinTransform("s1").inverse(data)
        )
        jitted_output, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_output = SphereSpinToCartesianSpinTransform("s1").backward(
            sample_dict
        )

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)


class TestSpinAnglesToCartesianSpinTransform:
    forward_keys = (
        "theta_jn",
        "phi_jl",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "a_1",
        "a_2",
        "M_c",
        "q",
        "phase_c",
    )

    backward_keys = (
        "iota",
        "s1_x",
        "s1_y",
        "s1_z",
        "s2_x",
        "s2_y",
        "s2_z",
        "M_c",
        "q",
        "phase_c",
    )

    def test_forward_backward_consistency(self):
        """
        Test that the forward and inverse transformations are consistent
        """

        n = 10
        key = jax.random.key(42)
        subkeys = jax.random.split(key, 7)

        S1, S2 = jax.random.uniform(subkeys[0], (2, 3, n), minval=-1, maxval=1)
        a1, a2 = jax.random.uniform(subkeys[1], (2, n), minval=1e-3, maxval=1)
        S1 *= a1 / jnp.linalg.norm(S1, axis=0)
        S2 *= a2 / jnp.linalg.norm(S2, axis=0)

        samples = jnp.array(
            [
                jax.random.uniform(subkeys[2], (n,), minval=0, maxval=jnp.pi),  # iota
                *S1,
                *S2,
                jax.random.uniform(subkeys[3], (n,), minval=1, maxval=100),  # M_c
                jax.random.uniform(subkeys[4], (n,), minval=0.125, maxval=1),  # q
                jax.random.uniform(
                    subkeys[5], (n,), minval=0, maxval=2 * jnp.pi
                ),  # phase_c
            ]
        ).T
        fRefs = jax.random.uniform(subkeys[6], (n,), minval=10, maxval=100)

        for fRef, sample in zip(fRefs, samples):
            jimgw_spins = SpinAnglesToCartesianSpinTransform(freq_ref=fRef).backward(
                dict(zip(self.backward_keys, sample))
            )
            jimgw_spins = SpinAnglesToCartesianSpinTransform(freq_ref=fRef).forward(
                jimgw_spins
            )
            jimgw_spins = jnp.array(list(jimgw_spins.values()))
            assert np.allclose(jimgw_spins, jnp.array([*sample[7:], *sample[:7]]))

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """

        subkeys = jax.random.split(jax.random.key(123), 11)

        theta_jn = jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)
        phi_jl = jax.random.uniform(subkeys[1], (1,), minval=0, maxval=2 * jnp.pi)
        tilt_1 = jax.random.uniform(subkeys[2], (1,), minval=0, maxval=jnp.pi)
        tilt_2 = jax.random.uniform(subkeys[3], (1,), minval=0, maxval=jnp.pi)
        phi_12 = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=2 * jnp.pi)
        a_1 = jax.random.uniform(subkeys[5], (1,), minval=0, maxval=1)
        a_2 = jax.random.uniform(subkeys[6], (1,), minval=0, maxval=1)
        M_c = jax.random.uniform(subkeys[7], (1,), minval=1, maxval=100)
        q = jax.random.uniform(subkeys[8], (1,), minval=0.125, maxval=1)
        phase_c = jax.random.uniform(subkeys[9], (1,), minval=0, maxval=2 * jnp.pi)
        f_ref = jax.random.uniform(subkeys[10], (1,), minval=10, maxval=100)

        sample = [
            theta_jn[0],
            phi_jl[0],
            tilt_1[0],
            tilt_2[0],
            phi_12[0],
            a_1[0],
            a_2[0],
            M_c[0],
            q[0],
            phase_c[0],
        ]
        freq_ref_sample = f_ref[0]
        sample_dict = dict(zip(self.forward_keys, sample))

        jit_transform = jax.jit(
            lambda data: SpinAnglesToCartesianSpinTransform(
                freq_ref=freq_ref_sample
            ).transform(data)
        )
        jitted_spins, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_spins = SpinAnglesToCartesianSpinTransform(
            freq_ref=freq_ref_sample
        ).forward(sample_dict)

        assert common_keys_allclose(jitted_spins, non_jitted_spins)
        assert_all_finite(jitted_jacobian)

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """

        subkeys = jax.random.split(jax.random.key(12), 8)

        iota = jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)
        M_c = jax.random.uniform(subkeys[1], (1,), minval=1, maxval=100)
        q = jax.random.uniform(subkeys[2], (1,), minval=0.125, maxval=1)
        fRef = jax.random.uniform(subkeys[3], (1,), minval=10, maxval=100)
        phiRef = jax.random.uniform(subkeys[4], (1,), minval=0, maxval=2 * jnp.pi)

        S1, S2 = jax.random.uniform(subkeys[5], (2, 3), minval=-1, maxval=1)
        a1, a2 = jax.random.uniform(subkeys[6], (2,), minval=1e-3, maxval=1)
        S1 *= a1 / jnp.linalg.norm(S1)
        S2 *= a2 / jnp.linalg.norm(S2)

        sample = [
            iota[0],
            *S1,
            *S2,
            M_c[0],
            q[0],
            phiRef[0],
        ]
        freq_ref_sample = fRef[0]
        sample_dict = dict(zip(self.backward_keys, sample))

        jit_inverse_transform = jax.jit(
            lambda data: SpinAnglesToCartesianSpinTransform(
                freq_ref=freq_ref_sample
            ).inverse(data)
        )
        jitted_spins, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_spins = SpinAnglesToCartesianSpinTransform(
            freq_ref=freq_ref_sample
        ).backward(sample_dict)

        assert common_keys_allclose(jitted_spins, non_jitted_spins)
        assert_all_finite(jitted_jacobian)


class TestSkyFrameToDetectorFrameSkyPositionTransform:
    def test_forward_backward_consistency(self):
        """
        Test that the forward and inverse transformations are consistent
        """
        ifos = [H1, L1, V1]
        gps_time = [1126259642.413, 1242442967.4]
        key = jax.random.key(42)

        for ifo_pair in combinations(ifos, 2):
            for time in gps_time:
                key, *subkeys = jax.random.split(key, 3)
                inputs = {
                    "zenith": jax.random.uniform(
                        subkeys[0], (10,), minval=0, maxval=jnp.pi
                    ),
                    "azimuth": jax.random.uniform(
                        subkeys[1], (10,), minval=0, maxval=2 * jnp.pi
                    ),
                }
                transform = SkyFrameToDetectorFrameSkyPositionTransform(
                    trigger_time=time,
                    ifos=list(ifo_pair),
                )
                forward_transform_output = jax.vmap(transform.backward)(inputs)
                outputs = jax.vmap(transform.forward)(forward_transform_output)

                assert common_keys_allclose(outputs, inputs)

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """
        subkeys = jax.random.split(jax.random.key(12), 2)
        sample_dict = {
            "ra": jax.random.uniform(subkeys[0], (1,), minval=0, maxval=2 * jnp.pi)[0],
            "dec": jax.random.uniform(subkeys[1], (1,), minval=0, maxval=jnp.pi)[0],
        }
        class_args = dict(trigger_time=1126259642.4, ifos=[H1, L1])

        jit_transform = jax.jit(
            lambda data: SkyFrameToDetectorFrameSkyPositionTransform(
                **class_args
            ).transform(data)
        )
        jitted_output, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_output = SkyFrameToDetectorFrameSkyPositionTransform(
            **class_args
        ).forward(sample_dict)

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """

        subkeys = jax.random.split(jax.random.key(123), 2)
        sample_dict = {
            "zenith": jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)[0],
            "azimuth": jax.random.uniform(
                subkeys[1], (1,), minval=0, maxval=2 * jnp.pi
            )[0],
        }
        class_args = dict(trigger_time=1126259642.4, ifos=[H1, L1])

        jit_inverse_transform = jax.jit(
            lambda data: SkyFrameToDetectorFrameSkyPositionTransform(
                **class_args
            ).inverse(data)
        )
        jitted_output, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_output = SkyFrameToDetectorFrameSkyPositionTransform(
            **class_args
        ).backward(sample_dict)

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)


class TestGeocentricArrivalTimeToDetectorArrivalTimeTransform:
    def test_forward_transform(self):
        """
        Test the forward transformation from geocentric to detector arrival time
        """
        gps_time = 1126259462.4
        transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time=gps_time,
            ifo=H1,
        )

        sample_dict = {
            "t_c": 0.1,
            "ra": 1.5,
            "dec": -0.5,
        }

        output, log_det = transform.transform(sample_dict)

        assert "t_det" in output
        assert np.isfinite(output["t_det"])
        assert np.allclose(log_det, 0.0)

    def test_inverse_transform(self):
        """
        Test the inverse transformation from detector to geocentric arrival time
        """
        gps_time = 1126259462.4
        transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time=gps_time,
            ifo=L1,
        )

        sample_dict = {
            "t_det": 0.05,
            "ra": 2.0,
            "dec": 0.3,
        }

        output, log_det = transform.inverse(sample_dict)

        assert "t_c" in output
        assert np.isfinite(output["t_c"])
        assert np.allclose(log_det, 0.0)

    def test_forward_backward_consistency(self):
        """
        Test that forward and inverse transforms are consistent
        """
        gps_time = 1126259462.4
        transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time=gps_time,
            ifo=H1,
        )

        original = {
            "t_c": 0.15,
            "ra": 1.2,
            "dec": -0.8,
        }

        forward_output = transform.forward(original.copy())
        forward_output["ra"] = original["ra"]
        forward_output["dec"] = original["dec"]
        recovered = transform.backward(forward_output)

        assert np.allclose(recovered["t_c"], original["t_c"], rtol=1e-10)

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """
        gps_time = 1126259462.4
        sample_dict = {
            "t_c": 0.1,
            "ra": 1.5,
            "dec": -0.5,
        }

        class_args = dict(trigger_time=gps_time, ifo=H1)

        jit_transform = jax.jit(
            lambda data: GeocentricArrivalTimeToDetectorArrivalTimeTransform(
                **class_args
            ).transform(data)
        )

        jitted_output, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_output = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            **class_args
        ).forward(sample_dict)

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """
        gps_time = 1126259462.4
        sample_dict = {
            "t_det": 0.05,
            "ra": 2.0,
            "dec": 0.3,
        }

        class_args = dict(trigger_time=gps_time, ifo=L1)

        jit_inverse_transform = jax.jit(
            lambda data: GeocentricArrivalTimeToDetectorArrivalTimeTransform(
                **class_args
            ).inverse(data)
        )

        jitted_output, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_output = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            **class_args
        ).backward(sample_dict)

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)

    def test_multiple_detectors(self):
        """
        Test transform with different detectors
        """
        gps_time = 1126259462.4
        sample_dict = {
            "t_c": 0.0,
            "ra": 1.95,
            "dec": -1.27,
        }

        for ifo in [H1, L1, V1]:
            transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
                trigger_time=gps_time,
                ifo=ifo,
            )
            output = transform.forward(sample_dict.copy())
            assert "t_det" in output
            assert np.isfinite(output["t_det"])


class TestGeocentricArrivalPhaseToDetectorArrivalPhaseTransform:
    def test_forward_transform(self):
        """
        Test the forward transformation from geocentric to detector arrival phase
        """
        gps_time = 1126259462.4
        transform = GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
            trigger_time=gps_time,
            ifo=H1,
        )

        sample_dict = {
            "phase_c": 1.5,
            "ra": 1.95,
            "dec": -1.27,
            "psi": 0.8,
            "iota": 2.5,
        }

        output, log_det = transform.transform(sample_dict)

        assert "phase_det" in output
        assert np.isfinite(output["phase_det"])
        assert 0.0 <= output["phase_det"] < 2.0 * jnp.pi
        assert np.isfinite(log_det)

    def test_inverse_transform(self):
        """
        Test the inverse transformation from detector to geocentric arrival phase
        """
        gps_time = 1126259462.4
        transform = GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
            trigger_time=gps_time,
            ifo=L1,
        )

        sample_dict = {
            "phase_det": 2.5,
            "ra": 1.0,
            "dec": 0.5,
            "psi": 1.2,
            "iota": 1.5,
        }

        output, log_det = transform.inverse(sample_dict)

        assert "phase_c" in output
        assert np.isfinite(output["phase_c"])
        assert 0.0 <= output["phase_c"] < 2.0 * jnp.pi
        assert np.isfinite(log_det)

    def test_forward_backward_consistency(self):
        """
        Test that forward and inverse transforms are approximately consistent
        """
        gps_time = 1126259462.4
        transform = GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
            trigger_time=gps_time,
            ifo=H1,
        )

        original = {
            "phase_c": 3.0,
            "ra": 1.95,
            "dec": -1.27,
            "psi": 0.8,
            "iota": 2.5,
        }

        forward_output, forward_log_det = transform.transform(original.copy())
        forward_output["ra"] = original["ra"]
        forward_output["dec"] = original["dec"]
        forward_output["psi"] = original["psi"]
        forward_output["iota"] = original["iota"]
        recovered, inverse_log_det = transform.inverse(forward_output)

        assert np.allclose(forward_log_det, -inverse_log_det, rtol=1e-10)
        assert np.isfinite(recovered["phase_c"])
        assert 0.0 <= recovered["phase_c"] < 2.0 * jnp.pi

    def test_jitted_forward_transform(self):
        """
        Test that the forward transformation is JIT compilable
        """
        gps_time = 1126259462.4
        sample_dict = {
            "phase_c": 1.5,
            "ra": 1.95,
            "dec": -1.27,
            "psi": 0.8,
            "iota": 2.5,
        }

        class_args = dict(trigger_time=gps_time, ifo=H1)

        jit_transform = jax.jit(
            lambda data: GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
                **class_args
            ).transform(data)
        )

        jitted_output, jitted_jacobian = jit_transform(sample_dict)
        non_jitted_output = GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
            **class_args
        ).forward(sample_dict)

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)

    def test_jitted_backward_transform(self):
        """
        Test that the backward transformation is JIT compilable
        """
        gps_time = 1126259462.4
        sample_dict = {
            "phase_det": 2.5,
            "ra": 1.0,
            "dec": 0.5,
            "psi": 1.2,
            "iota": 1.5,
        }

        class_args = dict(trigger_time=gps_time, ifo=L1)

        jit_inverse_transform = jax.jit(
            lambda data: GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
                **class_args
            ).inverse(data)
        )

        jitted_output, jitted_jacobian = jit_inverse_transform(sample_dict)
        non_jitted_output = GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
            **class_args
        ).backward(sample_dict)

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert_all_finite(jitted_jacobian)

    def test_multiple_detectors(self):
        """
        Test transform with different detectors
        """
        gps_time = 1126259462.4
        sample_dict = {
            "phase_c": 1.5,
            "ra": 1.95,
            "dec": -1.27,
            "psi": 0.8,
            "iota": 2.5,
        }

        for ifo in [H1, L1, V1]:
            transform = GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
                trigger_time=gps_time,
                ifo=ifo,
            )
            output = transform.forward(sample_dict.copy())
            assert "phase_det" in output
            assert np.isfinite(output["phase_det"])
            assert 0.0 <= output["phase_det"] < 2.0 * jnp.pi

    def test_phase_wrapping(self):
        """
        Test that phase values are properly wrapped to [0, 2π)
        """
        gps_time = 1126259462.4
        transform = GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
            trigger_time=gps_time,
            ifo=H1,
        )

        sample_dict = {
            "phase_c": 2.0 * jnp.pi - 0.1,
            "ra": 1.95,
            "dec": -1.27,
            "psi": 0.8,
            "iota": 2.5,
        }

        output = transform.forward(sample_dict)
        assert 0.0 <= output["phase_det"] < 2.0 * jnp.pi

        sample_dict_inv = {
            "phase_det": 2.0 * jnp.pi - 0.05,
            "ra": 1.95,
            "dec": -1.27,
            "psi": 0.8,
            "iota": 2.5,
        }

        output_inv = transform.backward(sample_dict_inv)
        assert 0.0 <= output_inv["phase_c"] < 2.0 * jnp.pi


class TestMassTransforms:
    def test_component_masses_to_chirp_mass_mass_ratio_forward(self):
        """Test forward transform from component masses to chirp mass and mass ratio."""
        sample_dict = {
            "m_1": 30.0,
            "m_2": 20.0,
        }

        output, log_det = ComponentMassesToChirpMassMassRatioTransform.transform(
            sample_dict
        )

        assert "M_c" in output
        assert "q" in output
        assert np.isfinite(output["M_c"])
        assert np.isfinite(output["q"])
        assert np.isfinite(log_det)
        assert np.allclose(output["q"], 20.0 / 30.0)
        assert output["q"] < 1.0

    def test_component_masses_to_chirp_mass_mass_ratio_inverse(self):
        """Test inverse transform from chirp mass and mass ratio to component masses."""
        sample_dict = {
            "M_c": 21.77,
            "q": 0.8,
        }

        output, log_det = ComponentMassesToChirpMassMassRatioTransform.inverse(
            sample_dict
        )

        assert "m_1" in output
        assert "m_2" in output
        assert np.isfinite(output["m_1"])
        assert np.isfinite(output["m_2"])
        assert np.isfinite(log_det)
        assert output["m_1"] >= output["m_2"]

    def test_component_masses_to_chirp_mass_mass_ratio_roundtrip(self):
        """Test forward-inverse roundtrip consistency."""
        original = {
            "m_1": 35.0,
            "m_2": 25.0,
        }

        forward_output = ComponentMassesToChirpMassMassRatioTransform.forward(original)
        recovered = ComponentMassesToChirpMassMassRatioTransform.backward(
            forward_output
        )

        assert np.allclose(recovered["m_1"], original["m_1"])
        assert np.allclose(recovered["m_2"], original["m_2"])

    def test_component_masses_to_chirp_mass_symmetric_mass_ratio_forward(self):
        """Test forward transform from component masses to chirp mass and symmetric mass ratio."""
        sample_dict = {
            "m_1": 30.0,
            "m_2": 20.0,
        }

        output, log_det = (
            ComponentMassesToChirpMassSymmetricMassRatioTransform.transform(sample_dict)
        )

        assert "M_c" in output
        assert "eta" in output
        assert np.isfinite(output["M_c"])
        assert np.isfinite(output["eta"])
        assert np.isfinite(log_det)
        assert 0.0 < output["eta"] <= 0.25

    def test_component_masses_to_chirp_mass_symmetric_mass_ratio_inverse(self):
        """Test inverse transform from chirp mass and symmetric mass ratio to component masses."""
        sample_dict = {
            "M_c": 21.77,
            "eta": 0.24,
        }

        output, log_det = ComponentMassesToChirpMassSymmetricMassRatioTransform.inverse(
            sample_dict
        )

        assert "m_1" in output
        assert "m_2" in output
        assert np.isfinite(output["m_1"])
        assert np.isfinite(output["m_2"])
        assert np.isfinite(log_det)
        assert output["m_1"] >= output["m_2"]

    def test_component_masses_to_chirp_mass_symmetric_mass_ratio_roundtrip(self):
        """Test forward-inverse roundtrip consistency."""
        original = {
            "m_1": 35.0,
            "m_2": 25.0,
        }

        forward_output = ComponentMassesToChirpMassSymmetricMassRatioTransform.forward(
            original
        )
        recovered = ComponentMassesToChirpMassSymmetricMassRatioTransform.backward(
            forward_output
        )

        assert np.allclose(recovered["m_1"], original["m_1"])
        assert np.allclose(recovered["m_2"], original["m_2"])

    def test_mass_ratio_to_symmetric_mass_ratio_forward(self):
        """Test forward transform from mass ratio to symmetric mass ratio."""
        sample_dict = {"q": 0.8}

        output, log_det = MassRatioToSymmetricMassRatioTransform.transform(sample_dict)

        assert "eta" in output
        assert np.isfinite(output["eta"])
        assert np.isfinite(log_det)
        assert 0.0 < output["eta"] <= 0.25

    def test_mass_ratio_to_symmetric_mass_ratio_inverse(self):
        """Test inverse transform from symmetric mass ratio to mass ratio."""
        sample_dict = {"eta": 0.24}

        output, log_det = MassRatioToSymmetricMassRatioTransform.inverse(sample_dict)

        assert "q" in output
        assert np.isfinite(output["q"])
        assert np.isfinite(log_det)
        assert 0.0 < output["q"] <= 1.0

    def test_mass_ratio_to_symmetric_mass_ratio_roundtrip(self):
        """Test forward-inverse roundtrip consistency."""
        original = {"q": 0.75}

        forward_output = MassRatioToSymmetricMassRatioTransform.forward(original)
        recovered = MassRatioToSymmetricMassRatioTransform.backward(forward_output)

        assert np.allclose(recovered["q"], original["q"])

    def test_chirp_mass_mass_ratio_to_component_masses_forward(self):
        """Test ChirpMassMassRatioToComponentMassesTransform (reverse of ComponentMasses->Mc,q)."""
        sample_dict = {
            "M_c": 21.77,
            "q": 0.8,
        }

        output, log_det = ChirpMassMassRatioToComponentMassesTransform.transform(
            sample_dict
        )

        assert "m_1" in output
        assert "m_2" in output
        assert np.isfinite(output["m_1"])
        assert np.isfinite(output["m_2"])
        assert np.isfinite(log_det)
        assert output["m_1"] >= output["m_2"]

    def test_chirp_mass_symmetric_mass_ratio_to_component_masses_forward(self):
        """Test ChirpMassSymmetricMassRatioToComponentMassesTransform."""
        sample_dict = {
            "M_c": 21.77,
            "eta": 0.24,
        }

        output, log_det = (
            ChirpMassSymmetricMassRatioToComponentMassesTransform.transform(sample_dict)
        )

        assert "m_1" in output
        assert "m_2" in output
        assert np.isfinite(output["m_1"])
        assert np.isfinite(output["m_2"])
        assert np.isfinite(log_det)
        assert output["m_1"] >= output["m_2"]

    def test_symmetric_mass_ratio_to_mass_ratio_forward(self):
        """Test SymmetricMassRatioToMassRatioTransform."""
        sample_dict = {"eta": 0.24}

        output, log_det = SymmetricMassRatioToMassRatioTransform.transform(sample_dict)

        assert "q" in output
        assert np.isfinite(output["q"])
        assert np.isfinite(log_det)
        assert 0.0 < output["q"] <= 1.0

    def test_mass_transforms_jit_compilable(self):
        """Test that mass transforms are JIT compilable."""
        sample_dict = {
            "m_1": 30.0,
            "m_2": 20.0,
        }

        jit_transform = jax.jit(ComponentMassesToChirpMassMassRatioTransform.transform)
        jitted_output, jitted_log_det = jit_transform(sample_dict)
        non_jitted_output, non_jitted_log_det = (
            ComponentMassesToChirpMassMassRatioTransform.transform(sample_dict)
        )

        assert common_keys_allclose(jitted_output, non_jitted_output)
        assert np.allclose(jitted_log_det, non_jitted_log_det)

    def test_mass_ratio_edge_case_equal_masses(self):
        """Test mass ratio transform with equal masses (q=1, eta=0.25)."""
        sample_dict = {"q": 1.0}

        output = MassRatioToSymmetricMassRatioTransform.forward(sample_dict)

        assert np.allclose(output["eta"], 0.25)

        recovered = MassRatioToSymmetricMassRatioTransform.backward(output)
        assert np.allclose(recovered["q"], 1.0)
