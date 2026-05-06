"""Cross-validation tests for sky position transforms against bilby.

This module contains comprehensive tests comparing Jim's sky position transform
implementations with bilby's implementations at both high-level (full transform)
and low-level (component functions) granularity.

Requires:
    - bilby
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from itertools import combinations

from tests.utils import assert_all_finite, check_bilby_available

# Check if bilby is available before running tests
try:
    check_bilby_available()
    import bilby_cython  # noqa: F401

    BILBY_AVAILABLE = True
except ImportError:
    BILBY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BILBY_AVAILABLE,
    reason="bilby required for cross-validation tests",
)

N_SAMPLES = 1000


class TestSkyFrameToDetectorFrameHighLevel:
    """High-level cross-validation tests comparing Jim sky transforms to bilby."""

    def test_transform_roundtrip(self):
        """Test both directions of the sky position transform against bilby.

        Both the forward (ra/dec → zenith/azimuth) and inverse (zenith/azimuth →
        ra/dec) transforms are validated using bilby's zenith_azimuth_to_ra_dec as
        the single reference:

        1. Generate random zenith/azimuth.
        2. bilby: zenith/azimuth → ra/dec (reference).
        3. Jim inverse: zenith/azimuth → ra/dec — assert matches bilby.
        4. Jim forward: ra/dec → zenith/azimuth — assert recovers original values.
        """
        from bilby.gw.detector import InterferometerList
        from bilby.gw.utils import zenith_azimuth_to_ra_dec
        from jimgw.core.single_event.detector import get_detector_preset
        from jimgw.core.single_event.transforms import (
            SkyFrameToDetectorFrameSkyPositionTransform,
        )

        detector_preset = get_detector_preset()
        H1 = detector_preset["H1"]
        L1 = detector_preset["L1"]
        V1 = detector_preset["V1"]
        ifos = [H1, L1, V1]

        n_samples = 50
        gps_times = [1126259462.4, 1242442967.4]
        key = jax.random.key(42)

        for ifo_pair in combinations(ifos, 2):
            ifo_names = [ifo.name for ifo in ifo_pair]
            for gps_time in gps_times:
                key, *subkeys = jax.random.split(key, 3)

                # Step 1: generate random zenith/azimuth
                zenith = np.array(
                    jax.random.uniform(
                        subkeys[0], (n_samples,), minval=0.1, maxval=jnp.pi - 0.1
                    )
                )
                azimuth = np.array(
                    jax.random.uniform(
                        subkeys[1], (n_samples,), minval=0, maxval=2 * jnp.pi
                    )
                )

                # Step 2: bilby reference zenith/azimuth → ra/dec
                bilby_ifos = InterferometerList(ifo_names)
                bilby_ra = []
                bilby_dec = []
                for i in range(n_samples):
                    ra, dec = zenith_azimuth_to_ra_dec(
                        zenith=float(zenith[i]),
                        azimuth=float(azimuth[i]),
                        geocent_time=float(gps_time),
                        ifos=bilby_ifos,
                    )
                    bilby_ra.append(ra)
                    bilby_dec.append(dec)

                bilby_ra = jnp.array(bilby_ra)
                bilby_dec = jnp.array(bilby_dec)

                transform = SkyFrameToDetectorFrameSkyPositionTransform(
                    trigger_time=gps_time,
                    ifos=list(ifo_pair),
                )

                # Step 3: Jim inverse (zenith/azimuth → ra/dec) matches bilby
                jim_ra_dec, inv_jacobian = jax.vmap(transform.inverse)(
                    {"zenith": jnp.array(zenith), "azimuth": jnp.array(azimuth)}
                )
                assert jnp.allclose(jim_ra_dec["ra"], bilby_ra), (
                    f"Jim and bilby RA disagree for {ifo_names} at gps={gps_time}"
                )
                assert jnp.allclose(jim_ra_dec["dec"], bilby_dec), (
                    f"Jim and bilby dec disagree for {ifo_names} at gps={gps_time}"
                )
                assert_all_finite(inv_jacobian)

                # Step 4: Jim forward (ra/dec → zenith/azimuth) recovers original
                jim_zenith_azimuth, fwd_jacobian = jax.vmap(transform.transform)(
                    {"ra": bilby_ra, "dec": bilby_dec}
                )
                assert jnp.allclose(jim_zenith_azimuth["zenith"], jnp.array(zenith)), (
                    f"Jim forward zenith round-trip failed for {ifo_names} at gps={gps_time}"
                )
                assert jnp.allclose(
                    jim_zenith_azimuth["azimuth"], jnp.array(azimuth)
                ), (
                    f"Jim forward azimuth round-trip failed for {ifo_names} at gps={gps_time}"
                )
                assert_all_finite(fwd_jacobian)


class TestThetaPhiToRaDec:
    """Test _theta_phi_to_ra_dec utility function against bilby."""

    def test_theta_phi_to_ra_dec(self):
        """Compare Jim's _theta_phi_to_ra_dec with bilby's implementation."""
        from jimgw.core.single_event.transform_utils import (
            _theta_phi_to_ra_dec as jim_theta_phi_to_ra_dec,
        )
        from bilby.core.utils import theta_phi_to_ra_dec as bilby_theta_phi_to_ra_dec

        key = jax.random.key(42)

        tol_diff_ra = 0
        tol_diff_dec = 0

        for _ in range(N_SAMPLES):
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, 3)
            theta = jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)
            phi = jax.random.uniform(subkeys[1], (1,), minval=0, maxval=2 * jnp.pi)
            gmst = jax.random.uniform(subkeys[2], (1,), minval=0, maxval=2 * jnp.pi)

            jim_ra, jim_dec = jim_theta_phi_to_ra_dec(theta, phi, gmst)
            bilby_ra, bilby_dec = bilby_theta_phi_to_ra_dec(theta, phi, gmst)
            bilby_ra = bilby_ra % (2 * jnp.pi)
            diff_ra = jnp.abs(jim_ra - bilby_ra)
            diff_dec = jnp.abs(jim_dec - bilby_dec)
            tol_diff_ra += diff_ra
            tol_diff_dec += diff_dec

            assert jnp.allclose(jim_ra, bilby_ra, atol=1e-5), (
                f"jim_ra: {jim_ra}, bilby_ra: {bilby_ra}"
            )
            assert jnp.allclose(jim_dec, bilby_dec, atol=1e-5), (
                f"jim_dec: {jim_dec}, bilby_dec: {bilby_dec}"
            )

        mean_ra_diff = tol_diff_ra / N_SAMPLES
        mean_dec_diff = tol_diff_dec / N_SAMPLES
        assert mean_ra_diff < 1e-5, f"Mean RA diff too large: {mean_ra_diff}"
        assert mean_dec_diff < 1e-5, f"Mean dec diff too large: {mean_dec_diff}"


class TestAngleRotation:
    """Test angle rotation functions against bilby_cython."""

    def test_angle_rotation(self):
        """Compare Jim's angle_rotation with bilby_cython's zenith_azimuth_to_theta_phi."""
        from jimgw.core.single_event.transform_utils import (
            angle_rotation as jim_angle_rotation,
        )
        from jimgw.core.single_event.transform_utils import euler_rotation
        from bilby_cython.geometry import (
            zenith_azimuth_to_theta_phi as bilby_angle_rotation,
        )
        from bilby_cython.geometry import rotation_matrix_from_delta

        tol_diff_theta = 0
        tol_diff_phi = 0

        for _ in range(N_SAMPLES):
            zenith = np.random.uniform(0, np.pi)
            azimuth = np.random.uniform(0, 2 * np.pi)
            delta_x = np.random.uniform(0, 1, size=3)

            # Ensure rotation matrices are the same
            jim_rot = euler_rotation(delta_x)
            bilby_rot = rotation_matrix_from_delta(delta_x)

            assert jnp.allclose(jim_rot, bilby_rot), (
                f"jim_rot: {jim_rot}, bilby_rot: {bilby_rot}"
            )

            jim_theta, jim_phi = jim_angle_rotation(zenith, azimuth, jim_rot)
            bilby_out = bilby_angle_rotation(zenith, azimuth, delta_x)

            diff_theta = jnp.abs(jim_theta - bilby_out[0])
            diff_phi = jnp.abs(jim_phi - bilby_out[1])

            tol_diff_theta += diff_theta
            tol_diff_phi += diff_phi

            assert jnp.allclose(jim_theta, bilby_out[0]), (
                f"jim_theta: {jim_theta}, bilby_theta: {bilby_out[0]}"
            )
            assert jnp.allclose(jim_phi, bilby_out[1]), (
                f"jim_phi: {jim_phi}, bilby_phi: {bilby_out[1]}"
            )

        mean_diff_theta = tol_diff_theta / N_SAMPLES
        mean_diff_phi = tol_diff_phi / N_SAMPLES
        assert mean_diff_theta < 1e-10, f"Mean theta diff too large: {mean_diff_theta}"
        assert mean_diff_phi < 1e-10, f"Mean phi diff too large: {mean_diff_phi}"


class TestDeltaX:
    """Test delta_x calculation against bilby."""

    def test_delta_x(self):
        """Compare Jim's detector vertex differences with bilby's."""
        from bilby.gw.detector import InterferometerList
        from jimgw.core.single_event.detector import get_detector_preset

        detector_preset = get_detector_preset()
        HLV = ["H1", "L1", "V1"]

        for ifos in combinations(HLV, 2):
            jim_ifos = [detector_preset[ifo] for ifo in ifos]
            bilby_ifos = InterferometerList(ifos)

            delta_x_j = jim_ifos[0].vertex - jim_ifos[1].vertex
            delta_x_b = bilby_ifos[0].vertex - bilby_ifos[1].vertex

            diff = jnp.abs(delta_x_j - delta_x_b)
            assert jnp.allclose(delta_x_j, delta_x_b, atol=1e-10), (
                f"Delta_x mismatch for {ifos}: max diff = {jnp.max(diff)}"
            )


class TestGMST:
    """Test Greenwich Mean Sidereal Time calculation against bilby."""

    def test_gmst(self):
        """Compare Jim's GMST with bilby_cython's implementation."""
        from jimgw.core.single_event.time_utils import (
            greenwich_mean_sidereal_time as jim_gmst,
        )
        from bilby_cython.time import greenwich_mean_sidereal_time

        tol_diff = 0
        gps_times = jax.random.uniform(
            jax.random.key(42), N_SAMPLES, minval=1, maxval=2e9 + 1234.5678
        )

        for time in gps_times:
            gmst_jim = jim_gmst(time) % (2 * np.pi)
            gmst_b = greenwich_mean_sidereal_time(time) % (2 * np.pi)
            tol_diff += jnp.abs(gmst_jim - gmst_b)

        mean_diff = tol_diff / N_SAMPLES
        # With the new GMST implementation, the difference should be extremely small
        assert mean_diff < 1e-10, f"Mean GMST diff too large: {mean_diff}"


class TestFullTransform:
    """Test full SkyFrameToDetectorFrameSkyPositionTransform against bilby."""

    def test_full_sky_transform(self):
        """Compare Jim's full sky transform with bilby's zenith_azimuth_to_ra_dec."""
        from jimgw.core.single_event.transforms import (
            SkyFrameToDetectorFrameSkyPositionTransform,
        )
        from jimgw.core.single_event.detector import get_detector_preset
        from bilby.gw.utils import (
            zenith_azimuth_to_ra_dec as bilby_zenith_azimuth_to_ra_dec,
        )
        from bilby.gw.detector import InterferometerList

        key = jax.random.key(42)
        gps_time = 1126259642.413

        detector_preset = get_detector_preset()
        jim_ifos = [detector_preset["H1"], detector_preset["L1"]]
        ifo_names = ["H1", "L1"]
        bilby_ifos = InterferometerList(ifo_names)

        tol_diff_dec = 0
        tol_diff_ra = 0

        for _ in range(N_SAMPLES):
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, 2)
            zenith = jax.random.uniform(subkeys[0], (1,), minval=0, maxval=jnp.pi)
            azimuth = jax.random.uniform(subkeys[1], (1,), minval=0, maxval=2 * jnp.pi)

            jim_transform = SkyFrameToDetectorFrameSkyPositionTransform(
                trigger_time=gps_time, ifos=jim_ifos
            )
            jim_outputs = jim_transform.backward(dict(zenith=zenith, azimuth=azimuth))
            bilby_ra, bilby_dec = bilby_zenith_azimuth_to_ra_dec(
                zenith[0], azimuth[0], gps_time, bilby_ifos
            )
            jim_ra = jim_outputs["ra"]
            jim_dec = jim_outputs["dec"]

            diff_ra = jnp.abs(jim_ra - bilby_ra)
            diff_dec = jnp.abs(jim_dec - bilby_dec)
            tol_diff_ra += diff_ra
            tol_diff_dec += diff_dec

        mean_ra_diff = tol_diff_ra / N_SAMPLES
        mean_dec_diff = tol_diff_dec / N_SAMPLES

        # With the new GMST implementation, differences should be very small
        assert mean_ra_diff < 1e-10, f"Mean RA diff too large: {mean_ra_diff}"
        assert mean_dec_diff < 1e-10, f"Mean dec diff too large: {mean_dec_diff}"


class TestAngleRotationEquivalence:
    """Test that old and new implementations of angle rotation are equivalent."""

    def test_old_vs_new_implementation(self):
        """Ensure new and old angle rotation implementations are equivalent."""
        from jimgw.core.single_event.transform_utils import euler_rotation

        key = jax.random.key(123)
        key, subkey = jax.random.split(key)
        zenith, azimuth = jax.random.uniform(
            key, (2, N_SAMPLES), minval=0, maxval=jnp.pi
        )
        azimuth *= 2.0
        delta_x = jax.random.uniform(subkey, (N_SAMPLES, 3), minval=0, maxval=1)

        def old_angle_rotation(zenith, azimuth, rotation):
            sin_azimuth = jnp.sin(azimuth)
            cos_azimuth = jnp.cos(azimuth)
            sin_zenith = jnp.sin(zenith)
            cos_zenith = jnp.cos(zenith)

            theta = jnp.acos(
                rotation[2][0] * sin_zenith * cos_azimuth
                + rotation[2][1] * sin_zenith * sin_azimuth
                + rotation[2][2] * cos_zenith
            )
            phi = jnp.fmod(
                jnp.atan2(
                    rotation[1][0] * sin_zenith * cos_azimuth
                    + rotation[1][1] * sin_zenith * sin_azimuth
                    + rotation[1][2] * cos_zenith,
                    rotation[0][0] * sin_zenith * cos_azimuth
                    + rotation[0][1] * sin_zenith * sin_azimuth
                    + rotation[0][2] * cos_zenith,
                )
                + 2 * jnp.pi,
                2 * jnp.pi,
            )
            return theta, phi

        def new_angle_rotation(zenith, azimuth, rotation):
            sky_loc_vec = jnp.array(
                [
                    jnp.sin(zenith) * jnp.cos(azimuth),
                    jnp.sin(zenith) * jnp.sin(azimuth),
                    jnp.cos(zenith),
                ]
            )
            rotated_vec = jnp.einsum("ij,j...->i...", rotation, sky_loc_vec)

            theta = jnp.acos(rotated_vec[2])
            phi = jnp.fmod(
                jnp.atan2(rotated_vec[1], rotated_vec[0]) + 2 * jnp.pi,
                2 * jnp.pi,
            )
            return theta, phi

        # Use stringent tolerance for this test to ensure equivalence
        atol = 1e-13
        rtol = 5e-15

        max_diff = []
        for delta_x_i in delta_x:
            rotation_mat = euler_rotation(delta_x_i)

            old_theta_phi = jnp.array(old_angle_rotation(zenith, azimuth, rotation_mat))
            new_theta_phi = jnp.array(new_angle_rotation(zenith, azimuth, rotation_mat))

            abs_diff = jnp.abs(old_theta_phi - new_theta_phi)
            max_diff.append(jnp.max(abs_diff))

            assert jnp.allclose(old_theta_phi, new_theta_phi, rtol=rtol, atol=atol), (
                f"Max abs diff: {jnp.max(abs_diff):.3e}; "
                + f"old_theta_phi: {old_theta_phi}, \nnew_theta_phi: {new_theta_phi}"
            )

        # Implementations should match at machine precision
        assert jnp.max(jnp.array(max_diff)) < 1e-12
