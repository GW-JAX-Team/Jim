"""Cross-validation tests for spin transforms against bilby.

Requires bilby to be installed.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from tests.utils import assert_all_finite, check_bilby_available, common_keys_allclose

# Check if bilby is available before running tests
try:
    check_bilby_available()
    BILBY_AVAILABLE = True
except ImportError:
    BILBY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BILBY_AVAILABLE,
    reason="bilby required for cross-validation tests",
)


class TestSpinAnglesToCartesianSpinTransformBilby:
    """Cross-validation tests comparing Jim spin transforms to bilby."""

    def test_transform_roundtrip(self):
        """Test spin angle ↔ Cartesian spin transforms against bilby.

        1. Generate random spin angles.
        2. bilby: angles → Cartesian (reference).
        3. Jim forward: angles → Cartesian — assert matches bilby.
        4. Jim inverse: reference Cartesian → recovered angles.
        5. bilby forward on recovered angles → reproduced Cartesian.
        6. Assert reproduced Cartesian ≈ reference Cartesian.
        """
        from bilby.gw.conversion import bilby_to_lalsimulation_spins
        from bilby.gw.utils import solar_mass as MSUN_SI
        from jimgw.core.single_event.transforms import (
            SpinAnglesToCartesianSpinTransform,
        )

        n_samples = 50
        key = jax.random.key(42)
        subkeys = jax.random.split(key, 11)

        # Step 1: generate random angles and masses
        theta_jn = jax.random.uniform(subkeys[0], (n_samples,), minval=0, maxval=jnp.pi)
        phi_jl = jax.random.uniform(
            subkeys[1], (n_samples,), minval=0, maxval=2 * jnp.pi
        )
        tilt_1 = jax.random.uniform(subkeys[2], (n_samples,), minval=0, maxval=jnp.pi)
        tilt_2 = jax.random.uniform(subkeys[3], (n_samples,), minval=0, maxval=jnp.pi)
        phi_12 = jax.random.uniform(
            subkeys[4], (n_samples,), minval=0, maxval=2 * jnp.pi
        )
        a_1 = jax.random.uniform(subkeys[5], (n_samples,), minval=0.01, maxval=0.99)
        a_2 = jax.random.uniform(subkeys[6], (n_samples,), minval=0.01, maxval=0.99)
        M_c = jax.random.uniform(subkeys[7], (n_samples,), minval=5, maxval=50)
        q = jax.random.uniform(subkeys[8], (n_samples,), minval=0.125, maxval=1)
        phase_c = jax.random.uniform(
            subkeys[9], (n_samples,), minval=0, maxval=2 * jnp.pi
        )

        for f_ref in [10.0, 20.0, 50.0]:
            # Derive component masses locally from Mc and q (independent of package under test).
            # m1 = Mc * (1 + q)^(1/5) / q^(3/5),  m2 = q * m1
            m1 = M_c * (1 + q) ** (1 / 5) / q ** (3 / 5)
            m2 = q * m1

            # Step 2: bilby reference forward pass
            bilby_results = []
            for i in range(n_samples):
                iota_b, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
                    theta_jn=float(theta_jn[i]),
                    phi_jl=float(phi_jl[i]),
                    tilt_1=float(tilt_1[i]),
                    tilt_2=float(tilt_2[i]),
                    phi_12=float(phi_12[i]),
                    a_1=float(a_1[i]),
                    a_2=float(a_2[i]),
                    mass_1=float(m1[i]) * MSUN_SI,
                    mass_2=float(m2[i]) * MSUN_SI,
                    reference_frequency=f_ref,
                    phase=float(phase_c[i]),
                )
                bilby_results.append(
                    {
                        "iota": iota_b,
                        "s1_x": s1x,
                        "s1_y": s1y,
                        "s1_z": s1z,
                        "s2_x": s2x,
                        "s2_y": s2y,
                        "s2_z": s2z,
                    }
                )
            bilby_cartesian = {
                k: jnp.array([r[k] for r in bilby_results])
                for k in bilby_results[0].keys()
            }

            transform = SpinAnglesToCartesianSpinTransform(freq_ref=f_ref)

            # Step 3: Jim forward — assert matches bilby
            input_dict = {
                "theta_jn": theta_jn,
                "phi_jl": phi_jl,
                "tilt_1": tilt_1,
                "tilt_2": tilt_2,
                "phi_12": phi_12,
                "a_1": a_1,
                "a_2": a_2,
                "M_c": M_c,
                "q": q,
                "phase_c": phase_c,
            }
            jimgw_cartesian, fwd_jacobian = jax.vmap(transform.transform)(input_dict)
            assert common_keys_allclose(jimgw_cartesian, bilby_cartesian), (
                f"Jim forward and bilby disagree at f_ref={f_ref}"
            )
            assert_all_finite(fwd_jacobian)

            # Step 4: Jim inverse on bilby reference Cartesian → recovered angles
            cartesian_with_masses = {
                **bilby_cartesian,
                "M_c": M_c,
                "q": q,
                "phase_c": phase_c,
            }
            recovered_angles, inv_jacobian = jax.vmap(transform.inverse)(
                cartesian_with_masses
            )
            assert_all_finite(inv_jacobian)

            # Step 5-6: bilby forward on recovered angles → reproduced Cartesian
            reprod_results = []
            for i in range(n_samples):
                iota_b, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
                    theta_jn=float(recovered_angles["theta_jn"][i]),
                    phi_jl=float(recovered_angles["phi_jl"][i]),
                    tilt_1=float(recovered_angles["tilt_1"][i]),
                    tilt_2=float(recovered_angles["tilt_2"][i]),
                    phi_12=float(recovered_angles["phi_12"][i]),
                    a_1=float(recovered_angles["a_1"][i]),
                    a_2=float(recovered_angles["a_2"][i]),
                    mass_1=float(m1[i]) * MSUN_SI,
                    mass_2=float(m2[i]) * MSUN_SI,
                    reference_frequency=f_ref,
                    phase=float(phase_c[i]),
                )
                reprod_results.append(
                    {
                        "iota": iota_b,
                        "s1_x": s1x,
                        "s1_y": s1y,
                        "s1_z": s1z,
                        "s2_x": s2x,
                        "s2_y": s2y,
                        "s2_z": s2z,
                    }
                )
            reprod_cartesian = {
                k: jnp.array([r[k] for r in reprod_results])
                for k in reprod_results[0].keys()
            }
            assert common_keys_allclose(reprod_cartesian, bilby_cartesian), (
                f"Round-trip (bilby → Jim inverse → bilby) fails at f_ref={f_ref}"
            )


class TestSpinTransformCornerCases:
    """Corner-case cross-validation of spin transforms against bilby."""

    # Fixed reference frequency used throughout these tests.
    F_REF = 20.0

    def _bilby_forward(
        self, theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, m1_si, m2_si, phase_c
    ):
        """Call bilby_to_lalsimulation_spins and return a dict."""
        from bilby.gw.conversion import bilby_to_lalsimulation_spins

        iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
            theta_jn=float(theta_jn),
            phi_jl=float(phi_jl),
            tilt_1=float(tilt_1),
            tilt_2=float(tilt_2),
            phi_12=float(phi_12),
            a_1=float(a_1),
            a_2=float(a_2),
            mass_1=float(m1_si),
            mass_2=float(m2_si),
            reference_frequency=self.F_REF,
            phase=float(phase_c),
        )
        return {
            "iota": iota,
            "s1_x": s1x,
            "s1_y": s1y,
            "s1_z": s1z,
            "s2_x": s2x,
            "s2_y": s2y,
            "s2_z": s2z,
        }

    def _jim_forward(self, params):
        """Call Jim's SpinAnglesToCartesianSpinTransform and return the output dict."""
        from jimgw.core.single_event.transforms import (
            SpinAnglesToCartesianSpinTransform,
        )

        transform = SpinAnglesToCartesianSpinTransform(freq_ref=self.F_REF)
        out, _ = transform.transform(params)
        return out

    def _msun_si(self, M_c, q):
        from bilby.gw.utils import solar_mass as MSUN_SI

        m1 = float(M_c * (1 + q) ** (1 / 5) / q ** (3 / 5))
        m2 = float(q * m1)
        return m1 * MSUN_SI, m2 * MSUN_SI

    @pytest.mark.parametrize(
        "tilt_1,tilt_2,label",
        [
            (0.0, 0.0, "aligned"),
            (jnp.pi, jnp.pi, "anti-aligned"),
            (jnp.pi / 2, jnp.pi / 2, "in-plane"),
            (0.0, jnp.pi, "mixed"),
        ],
    )
    def test_tilt_boundary(self, tilt_1, tilt_2, label):
        """Spin tilt boundary values (0, π/2, π) match bilby."""
        params = {
            "theta_jn": jnp.float64(0.8),
            "phi_jl": jnp.float64(1.2),
            "tilt_1": jnp.float64(tilt_1),
            "tilt_2": jnp.float64(tilt_2),
            "phi_12": jnp.float64(0.5),
            "a_1": jnp.float64(0.5),
            "a_2": jnp.float64(0.3),
            "M_c": jnp.float64(20.0),
            "q": jnp.float64(0.8),
            "phase_c": jnp.float64(0.3),
        }
        m1_si, m2_si = self._msun_si(params["M_c"], params["q"])
        ref = self._bilby_forward(
            params["theta_jn"],
            params["phi_jl"],
            tilt_1,
            tilt_2,
            params["phi_12"],
            params["a_1"],
            params["a_2"],
            m1_si,
            m2_si,
            params["phase_c"],
        )
        jim = self._jim_forward(params)
        assert common_keys_allclose(jim, ref), (
            f"tilt boundary '{label}': Jim and bilby disagree.\n"
            f"  Jim: {jim}\n  bilby: {ref}"
        )

    @pytest.mark.parametrize(
        "theta_jn,label",
        [
            (0.0, "face-on"),
            (jnp.pi, "face-away"),
            (jnp.pi / 2, "edge-on"),
        ],
    )
    def test_theta_jn_boundary(self, theta_jn, label):
        """Inclination angle boundary values (0, π/2, π) match bilby."""
        params = {
            "theta_jn": jnp.float64(theta_jn),
            "phi_jl": jnp.float64(1.2),
            "tilt_1": jnp.float64(0.4),
            "tilt_2": jnp.float64(0.6),
            "phi_12": jnp.float64(0.5),
            "a_1": jnp.float64(0.5),
            "a_2": jnp.float64(0.3),
            "M_c": jnp.float64(20.0),
            "q": jnp.float64(0.8),
            "phase_c": jnp.float64(0.3),
        }
        m1_si, m2_si = self._msun_si(params["M_c"], params["q"])
        ref = self._bilby_forward(
            theta_jn,
            params["phi_jl"],
            params["tilt_1"],
            params["tilt_2"],
            params["phi_12"],
            params["a_1"],
            params["a_2"],
            m1_si,
            m2_si,
            params["phase_c"],
        )
        jim = self._jim_forward(params)
        assert common_keys_allclose(jim, ref), (
            f"theta_jn '{label}': Jim and bilby disagree.\n  Jim: {jim}\n  bilby: {ref}"
        )

    @pytest.mark.parametrize(
        "phi_jl,label",
        [
            (0.0, "phi_jl=0"),
            (jnp.pi, "phi_jl=pi"),
            (2 * jnp.pi - 1e-10, "phi_jl≈2pi"),
        ],
    )
    def test_phi_jl_boundary(self, phi_jl, label):
        """phi_jl boundary values match bilby."""
        params = {
            "theta_jn": jnp.float64(0.8),
            "phi_jl": jnp.float64(phi_jl),
            "tilt_1": jnp.float64(0.4),
            "tilt_2": jnp.float64(0.6),
            "phi_12": jnp.float64(0.5),
            "a_1": jnp.float64(0.5),
            "a_2": jnp.float64(0.3),
            "M_c": jnp.float64(20.0),
            "q": jnp.float64(0.8),
            "phase_c": jnp.float64(0.3),
        }
        m1_si, m2_si = self._msun_si(params["M_c"], params["q"])
        ref = self._bilby_forward(
            params["theta_jn"],
            phi_jl,
            params["tilt_1"],
            params["tilt_2"],
            params["phi_12"],
            params["a_1"],
            params["a_2"],
            m1_si,
            m2_si,
            params["phase_c"],
        )
        jim = self._jim_forward(params)
        assert common_keys_allclose(jim, ref), (
            f"{label}: Jim and bilby disagree.\n  Jim: {jim}\n  bilby: {ref}"
        )

    def test_equal_mass(self):
        """q=1 (equal mass) matches bilby."""
        params = {
            "theta_jn": jnp.float64(0.8),
            "phi_jl": jnp.float64(1.2),
            "tilt_1": jnp.float64(0.4),
            "tilt_2": jnp.float64(0.6),
            "phi_12": jnp.float64(0.5),
            "a_1": jnp.float64(0.5),
            "a_2": jnp.float64(0.3),
            "M_c": jnp.float64(20.0),
            "q": jnp.float64(1.0),
            "phase_c": jnp.float64(0.3),
        }
        m1_si, m2_si = self._msun_si(params["M_c"], params["q"])
        ref = self._bilby_forward(
            params["theta_jn"],
            params["phi_jl"],
            params["tilt_1"],
            params["tilt_2"],
            params["phi_12"],
            params["a_1"],
            params["a_2"],
            m1_si,
            m2_si,
            params["phase_c"],
        )
        jim = self._jim_forward(params)
        assert common_keys_allclose(jim, ref), (
            f"equal mass (q=1): Jim and bilby disagree.\n  Jim: {jim}\n  bilby: {ref}"
        )

    def test_near_extremal_spin(self):
        """Near-extremal spin magnitudes (a=0.999) match bilby."""
        params = {
            "theta_jn": jnp.float64(0.8),
            "phi_jl": jnp.float64(1.2),
            "tilt_1": jnp.float64(0.4),
            "tilt_2": jnp.float64(0.6),
            "phi_12": jnp.float64(0.5),
            "a_1": jnp.float64(0.999),
            "a_2": jnp.float64(0.999),
            "M_c": jnp.float64(20.0),
            "q": jnp.float64(0.8),
            "phase_c": jnp.float64(0.3),
        }
        m1_si, m2_si = self._msun_si(params["M_c"], params["q"])
        ref = self._bilby_forward(
            params["theta_jn"],
            params["phi_jl"],
            params["tilt_1"],
            params["tilt_2"],
            params["phi_12"],
            params["a_1"],
            params["a_2"],
            m1_si,
            m2_si,
            params["phase_c"],
        )
        jim = self._jim_forward(params)
        assert common_keys_allclose(jim, ref), (
            f"near-extremal spin: Jim and bilby disagree.\n  Jim: {jim}\n  bilby: {ref}"
        )
