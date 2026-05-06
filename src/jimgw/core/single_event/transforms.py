from typing import Sequence
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Float, Array, jaxtyped

from jimgw.core.single_event.detector import GroundBased2G
from jimgw.core.transforms import (
    ConditionalBijectiveTransform,
    BijectiveTransform,
    reverse_bijective_transform,
)
from jimgw.core.utils import carte_to_spherical_angles
from jimgw.core.single_event.transform_utils import (
    m1_m2_to_Mc_q,
    Mc_q_to_m1_m2,
    m1_m2_to_Mc_eta,
    Mc_eta_to_m1_m2,
    q_to_eta,
    eta_to_q,
    ra_dec_to_zenith_azimuth,
    zenith_azimuth_to_ra_dec,
    euler_rotation,
    spin_angles_to_cartesian_spin,
    cartesian_spin_to_spin_angles,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)

# Move these to constants.
HR_TO_RAD = 2 * jnp.pi / 24
HR_TO_SEC = 3600
SEC_TO_RAD = HR_TO_RAD / HR_TO_SEC


@jaxtyped(typechecker=typechecker)
class SpinAnglesToCartesianSpinTransform(ConditionalBijectiveTransform):
    """Transform spin angles (J-frame convention) to Cartesian spin components.

    Converts ``(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2)`` to
    ``(iota, s1_x, s1_y, s1_z, s2_x, s2_y, s2_z)`` using the LALSimulation
    convention. The conditioning parameters are ``(M_c, q)`` and optionally
    ``phase_c``.

    Attributes:
        freq_ref (Float): Reference frequency used in the spin conversion.
    """

    freq_ref: Float

    def __repr__(self):
        return f"SpinAnglesToCartesianSpinTransform(freq_ref={self.freq_ref})"

    def __init__(
        self,
        freq_ref: Float,
        fixed_phase: bool = False,
    ) -> None:
        """
        Args:
            freq_ref (Float): Reference frequency in Hz for the spin-angle conversion.
            fixed_phase (bool): If True, the coalescence phase ``phase_c`` is not
                included in the conditioning parameters (treated as fixed at 0).
                Defaults to False.
        """
        name_mapping = (
            ["theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2"],
            ["iota", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z"],
        )

        conditional_names = ["M_c", "q"] if fixed_phase else ["M_c", "q", "phase_c"]
        super().__init__(name_mapping, conditional_names)

        self.freq_ref = freq_ref

        if fixed_phase:

            def named_transform(x):
                iota, s1x, s1y, s1z, s2x, s2y, s2z = spin_angles_to_cartesian_spin(
                    x["theta_jn"],
                    x["phi_jl"],
                    x["tilt_1"],
                    x["tilt_2"],
                    x["phi_12"],
                    x["a_1"],
                    x["a_2"],
                    x["M_c"],
                    x["q"],
                    self.freq_ref,
                    0.0,
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

            def named_inverse_transform(x):
                (
                    theta_jn,
                    phi_jl,
                    tilt_1,
                    tilt_2,
                    phi_12,
                    a_1,
                    a_2,
                ) = cartesian_spin_to_spin_angles(
                    x["iota"],
                    x["s1_x"],
                    x["s1_y"],
                    x["s1_z"],
                    x["s2_x"],
                    x["s2_y"],
                    x["s2_z"],
                    x["M_c"],
                    x["q"],
                    self.freq_ref,
                    0.0,
                )

                return {
                    "theta_jn": theta_jn,
                    "phi_jl": phi_jl,
                    "tilt_1": tilt_1,
                    "tilt_2": tilt_2,
                    "phi_12": phi_12,
                    "a_1": a_1,
                    "a_2": a_2,
                }
        else:

            def named_transform(x):
                iota, s1x, s1y, s1z, s2x, s2y, s2z = spin_angles_to_cartesian_spin(
                    x["theta_jn"],
                    x["phi_jl"],
                    x["tilt_1"],
                    x["tilt_2"],
                    x["phi_12"],
                    x["a_1"],
                    x["a_2"],
                    x["M_c"],
                    x["q"],
                    self.freq_ref,
                    x["phase_c"],
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

            def named_inverse_transform(x):
                (
                    theta_jn,
                    phi_jl,
                    tilt_1,
                    tilt_2,
                    phi_12,
                    a_1,
                    a_2,
                ) = cartesian_spin_to_spin_angles(
                    x["iota"],
                    x["s1_x"],
                    x["s1_y"],
                    x["s1_z"],
                    x["s2_x"],
                    x["s2_y"],
                    x["s2_z"],
                    x["M_c"],
                    x["q"],
                    self.freq_ref,
                    x["phase_c"],
                )

                return {
                    "theta_jn": theta_jn,
                    "phi_jl": phi_jl,
                    "tilt_1": tilt_1,
                    "tilt_2": tilt_2,
                    "phi_12": phi_12,
                    "a_1": a_1,
                    "a_2": a_2,
                }

        self.transform_func = named_transform
        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class SphereSpinToCartesianSpinTransform(BijectiveTransform):
    """Transform spin magnitude and angles to Cartesian spin components.

    Converts ``({label}_mag, {label}_theta, {label}_phi)`` to
    ``({label}_x, {label}_y, {label}_z)`` using the standard spherical-to-Cartesian
    conversion.
    """

    def __repr__(self):
        return f"SphereSpinToCartesianSpinTransform(name_mapping={self.name_mapping})"

    def __init__(
        self,
        label: str,
    ) -> None:
        """
        Args:
            label (str): Parameter label prefix (e.g. ``"s1"`` produces
                ``s1_mag``, ``s1_theta``, ``s1_phi`` → ``s1_x``, ``s1_y``, ``s1_z``).
        """
        name_mapping = (
            [label + "_mag", label + "_theta", label + "_phi"],
            [label + "_x", label + "_y", label + "_z"],
        )
        super().__init__(name_mapping)

        def named_transform(x):
            mag, theta, phi = x[label + "_mag"], x[label + "_theta"], x[label + "_phi"]
            x = mag * jnp.sin(theta) * jnp.cos(phi)
            y = mag * jnp.sin(theta) * jnp.sin(phi)
            z = mag * jnp.cos(theta)
            return {
                label + "_x": x,
                label + "_y": y,
                label + "_z": z,
            }

        def named_inverse_transform(x):
            x, y, z = x[label + "_x"], x[label + "_y"], x[label + "_z"]
            mag = jnp.sqrt(x**2 + y**2 + z**2)
            theta, phi = carte_to_spherical_angles(x, y, z)
            phi = jnp.mod(phi, 2.0 * jnp.pi)

            return {
                label + "_mag": mag,
                label + "_theta": theta,
                label + "_phi": phi,
            }

        self.transform_func = named_transform
        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class SkyFrameToDetectorFrameSkyPositionTransform(BijectiveTransform):
    """Transform sky position from equatorial (RA/Dec) to detector-frame (zenith/azimuth).

    Converts ``(ra, dec)`` to ``(zenith, azimuth)`` relative to the baseline between
    two detectors at the given trigger time. The rotation matrix is computed from the
    baseline vector between the first two detectors in ``ifos``.

    Attributes:
        gmst (Float): Greenwich Mean Sidereal Time at the trigger time in radians.
        rotation (Float[Array, "3 3"]): Rotation matrix from equatorial to detector frame.
        rotation_inv (Float[Array, "3 3"]): Inverse rotation matrix.
    """

    gmst: Float
    rotation: Float[Array, "3 3"]
    rotation_inv: Float[Array, "3 3"]

    def __repr__(self):
        return f"SkyFrameToDetectorFrameSkyPositionTransform(gmst={self.gmst})"

    def __init__(
        self,
        trigger_time: Float,
        ifos: Sequence[GroundBased2G],
    ) -> None:
        """
        Args:
            trigger_time (Float): GPS trigger time in seconds.
            ifos (Sequence[GroundBased2G]): At least two detectors; the rotation is
                computed from the baseline vector between ``ifos[0]`` and ``ifos[1]``.
        """
        name_mapping = (["ra", "dec"], ["zenith", "azimuth"])
        super().__init__(name_mapping)

        self.gmst = compute_gmst(trigger_time)
        delta_x = ifos[0].vertex - ifos[1].vertex
        self.rotation = euler_rotation(delta_x)
        self.rotation_inv = jnp.linalg.inv(self.rotation)

        def named_transform(x):
            zenith, azimuth = ra_dec_to_zenith_azimuth(
                x["ra"], x["dec"], self.gmst, self.rotation_inv
            )
            return {"zenith": zenith, "azimuth": azimuth}

        self.transform_func = named_transform

        def named_inverse_transform(x):
            ra, dec = zenith_azimuth_to_ra_dec(
                x["zenith"], x["azimuth"], self.gmst, self.rotation
            )
            return {"ra": ra, "dec": dec}

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class GeocentricArrivalTimeToDetectorArrivalTimeTransform(
    ConditionalBijectiveTransform
):
    """Transform geocentric coalescence time offset to detector arrival time offset.

    In the geocentric convention the signal arrives at Earth's centre at
    ``trigger_time + t_c``.  In the detector convention it arrives at the
    detector at ``trigger_time + time_delay_from_geocenter + t_det``.

    Maps ``t_c`` → ``t_det`` (forward) and ``t_det`` → ``t_c`` (inverse).
    Conditioning parameters are ``(ra, dec)``.

    Attributes:
        gmst (Float): Greenwich Mean Sidereal Time at the trigger time in radians.
        ifo (GroundBased2G): The target detector.
    """

    gmst: Float
    ifo: GroundBased2G

    def __repr__(self):
        return f"GeocentricArrivalTimeToDetectorArrivalTimeTransform(gmst={self.gmst}, ifo={self.ifo.name})"

    def __init__(
        self,
        trigger_time: Float,
        ifo: GroundBased2G,
    ) -> None:
        """
        Args:
            trigger_time (Float): GPS trigger time in seconds.
            ifo (GroundBased2G): The target detector for which to compute the
                time delay from the geocentre.
        """
        name_mapping = (["t_c"], ["t_det"])
        conditional_names = ["ra", "dec"]
        super().__init__(name_mapping, conditional_names)

        self.gmst = compute_gmst(trigger_time)
        self.ifo = ifo

        assert "t_c" in name_mapping[0] and "t_det" in name_mapping[1]
        assert "ra" in conditional_names and "dec" in conditional_names

        def time_delay(ra, dec, gmst):
            return self.ifo.delay_from_geocenter(ra, dec, gmst)

        def named_transform(x):
            time_shift = time_delay(x["ra"], x["dec"], self.gmst)

            t_det = x["t_c"] + time_shift

            return {
                "t_det": t_det,
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):
            time_shift = self.ifo.delay_from_geocenter(x["ra"], x["dec"], self.gmst)

            t_c = x["t_det"] - time_shift

            return {
                "t_c": t_c,
            }

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(
    ConditionalBijectiveTransform
):
    """Transform geocentric coalescence phase to detector arrival phase.

    In the geocentric convention the orbital phase at coalescence is
    ``phase_c`` (so the GW phase is ``phase_c / 2``).  In the detector
    convention the arrival phase is

    .. math::

        \\phi_{\\mathrm{det}} = \\frac{\\phi_c}{2} + \\arg R_{\\mathrm{det}}

    where :math:`R_{\\mathrm{det}}` is the complex detector response.

    Conditioning parameters are ``(ra, dec, psi, iota)``.

    Warning:
        This transform is derived under the assumption that the waveform consists
        only of the dominant quadrupolar mode (:math:`\\ell = 2, |m| = 2`), following
        the parameterisation in `arXiv:2207.03508 <https://arxiv.org/abs/2207.03508>`_.
        It is **not valid** for waveforms that include higher harmonics or orbital
        precession.  Use at your own discretion when such waveform approximants are employed.

    Attributes:
        gmst (Float): Greenwich Mean Sidereal Time at the trigger time in radians.
        ifo (GroundBased2G): The target detector.
    """

    gmst: Float
    ifo: GroundBased2G

    def __repr__(self):
        return f"GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gmst={self.gmst}, ifo={self.ifo.name})"

    def __init__(
        self,
        trigger_time: Float,
        ifo: GroundBased2G,
    ) -> None:
        """
        Args:
            trigger_time (Float): GPS trigger time in seconds.
            ifo (GroundBased2G): The target detector used to compute the complex
                antenna response.
        """
        name_mapping = (["phase_c"], ["phase_det"])
        conditional_names = ["ra", "dec", "psi", "iota"]
        super().__init__(name_mapping, conditional_names)

        self.gmst = compute_gmst(trigger_time)
        self.ifo = ifo

        assert "phase_c" in name_mapping[0] and "phase_det" in name_mapping[1]
        assert (
            "ra" in conditional_names
            and "dec" in conditional_names
            and "psi" in conditional_names
            and "iota" in conditional_names
        )

        def _calc_R_det_arg(ra, dec, psi, iota, gmst):
            p_iota_term = (1.0 + jnp.cos(iota) ** 2) / 2.0
            c_iota_term = jnp.cos(iota)

            antenna_pattern = self.ifo.antenna_pattern(ra, dec, psi, gmst)
            p_mode_term = p_iota_term * antenna_pattern["p"]
            c_mode_term = c_iota_term * antenna_pattern["c"]

            return jnp.angle(p_mode_term - 1j * c_mode_term)

        def named_transform(x):
            R_det_arg = _calc_R_det_arg(
                x["ra"], x["dec"], x["psi"], x["iota"], self.gmst
            )
            phase_det = R_det_arg + x["phase_c"] / 2.0
            return {
                "phase_det": phase_det % (2.0 * jnp.pi),
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):
            R_det_arg = _calc_R_det_arg(
                x["ra"], x["dec"], x["psi"], x["iota"], self.gmst
            )
            phase_c = -R_det_arg + x["phase_det"] * 2.0
            return {
                "phase_c": phase_c % (2.0 * jnp.pi),
            }

        self.inverse_transform_func = named_inverse_transform


@jaxtyped(typechecker=typechecker)
class DistanceToSNRWeightedDistanceTransform(ConditionalBijectiveTransform):
    """Transform luminosity distance to the network SNR-weighted distance.

    The SNR-weighted distance ``d_hat`` absorbs the network sensitivity and chirp-mass
    dependence into the distance parameter, making it closer to uniform in the
    posterior:

    .. math::

        d_{\\hat} = \\frac{d_L}{\\mathcal{M}_c^{5/6}\\, R_{\\mathrm{net}}}

    Conditioning parameters are ``(M_c, ra, dec, psi, iota)``.

    Warning:
        This transform is derived under the assumption that the waveform consists
        only of the dominant quadrupolar mode (:math:`\\ell = 2, |m| = 2`), following
        the parameterisation in `arXiv:2207.03508 <https://arxiv.org/abs/2207.03508>`_.
        It is **not valid** for waveforms that include higher harmonics or orbital
        precession.  Use at your own discretion when such waveform approximants are employed.

    Attributes:
        gmst (Float): Greenwich Mean Sidereal Time at the trigger time in radians.
        ifos (Sequence[GroundBased2G]): List of detectors forming the network.
    """

    gmst: Float
    ifos: Sequence[GroundBased2G]

    def __repr__(self):
        return f"DistanceToSNRWeightedDistanceTransform(gmst={self.gmst}, ifos={[ifo.name for ifo in self.ifos]})"

    def __init__(
        self,
        trigger_time: Float,
        ifos: Sequence[GroundBased2G],
    ) -> None:
        """
        Args:
            trigger_time (Float): GPS trigger time in seconds.
            ifos (Sequence[GroundBased2G]): Detectors that form the network;
                used to compute the network antenna response ``R_net``.
        """
        name_mapping = (["d_L"], ["d_hat"])
        conditional_names = ["M_c", "ra", "dec", "psi", "iota"]
        super().__init__(name_mapping, conditional_names)

        self.gmst = compute_gmst(trigger_time)
        self.ifos = ifos

        assert "d_L" in name_mapping[0] and "d_hat" in name_mapping[1]
        assert (
            "ra" in conditional_names
            and "dec" in conditional_names
            and "psi" in conditional_names
            and "iota" in conditional_names
            and "M_c" in conditional_names
        )

        def _calc_R_dets(ra, dec, psi, iota):
            p_iota_term = (1.0 + jnp.cos(iota) ** 2) / 2.0
            c_iota_term = jnp.cos(iota)
            R_dets2 = 0.0

            for ifo in self.ifos:
                antenna_pattern = ifo.antenna_pattern(ra, dec, psi, self.gmst)
                p_mode_term = p_iota_term * antenna_pattern["p"]
                c_mode_term = c_iota_term * antenna_pattern["c"]
                R_dets2 += p_mode_term**2 + c_mode_term**2

            return jnp.sqrt(R_dets2)

        def named_transform(x):
            d_L, M_c = (
                x["d_L"],
                x["M_c"],
            )
            R_dets = _calc_R_dets(x["ra"], x["dec"], x["psi"], x["iota"])

            scale_factor = 1.0 / jnp.power(M_c, 5.0 / 6.0) / R_dets
            d_hat = scale_factor * d_L

            return {
                "d_hat": d_hat,
            }

        self.transform_func = named_transform

        def named_inverse_transform(x):
            d_hat, M_c = (
                x["d_hat"],
                x["M_c"],
            )
            R_dets = _calc_R_dets(x["ra"], x["dec"], x["psi"], x["iota"])

            scale_factor = 1.0 / jnp.power(M_c, 5.0 / 6.0) / R_dets

            d_L = d_hat / scale_factor
            return {
                "d_L": d_L,
            }

        self.inverse_transform_func = named_inverse_transform


def named_m1_m2_to_Mc_q(x):
    Mc, q = m1_m2_to_Mc_q(x["m_1"], x["m_2"])
    return {"M_c": Mc, "q": q}


def named_Mc_q_to_m1_m2(x):
    m1, m2 = Mc_q_to_m1_m2(x["M_c"], x["q"])
    return {"m_1": m1, "m_2": m2}


ComponentMassesToChirpMassMassRatioTransform = BijectiveTransform(
    (["m_1", "m_2"], ["M_c", "q"])
)
ComponentMassesToChirpMassMassRatioTransform.transform_func = named_m1_m2_to_Mc_q
ComponentMassesToChirpMassMassRatioTransform.inverse_transform_func = (
    named_Mc_q_to_m1_m2
)


def named_m1_m2_to_Mc_eta(x):
    Mc, eta = m1_m2_to_Mc_eta(x["m_1"], x["m_2"])
    return {"M_c": Mc, "eta": eta}


def named_Mc_eta_to_m1_m2(x):
    m1, m2 = Mc_eta_to_m1_m2(x["M_c"], x["eta"])
    return {"m_1": m1, "m_2": m2}


ComponentMassesToChirpMassSymmetricMassRatioTransform = BijectiveTransform(
    (["m_1", "m_2"], ["M_c", "eta"])
)
ComponentMassesToChirpMassSymmetricMassRatioTransform.transform_func = (
    named_m1_m2_to_Mc_eta
)
ComponentMassesToChirpMassSymmetricMassRatioTransform.inverse_transform_func = (
    named_Mc_eta_to_m1_m2
)


def named_q_to_eta(x):
    return {"eta": q_to_eta(x["q"])}


def named_eta_to_q(x):
    return {"q": eta_to_q(x["eta"])}


MassRatioToSymmetricMassRatioTransform = BijectiveTransform((["q"], ["eta"]))
MassRatioToSymmetricMassRatioTransform.transform_func = named_q_to_eta
MassRatioToSymmetricMassRatioTransform.inverse_transform_func = named_eta_to_q


ChirpMassMassRatioToComponentMassesTransform = reverse_bijective_transform(
    ComponentMassesToChirpMassMassRatioTransform
)


ChirpMassSymmetricMassRatioToComponentMassesTransform = reverse_bijective_transform(
    ComponentMassesToChirpMassSymmetricMassRatioTransform
)


SymmetricMassRatioToMassRatioTransform = reverse_bijective_transform(
    MassRatioToSymmetricMassRatioTransform
)
