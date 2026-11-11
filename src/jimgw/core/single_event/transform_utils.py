"""Low-level transform utilities for compact-binary parameter spaces.

Three sections, each self-contained:

1. **Mass** — conversions between component masses (m1, m2) and derived
   parameterisations: total mass M, chirp mass Mc, mass ratio q, symmetric
   mass ratio eta.

2. **Spin** — J-frame spin-angle ↔ Cartesian-spin component conversions,
   plus the internal orbital-angular-momentum helper they depend on.

3. **Sky / detector coordinate** — sky-position transforms between the
   equatorial (RA/Dec) and detector-frame (zenith/azimuth) representations,
   including Euler rotation matrix construction.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from jimgw.core.constants import MTSUN
from jimgw.core.utils import carte_to_spherical_angles, safe_arctan2
from jimgw.typing import FloatLike, FloatScalar

# ----------------------------------------------------------------------
# Mass-parameter conversions
# ----------------------------------------------------------------------


def m1_m2_to_M_q(m1: FloatScalar, m2: FloatScalar) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforms the primary mass m1 and secondary mass m2 to the total mass M
    and mass ratio q.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.

    Returns:
        tuple[Float, Float]: Total mass (M_tot) and mass ratio (q).
    """
    M_tot = m1 + m2
    q = m2 / m1
    return M_tot, q


def M_q_to_m1_m2(M_tot: FloatScalar, q: FloatScalar) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforms the total mass M and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Args:
        M_tot (Float): Total mass.
        q (Float): Mass ratio.

    Returns:
        tuple[Float, Float]: Primary mass (m1) and secondary mass (m2).
    """
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


def m1_m2_to_Mc_q(m1: FloatScalar, m2: FloatScalar) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforms the primary mass m1 and secondary mass m2 to the chirp mass M_c
    and mass ratio q.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.

    Returns:
        tuple[Float, Float]: Chirp mass (M_c) and mass ratio (q).
    """
    M_tot = m1 + m2
    eta = m1 * m2 / M_tot**2
    M_c = M_tot * eta ** (3.0 / 5)
    q = m2 / m1
    return M_c, q


def Mc_q_to_m1_m2(M_c: FloatScalar, q: FloatScalar) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforms the chirp mass M_c and mass ratio q to the primary mass m1 and
    secondary mass m2.

    Args:
        M_c (Float): Chirp mass.
        q (Float): Mass ratio.

    Returns:
        tuple[Float, Float]: Primary mass (m1) and secondary mass (m2).
    """
    eta = q / (1 + q) ** 2
    M_tot = M_c / eta ** (3.0 / 5)
    m1 = M_tot / (1 + q)
    m2 = m1 * q
    return m1, m2


def m1_m2_to_M_eta(m1: FloatScalar, m2: FloatScalar) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforms the primary mass m1 and secondary mass m2 to the total mass M
    and symmetric mass ratio eta.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.

    Returns:
        tuple[Float, Float]: Total mass (M) and symmetric mass ratio (eta).
    """
    M_tot = m1 + m2
    eta = m1 * m2 / M_tot**2
    return M_tot, eta


def M_eta_to_m1_m2(
    M_tot: FloatScalar, eta: FloatScalar
) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforms the total mass M and symmetric mass ratio eta to the primary mass m1
    and secondary mass m2.

    Args:
        M_tot (Float): Total mass.
        eta (Float): Symmetric mass ratio.

    Returns:
        tuple[Float, Float]: Primary mass (m1) and secondary mass (m2).
    """
    m1 = M_tot * (1 + jnp.sqrt(1 - 4 * eta)) / 2
    m2 = M_tot * (1 - jnp.sqrt(1 - 4 * eta)) / 2
    return m1, m2


def m1_m2_to_Mc_eta(
    m1: FloatScalar, m2: FloatScalar
) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforms the primary mass m1 and secondary mass m2 to the chirp mass M_c
    and symmetric mass ratio eta.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.

    Returns:
        tuple[Float, Float]: Chirp mass (M_c) and symmetric mass ratio (eta).
    """
    M = m1 + m2
    eta = m1 * m2 / M**2
    M_c = M * eta ** (3.0 / 5)
    return M_c, eta


def Mc_eta_to_m1_m2(
    M_c: FloatScalar, eta: FloatScalar
) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforming the chirp mass M_c and symmetric mass ratio eta to the primary mass m1
    and secondary mass m2.

    Args:
        M_c (Float): Chirp mass.
        eta (Float): Symmetric mass ratio.

    Returns:
        tuple[Float, Float]: Primary mass (m1) and secondary mass (m2).
    """
    M = M_c / eta ** (3.0 / 5)
    m1 = M * (1 + jnp.sqrt(1 - 4 * eta)) / 2
    m2 = M * (1 - jnp.sqrt(1 - 4 * eta)) / 2
    return m1, m2


def q_to_eta(q: FloatScalar) -> FloatScalar:
    """
    Transforming the chirp mass M_c and mass ratio q to the symmetric mass ratio eta.

    Args:
        q (Float): Mass ratio.

    Returns:
        Float: Symmetric mass ratio (eta).
    """
    eta = q / (1 + q) ** 2
    return eta


def eta_to_q(eta: FloatScalar) -> FloatScalar:
    """
    Transforming the symmetric mass ratio eta to the mass ratio q.

    Copied and modified from bilby/gw/conversion.py

    Args:
        eta (Float): Symmetric mass ratio.

    Returns:
        Float: Mass ratio (q).
    """
    temp = 1 / eta / 2 - 1
    return temp - (temp**2 - 1) ** 0.5


# ----------------------------------------------------------------------
# Spin-angle <-> Cartesian-spin conversions
# ----------------------------------------------------------------------


def _Lmag_2PN(m1: FloatScalar, m2: FloatScalar, v0: FloatScalar) -> FloatScalar:
    """
    Compute the magnitude of the orbital angular momentum
    to 2 post-Newtonian orders.

    Args:
        m1 (Float): Primary mass.
        m2 (Float): Secondary mass.
        v0 (Float): Relative velocity at the reference frequency.

    Returns:
        Float: Magnitude of the orbital angular momentum.
    """
    eta = m1 * m2 / (m1 + m2) ** 2
    LN = (m1 + m2) * (m1 + m2) * eta / v0
    L_2PN = 1.5 + eta / 6.0
    return LN * (1.0 + v0 * v0 * L_2PN)


def spin_angles_to_cartesian_spin(
    theta_jn: FloatScalar,
    phi_jl: FloatScalar,
    tilt_1: FloatScalar,
    tilt_2: FloatScalar,
    phi_12: FloatScalar,
    chi_1: FloatScalar,
    chi_2: FloatScalar,
    M_c: FloatScalar,
    q: FloatScalar,
    fRef: FloatLike,
    phiRef: FloatLike,
) -> tuple[
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
]:
    """
    Convert the spin angles to Cartesian spins in the LALSimulation source frame.

    Args:
        theta_jn (Float): Zenith angle between the total angular momentum and the line of sight.
        phi_jl (Float): Difference between total and orbital angular momentum azimuthal angles.
        tilt_1 (Float): Zenith angle between the spin and orbital angular momenta for the primary object.
        tilt_2 (Float): Zenith angle between the spin and orbital angular momenta for the secondary object.
        phi_12 (Float): Difference between the azimuthal angles of the individual spin vector projections
            onto the orbital plane.
        chi_1 (Float): Primary object aligned spin.
        chi_2 (Float): Secondary object aligned spin.
        M_c (Float): The chirp mass.
        q (Float): The mass ratio.
        fRef (Float): The reference frequency.
        phiRef (Float): Binary phase at a reference frequency.

    Returns:
        tuple[Float, Float, Float, Float, Float, Float, Float]: Tuple of (iota, S1x, S1y, S1z, S2x, S2y, S2z):

            - iota: Zenith angle between the orbital angular momentum and the line of sight.
            - S1x: The x-component of the primary spin.
            - S1y: The y-component of the primary spin.
            - S1z: The z-component of the primary spin.
            - S2x: The x-component of the secondary spin.
            - S2y: The y-component of the secondary spin.
            - S2z: The z-component of the secondary spin.
    """
    m1, m2 = Mc_q_to_m1_m2(M_c, q)
    v0 = jnp.cbrt((m1 + m2) * MTSUN * jnp.pi * fRef)

    # Work in a frame with L along z and the primary spin in the x-z plane.
    s1_dir = jnp.array([jnp.sin(tilt_1), 0.0, jnp.cos(tilt_1)])
    s2_dir = jnp.array(
        [
            jnp.sin(tilt_2) * jnp.cos(phi_12),
            jnp.sin(tilt_2) * jnp.sin(phi_12),
            jnp.cos(tilt_2),
        ]
    )
    J = (
        m1 * m1 * chi_1 * s1_dir
        + m2 * m2 * chi_2 * s2_dir
        + jnp.array([0.0, 0.0, _Lmag_2PN(m1, m2, v0)])
    )
    J_dir = J / jnp.linalg.norm(J)
    theta_JL, phi_J = carte_to_spherical_angles(*J_dir)

    # theta_jn and phi_jl are angles in the frame with J along z and N in the y-z
    # plane, where L has azimuth phi_jl. u is the unit vector along the part of L
    # perpendicular to J and w = J x u; both are written out so that they stay
    # defined when L is parallel to J.
    u = jnp.array(
        [
            -jnp.cos(theta_JL) * jnp.cos(phi_J),
            -jnp.cos(theta_JL) * jnp.sin(phi_J),
            jnp.sin(theta_JL),
        ]
    )
    w = jnp.array([jnp.sin(phi_J), -jnp.cos(phi_J), 0.0])
    N = (
        jnp.sin(theta_jn) * (jnp.sin(phi_jl) * u + jnp.cos(phi_jl) * w)
        + jnp.cos(theta_jn) * J_dir
    )
    iota = jnp.arccos(jnp.clip(N[2], -1.0, 1.0))

    # The output frame keeps L along z and puts N at azimuth pi/2 - phiRef, so the
    # spins turn about z by the angle that takes N there.
    alpha = jnp.pi / 2.0 - phiRef - safe_arctan2(N[1], N[0])
    S1 = chi_1 * jnp.array(
        [
            jnp.sin(tilt_1) * jnp.cos(alpha),
            jnp.sin(tilt_1) * jnp.sin(alpha),
            jnp.cos(tilt_1),
        ]
    )
    S2 = chi_2 * jnp.array(
        [
            jnp.sin(tilt_2) * jnp.cos(alpha + phi_12),
            jnp.sin(tilt_2) * jnp.sin(alpha + phi_12),
            jnp.cos(tilt_2),
        ]
    )
    return iota, S1[0], S1[1], S1[2], S2[0], S2[1], S2[2]


def cartesian_spin_to_spin_angles(
    iota: FloatScalar,
    S1x: FloatScalar,
    S1y: FloatScalar,
    S1z: FloatScalar,
    S2x: FloatScalar,
    S2y: FloatScalar,
    S2z: FloatScalar,
    M_c: FloatScalar,
    q: FloatScalar,
    fRef: FloatLike,
    phiRef: FloatLike,
) -> tuple[
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
    FloatScalar,
]:
    """
    Convert Cartesian spins in the LALSimulation source frame to the spin angles.

    Args:
        iota (Float): Zenith angle between the orbital angular momentum and the line of sight.
        S1x (Float): The x-component of the primary spin.
        S1y (Float): The y-component of the primary spin.
        S1z (Float): The z-component of the primary spin.
        S2x (Float): The x-component of the secondary spin.
        S2y (Float): The y-component of the secondary spin.
        S2z (Float): The z-component of the secondary spin.
        M_c (Float): The chirp mass.
        q (Float): The mass ratio.
        fRef (Float): The reference frequency.
        phiRef (Float): The binary phase at the reference frequency.

    Returns:
        tuple[Float, ...]: Tuple of (theta_jn, phi_jl, tilt_1, tilt_2, phi_12, chi_1, chi_2):

            - theta_jn: Zenith angle between the total angular momentum and the line of sight.
            - phi_jl: Difference between total and orbital angular momentum azimuthal angles.
            - tilt_1: Zenith angle between the spin and orbital angular momenta for the primary object.
            - tilt_2: Zenith angle between the spin and orbital angular momenta for the secondary object.
            - phi_12: Difference between the azimuthal angles of the individual spin vector projections onto the orbital plane.
            - chi_1: Primary object aligned spin.
            - chi_2: Secondary object aligned spin.
    """
    s1 = jnp.array([S1x, S1y, S1z])
    s2 = jnp.array([S2x, S2y, S2z])
    chi_1 = jnp.linalg.norm(s1)
    chi_2 = jnp.linalg.norm(s2)
    s1_dir = jnp.where(chi_1 > 0, s1 / chi_1, jnp.zeros_like(s1))
    s2_dir = jnp.where(chi_2 > 0, s2 / chi_2, jnp.zeros_like(s2))
    tilt_1, phi_1 = carte_to_spherical_angles(*s1_dir, default_value=0.0)
    tilt_2, phi_2 = carte_to_spherical_angles(*s2_dir, default_value=0.0)
    phi_12 = jnp.mod(phi_2 - phi_1, 2 * jnp.pi)

    m1, m2 = Mc_q_to_m1_m2(M_c, q)
    v0 = jnp.cbrt((m1 + m2) * MTSUN * jnp.pi * fRef)
    J = m1 * m1 * s1 + m2 * m2 * s2 + jnp.array([0.0, 0.0, _Lmag_2PN(m1, m2, v0)])
    J_dir = J / jnp.linalg.norm(J)

    # L is along z and N is at azimuth pi/2 - phiRef.
    N = jnp.array(
        [
            jnp.sin(iota) * jnp.sin(phiRef),
            jnp.sin(iota) * jnp.cos(phiRef),
            jnp.cos(iota),
        ]
    )
    N_dot_J = jnp.dot(N, J_dir)
    theta_jn = jnp.arccos(jnp.clip(N_dot_J, -1.0, 1.0))

    # phi_jl is the azimuth of L about J, measured in the frame where the part of N
    # perpendicular to J is the y-axis (so x = N x J). Both components below are
    # that frame's components of L times the same positive factor, which arctan2
    # ignores.
    L_y = N[2] - N_dot_J * J_dir[2]
    L_x = N[0] * J_dir[1] - N[1] * J_dir[0]
    phi_jl = jnp.mod(safe_arctan2(L_y, L_x), 2 * jnp.pi)

    return theta_jn, phi_jl, tilt_1, tilt_2, phi_12, chi_1, chi_2


# ----------------------------------------------------------------------
# Sky / detector coordinate transforms
# ----------------------------------------------------------------------


def euler_rotation(delta_x: Float[Array, "3"]) -> Float[Array, "3 3"]:
    """
    Calculate the rotation matrix mapping the vector (0, 0, 1) to delta_x
    while preserving the origin of the azimuthal angle.

    This is decomposed into three Euler angles, alpha, beta, gamma, which rotate
    about the z-, y-, and z- axes respectively.

    Copied and modified from bilby-cython/geometry.pyx
    """
    norm = jnp.linalg.vector_norm(delta_x)

    cos_beta = delta_x[2] / norm
    sin_beta = jnp.sqrt(1 - cos_beta**2)

    alpha = jnp.arctan2(-delta_x[1] * cos_beta, delta_x[0])
    gamma = jnp.arctan2(delta_x[1], delta_x[0])

    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    cos_gamma = jnp.cos(gamma)
    sin_gamma = jnp.sin(gamma)

    rotation = jnp.array(
        [
            [
                cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma,
                -sin_alpha * cos_beta * cos_gamma - cos_alpha * sin_gamma,
                sin_beta * cos_gamma,
            ],
            [
                cos_alpha * cos_beta * sin_gamma + sin_alpha * cos_gamma,
                -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma,
                sin_beta * sin_gamma,
            ],
            [-cos_alpha * sin_beta, sin_alpha * sin_beta, cos_beta],
        ]
    )

    return rotation


def angle_rotation(
    zenith: FloatScalar, azimuth: FloatScalar, rotation: Float[Array, "3 3"]
) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame
    to the polar angle and azimuthal angle in sky frame.

    Modified from bilby-cython/geometry.pyx.

    Args:
        zenith (Float): Zenith angle.
        azimuth (Float): Azimuthal angle.
        rotation (Float[Array, "3 3"]): The rotation matrix.

    Returns:
        tuple[Float, Float]: Polar angle (theta) and azimuthal angle (phi).
    """
    sky_loc_vec = jnp.array(
        [
            jnp.sin(zenith) * jnp.cos(azimuth),
            jnp.sin(zenith) * jnp.sin(azimuth),
            jnp.cos(zenith),
        ]
    )
    rotated_vec = jnp.einsum("ij,j...->i...", rotation, sky_loc_vec)

    theta = jnp.acos(jnp.clip(rotated_vec[2], -1.0, 1.0))
    phi = jnp.fmod(
        jnp.arctan2(rotated_vec[1], rotated_vec[0]) + 2 * jnp.pi,
        2 * jnp.pi,
    )
    return theta, phi


def _theta_phi_to_ra_dec(
    theta: FloatScalar, phi: FloatScalar, gmst: FloatScalar
) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforming the polar angle and azimuthal angle to right ascension and declination.

    Args:
        theta (Float): Polar angle.
        phi (Float): Azimuthal angle.
        gmst (Float): Greenwich mean sidereal time.

    Returns:
        tuple[Float, Float]: Right ascension (ra) and declination (dec).
    """
    ra = phi + gmst
    dec = jnp.pi / 2 - theta
    ra = ra % (2 * jnp.pi)
    return ra, dec


def zenith_azimuth_to_ra_dec(
    zenith: FloatScalar,
    azimuth: FloatScalar,
    gmst: FloatScalar,
    rotation: Float[Array, "3 3"],
) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforming the azimuthal angle and zenith angle in Earth frame to right ascension and declination.

    Copied and modified from bilby/gw/utils.py

    Args:
        zenith (Float): Zenith angle.
        azimuth (Float): Azimuthal angle.
        gmst (Float): Greenwich mean sidereal time.
        rotation (Float[Array, "3 3"]): The rotation matrix.

    Returns:
        tuple[Float, Float]: Right ascension (ra) and declination (dec).
    """
    theta, phi = angle_rotation(zenith, azimuth, rotation)
    ra, dec = _theta_phi_to_ra_dec(theta, phi, gmst)
    return ra, dec


def _ra_dec_to_theta_phi(
    ra: FloatScalar, dec: FloatScalar, gmst: FloatScalar
) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforming the right ascension ra and declination dec to the polar angle
    theta and azimuthal angle phi.

    Args:
        ra (Float): Right ascension.
        dec (Float): Declination.
        gmst (Float): Greenwich mean sidereal time.

    Returns:
        tuple[Float, Float]: Polar angle (theta) and azimuthal angle (phi).
    """
    phi = ra - gmst
    theta = jnp.pi / 2 - dec
    phi = (phi + 2 * jnp.pi) % (2 * jnp.pi)
    return theta, phi


def ra_dec_to_zenith_azimuth(
    ra: FloatScalar,
    dec: FloatScalar,
    gmst: FloatScalar,
    rotation: Float[Array, "3 3"],
) -> tuple[FloatScalar, FloatScalar]:
    """
    Transforming the right ascension and declination to the zenith angle and azimuthal angle.

    Args:
        ra (Float): Right ascension.
        dec (Float): Declination.
        gmst (Float): Greenwich mean sidereal time.
        rotation (Float[Array, "3 3"]): The rotation matrix.

    Returns:
        tuple[Float, Float]: Zenith angle and azimuthal angle.
    """
    theta, phi = _ra_dec_to_theta_phi(ra, dec, gmst)
    zenith, azimuth = angle_rotation(theta, phi, rotation)
    return zenith, azimuth
