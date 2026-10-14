"""Parameter-name constants shared across the CLI pipeline.

These frozensets and tuples name the recognised parameter groups.  They are
intentionally JAX-free so they can be imported at config-parse time.
"""

# J-frame precessing-spin angles (all 7 required together)
J_FRAME_SPIN_PARAMS = frozenset(
    {"theta_jn", "phi_jl", "tilt_1", "tilt_2", "phi_12", "a_1", "a_2"}
)

# Labels for the two per-body spherical spin vectors (s1, s2)
SPHERE_SPIN_LABELS: tuple[str, ...] = ("s1", "s2")

# Aligned (z-only) spin components
ALIGNED_SPIN_PARAMS = frozenset({"s1_z", "s2_z"})

# Full 3-D Cartesian spin components (includes aligned)
CARTESIAN_SPIN_PARAMS = frozenset({"s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z"})

# Sky-position parametrizations
EQUATORIAL_SKY_PARAMS = frozenset({"ra", "dec"})
DETECTOR_SKY_PARAMS = frozenset({"azimuth", "zenith"})

# Detector names supported by get_detector_preset()
SUPPORTED_DETECTORS = frozenset({"H1", "L1", "V1", "ET", "CE"})
