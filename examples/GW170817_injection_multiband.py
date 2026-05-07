"""
GW170817-like injection analysis using MultibandedTransientLikelihoodFD.

This example demonstrates the multibanding likelihood for BNS parameter estimation,
which provides significant speedup (~20x) compared to standard likelihood evaluation.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import MultibandedTransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.transforms import (
    BoundToBound,
    CosineTransform,
    SineTransform,
    PowerLawTransform,
    reverse_bijective_transform,
)
from jimgw.core.single_event.transforms import (
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
)
from jimgw.samplers.config import BlackJAXNSAWConfig

from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()
rng_key = jax.random.PRNGKey(int(total_time_start))
rng_key, *sub_key = jax.random.split(rng_key, 2)

# Simulation parameters
gps = gps_time = total_time_start - 1000
duration = 128.0
fmin = minimum_frequency = 20
fmax = maximum_frequency = 2048
f_ref = fmin
sampling_frequency = fmax * 2

# Initialize waveform (needed for injection)
waveform = RippleIMRPhenomPv2(f_ref=f_ref)

# Generate random sky location
random_samples = jax.random.uniform(sub_key[0], (3,), maxval=jnp.pi)

# Injection parameters for GW170817-like BNS
injection_parameters = {
    "M_c": 1.17,  # chirp mass in solar masses
    "eta": 0.249,  # from q=0.9: eta = q/(1+q)^2
    "s1_x": 0.0,  # zero spins
    "s1_y": 0.0,
    "s1_z": 0.0,
    "s2_x": 0.0,
    "s2_y": 0.0,
    "s2_z": 0.0,
    "ra": 3.44616,
    "dec": -0.408084,
    "psi": 0.0,
    "d_L": 40.0,  # distance in Mpc (approximately GW170817)
    "iota": 0.4,
    "phase_c": jnp.pi - 0.3,
    "t_c": 0.0,
    "trigger_time": gps_time,
}
injection_parameters["gmst"] = compute_gmst(gps_time)

# Convert injection parameters to include spherical spin coordinates
_inj_params = injection_parameters.copy()
q_eta_transform = MassRatioToSymmetricMassRatioTransform
s1_transform = SphereSpinToCartesianSpinTransform("s1")
s2_transform = SphereSpinToCartesianSpinTransform("s2")
_inj_params = q_eta_transform.backward(_inj_params)
_inj_params = s1_transform.backward(_inj_params)
_inj_params = s2_transform.backward(_inj_params)
injection_parameters.update(_inj_params)

print("The injection parameters are")
for key, value in injection_parameters.items():
    print(f"-- {key + ':':10} {float(value):> 13.6f}")
injection_parameters = {
    key: float(value) for key, value in injection_parameters.items()
}

# initialize detectors
ifos = [get_H1(), get_L1(), get_V1()]

print("Setting up detectors with injected signal...")
data_start = time.time()

for ifo in ifos:
    ifo.load_and_set_psd()
    ifo.frequency_bounds = (fmin, fmax)
    ifo.inject_signal(
        duration,
        sampling_frequency,
        0.0,
        waveform,
        injection_parameters,
        f_min=fmin,
        f_max=fmax,
        zero_noise=False,
    )

print(f"Injection setup completed in {time.time() - data_start:.1f}s")

###########################################
########## Set up priors ##################
###########################################

prior = []

# Mass prior - BNS chirp mass range
M_c_min, M_c_max = 1.18, 1.21
q_min, q_max = 0.125, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

# Spin prior - low spins for BNS
s1_prior = UniformSpherePrior(parameter_names=["s1"], max_mag=0.05)
s2_prior = UniformSpherePrior(parameter_names=["s2"], max_mag=0.05)
iota_prior = SinePrior(parameter_names=["iota"])

prior = prior + [
    s1_prior,
    s2_prior,
    iota_prior,
]

# Extrinsic prior
dL_prior = PowerLawPrior(1.0, 75.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = prior + [
    dL_prior,
    t_c_prior,
    phase_c_prior,
    psi_prior,
    ra_prior,
    dec_prior,
]

prior = CombinePrior(prior)

###########################################
########## Set up transforms ##############
###########################################

# Map all parameters to the unit hypercube [0, 1]^n_dims required by NS-AW.
#
# Transform patterns:
#   Uniform [a, b]        → BoundToBound([a, b] → [0, 1])
#   SinePrior  [0, π]     → CosineTransform → BoundToBound([-1, 1] → [0, 1])
#   CosinePrior [-π/2, π/2] → SineTransform → BoundToBound([-1, 1] → [0, 1])
#   PowerLawPrior (α=2)   → reverse_bijective_transform(PowerLawTransform)
sample_transforms = [
    # Masses
    BoundToBound(
        name_mapping=(["M_c"], ["M_c_unit"]),
        original_lower_bound=M_c_min,
        original_upper_bound=M_c_max,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["q"], ["q_unit"]),
        original_lower_bound=q_min,
        original_upper_bound=q_max,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Spin 1
    BoundToBound(
        name_mapping=(["s1_mag"], ["s1_mag_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=0.05,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    CosineTransform(name_mapping=(["s1_theta"], ["cos_s1_theta"])),
    BoundToBound(
        name_mapping=(["cos_s1_theta"], ["cos_s1_theta_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["s1_phi"], ["s1_phi_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Spin 2
    BoundToBound(
        name_mapping=(["s2_mag"], ["s2_mag_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=0.05,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    CosineTransform(name_mapping=(["s2_theta"], ["cos_s2_theta"])),
    BoundToBound(
        name_mapping=(["cos_s2_theta"], ["cos_s2_theta_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["s2_phi"], ["s2_phi_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Inclination (SinePrior → cosine)
    CosineTransform(name_mapping=(["iota"], ["cos_iota"])),
    BoundToBound(
        name_mapping=(["cos_iota"], ["cos_iota_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Luminosity distance (PowerLawPrior α=2 → unit cube)
    reverse_bijective_transform(
        PowerLawTransform(
            name_mapping=(["d_L_unit"], ["d_L"]),
            xmin=1.0,
            xmax=75.0,
            alpha=2.0,
        )
    ),
    # Coalescence time
    BoundToBound(
        name_mapping=(["t_c"], ["t_c_unit"]),
        original_lower_bound=-0.1,
        original_upper_bound=0.1,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Coalescence phase (periodic)
    BoundToBound(
        name_mapping=(["phase_c"], ["phase_c_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Polarization angle (periodic)
    BoundToBound(
        name_mapping=(["psi"], ["psi_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Right ascension (periodic)
    BoundToBound(
        name_mapping=(["ra"], ["ra_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Declination (CosinePrior [-π/2, π/2] → sine → [-1, 1] → [0, 1])
    SineTransform(name_mapping=(["dec"], ["sin_dec"])),
    BoundToBound(
        name_mapping=(["sin_dec"], ["sin_dec_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]

###########################################
########## Set up likelihood ##############
###########################################

print("\nSetting up MultibandedTransientLikelihoodFD...")
likelihood_start = time.time()

# Use the minimum chirp mass from the prior as reference
reference_chirp_mass = M_c_min

likelihood = MultibandedTransientLikelihoodFD(
    detectors=ifos,
    waveform=waveform,
    reference_chirp_mass=reference_chirp_mass,
    f_min=fmin,
    f_max=fmax,
    trigger_time=gps_time,
)

print(f"Likelihood setup completed in {time.time() - likelihood_start:.1f}s")
print("\nMultibanding statistics:")
print(f"  Number of bands: {likelihood.number_of_bands}")
print(f"  Band durations: {[float(d) for d in likelihood.durations]}")
print(f"  Unique frequency points: {len(likelihood.unique_frequencies)}")

# Calculate speedup
full_grid_points = int((fmax - fmin) * duration)
speedup = full_grid_points / len(likelihood.unique_frequencies)
print(f"  Full grid would need: {full_grid_points:,} points")
print(f"  Speedup factor: {speedup:.1f}x")

###########################################
########## Test likelihood ################
###########################################

print("\nTesting likelihood evaluation...")

# Test parameters (approximately GW170817)
test_params = {
    "M_c": 1.186,
    "q": 0.9,
    "s1_mag": 0.01,
    "s1_theta": 0.5,
    "s1_phi": 0.0,
    "s2_mag": 0.01,
    "s2_theta": 0.5,
    "s2_phi": 0.0,
    "d_L": 40.0,
    "t_c": 0.0,
    "phase_c": 0.0,
    "iota": 0.4,
    "psi": 0.0,
    "ra": 3.44,
    "dec": -0.41,
}

# Apply likelihood transforms to get eta and Cartesian spins

# Transform q -> eta
eta_transform = MassRatioToSymmetricMassRatioTransform
test_params_transformed = eta_transform.forward(test_params)

# Transform sphere spins to Cartesian
s1_transform = SphereSpinToCartesianSpinTransform("s1")
s2_transform = SphereSpinToCartesianSpinTransform("s2")
test_params_transformed = s1_transform.forward(test_params_transformed)
test_params_transformed = s2_transform.forward(test_params_transformed)

# Time the likelihood evaluation
eval_start = time.time()
log_L = likelihood.evaluate(test_params_transformed, {})
eval_time = time.time() - eval_start

print(f"  Log-likelihood: {float(log_L):.2f}")
print(f"  Evaluation time: {eval_time*1000:.1f} ms")

# Test multiple evaluations for timing
n_evals = 10
eval_start = time.time()
for _ in range(n_evals):
    _ = likelihood.evaluate(test_params_transformed, {})
avg_eval_time = (time.time() - eval_start) / n_evals

print(f"  Average evaluation time ({n_evals} evals): {avg_eval_time*1000:.1f} ms")

###########################################
########## Run sampling ###################
###########################################

print("\n" + "="*50)
print("Setting up Jim sampler...")
print("="*50)

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXNSAWConfig(
        n_live=1400,
        n_delete_frac=0.5,
        n_target=60,
        max_mcmc=5000,
        termination_dlogz=0.1,
        verbose=True
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    seed=42,
    periodic=["phase_c_unit", "psi_unit", "ra_unit"],
)

print("\nStarting sampling...")
sampling_start = time.time()

jim.sample()
samples = jim.get_samples()
print("Going to save samples:")
np.savez("samples.npz", **samples)
print("Going to save samples DONE.")

sampling_time = time.time() - sampling_start
total_time = time.time() - total_time_start

print("\n" + "="*50)
print("Sampling complete!")
print("="*50)
print(f"Sampling time: {sampling_time/60:.1f} minutes")
print(f"Total time: {total_time/60:.1f} minutes")
