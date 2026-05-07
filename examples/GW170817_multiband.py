"""GW170817 analysis with MultibandedTransientLikelihoodFD and NS-AW sampler.

Combines:
- Real GW170817 data loaded from pre-processed txt files (re/im + freq + PSD),
  IMRPhenomXAS_NRTidalv3 waveform, and tidal priors.
- MultibandedTransientLikelihoodFD + BlackJAXNSAWConfig + unit-cube transforms.

FIXME: The data is stored on a local cluster so other users cannot run this example.
Need to consider storing the data somewhere public (Zenodo?) and updating the code to load from there.
Leaving it as is for now for personal testing

Data files are expected at DATA_PATH (set below).  On the cluster the path is
  /projects/prjs1678/gw-datasets/GW170817/
and locally the files live at
  hackathon_multibanding/turbope-bns/data/GW170817/
"""

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
)
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import MultibandedTransientLikelihoodFD
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.waveform import IMRPhenomXAS_NRTidalv3
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.core.transforms import (
    BoundToBound,
    CosineTransform,
    SineTransform,
    PowerLawTransform,
    reverse_bijective_transform,
)
from jimgw.samplers.config import BlackJAXNSAWConfig

# --- Data path ---
# Switch to the cluster path when running on the cluster.
DATA_PATH = Path("/Users/Woute029/Documents/Code/projects/25_jim_hackathon/hackathon_multibanding/turbope-bns/data/GW170817")
# DATA_PATH = Path("/projects/prjs1678/gw-datasets/GW170817")

# --- Event parameters ---

gps = 1187008882.43
duration = 128.0
fmin = 20.0
fmax = 2048.0

# --- Load pre-processed data and PSDs from txt files ---

ifos = [get_H1(), get_L1(), get_V1()]

print("Loading data and PSDs from files...")

for ifo in ifos:
    name = ifo.name  # "H1", "L1", or "V1"
    freqs = np.genfromtxt(DATA_PATH / f"{name}_freq.txt")
    data_fd = (
        np.genfromtxt(DATA_PATH / f"{name}_data_re.txt")
        + 1j * np.genfromtxt(DATA_PATH / f"{name}_data_im.txt")
    )
    psd_freqs, psd_vals = np.loadtxt(
        DATA_PATH / f"GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_{name}_psd.txt",
        unpack=True,
    )

    strain_data = Data.from_fd(
        jnp.array(data_fd),
        jnp.array(freqs),
        start_time=gps + 2.0 - duration,
        name=name,
    )
    ifo.set_data(strain_data)

    psd = PowerSpectrum(jnp.array(psd_vals), jnp.array(psd_freqs), name=f"{name}_psd")
    ifo.set_psd(psd)

print("Data and PSDs loaded successfully.")

# --- Waveform model ---

waveform = IMRPhenomXAS_NRTidalv3(f_ref=20.0)

# --- Prior ---

M_c_min, M_c_max = 1.18, 1.21
q_min, q_max = 0.125, 1.0
d_L_min, d_L_max = 1.0, 100.0

prior = CombinePrior(
    [
        UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
        UniformPrior(q_min, q_max, parameter_names=["q"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s1_z"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(d_L_min, d_L_max, 2.0, parameter_names=["d_L"]),
        UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
    ]
)

# --- Transforms ---
# Map all parameters to the unit hypercube [0, 1]^n_dims required by NS-AW.
#
# Transform patterns:
#   Uniform [a, b]          → BoundToBound([a, b] → [0, 1])
#   SinePrior  [0, π]       → CosineTransform → BoundToBound([-1, 1] → [0, 1])
#   CosinePrior [-π/2, π/2] → SineTransform   → BoundToBound([-1, 1] → [0, 1])
#   PowerLawPrior (α=2)     → reverse_bijective_transform(PowerLawTransform)

sample_transforms = [
    # Chirp mass
    BoundToBound(
        name_mapping=(["M_c"], ["M_c_unit"]),
        original_lower_bound=M_c_min,
        original_upper_bound=M_c_max,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Mass ratio
    BoundToBound(
        name_mapping=(["q"], ["q_unit"]),
        original_lower_bound=q_min,
        original_upper_bound=q_max,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Aligned spins
    BoundToBound(
        name_mapping=(["s1_z"], ["s1_z_unit"]),
        original_lower_bound=-0.05,
        original_upper_bound=0.05,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["s2_z"], ["s2_z_unit"]),
        original_lower_bound=-0.05,
        original_upper_bound=0.05,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Inclination (SinePrior [0, π] → cosine ∈ [-1, 1] → [0, 1])
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
            xmin=d_L_min,
            xmax=d_L_max,
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
    # Declination (CosinePrior [-π/2, π/2] → sine ∈ [-1, 1] → [0, 1])
    SineTransform(name_mapping=(["dec"], ["sin_dec"])),
    BoundToBound(
        name_mapping=(["sin_dec"], ["sin_dec_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Tidal deformabilities
    BoundToBound(
        name_mapping=(["lambda_1"], ["lambda_1_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=5000.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["lambda_2"], ["lambda_2_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=5000.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
]

# --- Likelihood ---

print("Setting up MultibandedTransientLikelihoodFD...")
likelihood_start = time.time()

likelihood = MultibandedTransientLikelihoodFD(
    detectors=ifos,
    waveform=waveform,
    reference_chirp_mass=M_c_min,
    f_min=fmin,
    f_max=fmax,
    trigger_time=gps,
)

print(f"Likelihood setup completed in {time.time() - likelihood_start:.1f}s")
print(f"  Number of bands: {likelihood.number_of_bands}")
print(f"  Unique frequency points: {len(likelihood.unique_frequencies)}")

full_grid_points = int((fmax - fmin) * duration)
speedup = full_grid_points / len(likelihood.unique_frequencies)
print(f"  Full grid would need: {full_grid_points:,} points")
print(f"  Speedup factor: {speedup:.1f}x")

# --- Sample ---

jim = Jim(
    likelihood,
    prior,
    sampler_config=BlackJAXNSAWConfig(
        n_live=1400,
        n_delete_frac=0.5,
        n_target=60,
        max_mcmc=5000,
        termination_dlogz=0.1,
        verbose=True,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    seed=42,
    periodic=["phase_c_unit", "psi_unit", "ra_unit"],
)

print("\nStarting sampling...")

start_time = time.time()
jim.sample()
end_time = time.time()
print(f"Sampling took {(end_time - start_time) / 60:.2f} mins")

# --- Results ---

samples = jim.get_samples()
np.savez(str(Path(__file__).parent / "GW170817_multiband_samples.npz"), **samples)
print("Samples saved to GW170817_multiband_samples.npz")
