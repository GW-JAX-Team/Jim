"""GW150914 analysis with the BlackJAX NS-AW nested sampler."""

import time
from pathlib import Path

# Plotting requires the visualize extra: pip install jimgw[visualize]
import corner
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
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomXAS
from jimgw.core.single_event.transforms import (
    MassRatioToSymmetricMassRatioTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
)
from jimgw.core.transforms import (
    BoundToBound,
    CosineTransform,
    PowerLawTransform,
    reverse_bijective_transform,
)
from jimgw.samplers.config import BlackJAXNSAWConfig


# --- Fetch data ---

gps = 1126259462.4
duration = 4.0
start = gps + 2.0 - duration
end = start + duration

psd_start = start - 2048
psd_end = start

fmin = 20.0
fmax = 896.0

ifos = [get_H1(), get_L1()]

for ifo in ifos:
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    ifo.set_psd(psd_data.to_psd(nperseg=data.duration * data.sampling_frequency))

# --- Waveform model ---

waveform = RippleIMRPhenomXAS(f_ref=20)

# --- Prior ---

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
d_L_min, d_L_max = 1.0, 2000.0
t_det_min, t_det_max = -0.1, 0.1

prior = CombinePrior(
    [
        UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
        UniformPrior(q_min, q_max, parameter_names=["q"]),
        UniformPrior(-0.99, 0.99, parameter_names=["s1_z"]),
        UniformPrior(-0.99, 0.99, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(d_L_min, d_L_max, 2.0, parameter_names=["d_L"]),
        UniformPrior(t_det_min, t_det_max, parameter_names=["t_det"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
    ]
)

# --- Transforms ---
#
# Each parameter is mapped to [0, 1].  Transform patterns:
#   Uniform [a, b]           → BoundToBound([a, b] → [0, 1])
#   SinePrior  [0, π]        → CosineTransform → BoundToBound([-1, 1] → [0, 1])
#   PowerLawPrior (α=2)      → reverse_bijective_transform(PowerLawTransform)

sample_transforms = [
    SkyFrameToDetectorFrameSkyPositionTransform(trigger_time=gps, ifos=ifos),
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
        name_mapping=(["s1_z"], ["s1_z_unit"]),
        original_lower_bound=-0.99,
        original_upper_bound=0.99,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Spin 2
    BoundToBound(
        name_mapping=(["s2_z"], ["s2_z_unit"]),
        original_lower_bound=-0.99,
        original_upper_bound=0.99,
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
            xmin=d_L_min,
            xmax=d_L_max,
            alpha=2.0,
        )
    ),
    # Coalescence time
    BoundToBound(
        name_mapping=(["t_det"], ["t_det_unit"]),
        original_lower_bound=t_det_min,
        original_upper_bound=t_det_max,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Phase and polarization angle
    BoundToBound(
        name_mapping=(["phase_c"], ["phase_c_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    BoundToBound(
        name_mapping=(["psi"], ["psi_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    # Sky position — azimuth and zenith
    BoundToBound(
        name_mapping=(["azimuth"], ["azimuth_unit"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
    CosineTransform(name_mapping=(["zenith"], ["cos_zenith"])),
    BoundToBound(
        name_mapping=(["cos_zenith"], ["cos_zenith_unit"]),
        original_lower_bound=-1.0,
        original_upper_bound=1.0,
        target_lower_bound=0.0,
        target_upper_bound=1.0,
    ),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    reverse_bijective_transform(
        GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time=gps, ifo=ifos[0]
        )
    ),
]

# --- Likelihood ---

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    trigger_time=gps,
    f_min=fmin,
    f_max=fmax,
)

# --- Sample ---

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    sampler_config=BlackJAXNSAWConfig(
        n_live=1000,
        n_delete_frac=0.5,
        n_target=60,
        periodic=["phase_c_unit", "psi_unit", "azimuth_unit"],
    ),
)

start_time = time.time()
jim.sample()
end_time = time.time()
print(f"Sampling took {(end_time - start_time) / 60:.2f} mins")

# --- Results ---

diagnostics = jim.get_diagnostics()
print(f"log Z = {diagnostics['log_Z']:.2f} ± {diagnostics['log_Z_error']:.2f}")
print(f"Likelihood evaluations: {diagnostics['n_likelihood_evaluations']:,}")

chains = jim.get_samples()

parameter_labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "q": r"$q$",
    "s1_z": r"$s_{1,z}$",
    "s2_z": r"$s_{2,z}$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_det": r"$t_{\mathrm{det}}\,[\mathrm{s}]$",
    "phase_c": r"$\phi_c$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
}

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T,
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
)
fig.savefig(Path(__file__).parent / "GW150914_NS_AW.png")
