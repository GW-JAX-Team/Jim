"""GW170817 analysis with the flowMC sampler."""

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
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import IMRPhenomXAS_NRTidalv3
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
)
from jimgw.samplers.config import FlowMCConfig


# --- Fetch data ---

# fetch a 128s segment centered on GW170817
gps = 1187008882.43
duration = 128.0
# Request a segment with 2.0 s post-merger
start = gps + 2.0 - duration
end = start + duration

# fetch 2048s of data to estimate the PSD (4096s has a gap in V1; 2048s is clean for all three)
psd_start = start - 2048
psd_end = start

# set the frequency range for the analysis
fmin = 20.0
fmax = 2048.0

# initialize detectors
ifos = [get_H1(), get_L1(), get_V1()]

for ifo in ifos:
    strain_data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(strain_data)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    ifo.set_psd(
        psd_data.to_psd(nperseg=strain_data.duration * strain_data.sampling_frequency)
    )

# --- Waveform model ---

waveform = IMRPhenomXAS_NRTidalv3(f_ref=20)

# --- Prior ---

prior = CombinePrior(
    [
        UniformPrior(1.18, 1.21, parameter_names=["M_c"]),
        UniformPrior(0.125, 1.0, parameter_names=["q"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s1_z"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(1.0, 100.0, 2.0, parameter_names=["d_L"]),
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

sample_transforms = [
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(trigger_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(trigger_time=gps, ifos=ifos),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
]

# --- Likelihood ---

likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    n_bins=1000,
    trigger_time=gps,
    f_min=fmin,
    f_max=fmax,
    prior=prior,
    likelihood_transforms=likelihood_transforms,
)

# --- Sample ---

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    sampler_config=FlowMCConfig(
        n_chains=1000,
        n_local_steps=100,
        n_global_steps=1000,
        n_training_loops=50,
        n_production_loops=10,
        n_NFproposal_batch_size=100,
        global_thinning=100,
        periodic={
            "phase_c": (0.0, 2 * float(jnp.pi)),
            "psi": (0.0, float(jnp.pi)),
            "azimuth": (0.0, 2 * float(jnp.pi)),
        },
        verbose=True,
    ),
)

start_time = time.time()
jim.sample()
end_time = time.time()
print(f"Sampling took {(end_time - start_time) / 60:.2f} mins")

# --- Results ---

diagnostics = jim.get_diagnostics()
print(f"Likelihood evaluations: {diagnostics['n_likelihood_evaluations']:,}")

chains = jim.get_samples()

parameter_labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "q": r"$q$",
    "s1_z": r"$\chi_1$",
    "s2_z": r"$\chi_2$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$",
    "phase_c": r"$\phi_c$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
}

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
)
fig.savefig(Path(__file__).parent / "GW170817.png")
