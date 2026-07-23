"""GW170817 analysis with the BlackJAX SwiG sampler."""

import time
from pathlib import Path

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
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
)
from jimgw.samplers.config import BlackJAXSwiGConfig


# --- Fetch data ---

gps = 1187008882.43
duration = 128.0
start = gps + 2.0 - duration
end = start + duration

psd_start = start - 2048
psd_end = start

fmin = 20.0
fmax = 2048.0

ifos = [get_H1(), get_L1(), get_V1()]

for ifo in ifos:
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)

    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    ifo.set_psd(psd_data.to_psd(nperseg=int(data.duration * data.sampling_frequency)))

# --- Waveform model ---

# Ripple's built-in aligned-spin tidal model.
waveform = RippleIMRPhenomD_NRTidalv2(f_ref=20.0)

# --- Prior ---

# t_c and phase_c are analytically marginalized below, so neither is a
# sampled (or sampling-space) parameter and both are omitted here.
prior = CombinePrior(
    [
        UniformPrior(1.18, 1.21, parameter_names=["M_c"]),
        UniformPrior(0.5, 1.0, parameter_names=["q"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s1_z"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s2_z"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(30.0, 150.0, 2.0, parameter_names=["d_L"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
    ]
)

# --- Transforms ---

sample_transforms = [
    SkyFrameToDetectorFrameSkyPositionTransform(trigger_time=gps, ifos=ifos),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
]

# --- Likelihood ---

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    trigger_time=gps,
    f_min=fmin,
    f_max=fmax,
    phase_marginalization=True,
    time_marginalization={"tc_range": (-0.03, 0.03)},
)

# --- Sample ---

# Blocks are given in sampling-space names: intrinsic/waveform-affecting
# parameters ("M_c", "q", "lambda_1", "lambda_2", "iota", "d_L") refresh the
# waveform cache; detector-projection-only blocks ("zenith"/"azimuth", "psi")
# reuse it.
jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    periodic={
        "psi": (0.0, float(jnp.pi)),
        "azimuth": (0.0, 2 * float(jnp.pi)),
    },
    sampler_config=BlackJAXSwiGConfig(
        blocks=[
            ["M_c", "q", "lambda_1", "lambda_2"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["d_L"],
            ["zenith", "azimuth"],
            ["psi"],
        ],
        n_live=512,
        n_delete_frac=0.125,
        num_inner_steps_per_dim=1,
        num_gibbs_sweeps=2,
    ),
    verbose=True,
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
    "s1_z": r"$\chi_1$",
    "s2_z": r"$\chi_2$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "psi": r"$\psi$",
}

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T,
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
)
fig.savefig(Path(__file__).parent / "GW170817_SwiG.png")
