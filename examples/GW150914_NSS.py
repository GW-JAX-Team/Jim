"""GW150914 analysis with the BlackJAX NSS nested sampler."""

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
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomXAS
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
)
from jimgw.samplers.config import BlackJAXNSSConfig


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
    ifo.set_psd(psd_data.to_psd(nperseg=int(data.duration * data.sampling_frequency)))

# --- Waveform model ---

waveform = RippleIMRPhenomXAS(f_ref=20)

# --- Prior ---

prior = CombinePrior(
    [
        UniformPrior(20.0, 40.0, parameter_names=["M_c"]),
        UniformPrior(0.125, 1.0, parameter_names=["q"]),
        UniformPrior(-0.99, 0.99, parameter_names=["s1_z"]),
        UniformPrior(-0.99, 0.99, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"]),
        UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
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

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    trigger_time=gps,
    f_min=fmin,
    f_max=fmax,
    phase_marginalization=True,
)

# --- Sample ---

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    periodic={
        "psi": (0.0, float(jnp.pi)),
        "azimuth": (0.0, 2 * float(jnp.pi)),
    },
    sampler_config=BlackJAXNSSConfig(
        n_live=2000,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=20,
        termination_dlogz=0.1,
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
    "t_c": r"$t_c\,[\mathrm{s}]$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
}

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T,
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
)
fig.savefig(Path(__file__).parent / "GW150914_NSS.png")
