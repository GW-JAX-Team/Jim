"""Injection-recovery test with the flowMC sampler."""

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
from jimgw.core.single_event.waveform import RippleIMRPhenomXAS
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
)
from jimgw.samplers.config import FlowMCConfig


# --- Waveform model ---

waveform = RippleIMRPhenomXAS(f_ref=20)

# --- Inject signal ---

gps = time.time() - 1000
random_samples = jax.random.uniform(jax.random.key(0), (3,), maxval=jnp.pi)

# Injection parameters in likelihood space.
injection_parameters = {
    "M_c": 30.0,
    "eta": 0.24,
    "s1_z": 0.3,
    "s2_z": -0.2,
    "ra": random_samples[0] * 2.0,
    "dec": random_samples[1] - jnp.pi / 2,
    "psi": random_samples[2],
    "d_L": 500.0,
    "iota": 0.5,
    "phase_c": jnp.pi - 0.3,
    "t_c": 0.03,
}

print("The injection parameters are")
for key, value in injection_parameters.items():
    print(f"-- {key + ':':10} {float(value):> 13.6f}")

f_min = 20.0
f_max = 1024.0
duration = 4.0
sampling_frequency = f_max * 2

ifos = [get_H1(), get_L1()]

for ifo in ifos:
    ifo.load_and_set_psd()

    # inject the signal
    ifo.inject_signal(
        duration=duration,
        sampling_frequency=sampling_frequency,
        trigger_time=gps,
        waveform_model=waveform,
        parameters=injection_parameters,
        f_min=f_min,
        f_max=f_max,
        zero_noise=False,
    )

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
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
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
    f_min=f_min,
    f_max=f_max,
)

# --- Sample ---

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    periodic={
        "phase_c": (0.0, 2 * float(jnp.pi)),
        "psi": (0.0, float(jnp.pi)),
        "azimuth": (0.0, 2 * float(jnp.pi)),
    },
    sampler_config=FlowMCConfig(
        n_chains=1000,
        n_local_steps=100,
        n_global_steps=1000,
        n_training_loops=50,
        n_production_loops=10,
        n_NFproposal_batch_size=100,
        global_thinning=100,
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
    "s1_z": r"$s_{1,z}$",
    "s2_z": r"$s_{2,z}$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$",
    "phase_c": r"$\phi_c$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
}

truth_values = injection_parameters.copy()
for transform in reversed(likelihood_transforms):
    truth_values = transform.backward(truth_values)
truths = [float(truth_values[k]) for k in jim.prior.parameter_names]

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
    truths=truths,
)
fig.savefig(Path(__file__).parent / "injection.png")
