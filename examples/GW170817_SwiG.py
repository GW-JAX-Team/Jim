"""GW170817-style 128 s BNS analysis with BlackJAX Nested SwiG.

This example deliberately uses Ripple's built-in aligned-spin
IMRPhenomD_NRTidalv2 model. Coalescence time and phase are analytically
marginalized; the remaining parameters are partitioned into waveform and
projection blocks.
"""

import time
import math

import jax
import jax.numpy as jnp
from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    PowerLawPrior,
    SinePrior,
    UniformPrior,
)
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.samplers.config import BlackJAXSwiGConfig

jax.config.update("jax_enable_x64", True)


gps = 1187008882.43
duration = 128.0
start = gps + 2.0 - duration
end = start + duration
psd_start = start - 2048.0

ifos = [get_H1(), get_L1(), get_V1()]
for ifo in ifos:
    strain = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(strain)
    psd_data = Data.from_gwosc(ifo.name, psd_start, start)
    ifo.set_psd(
        psd_data.to_psd(nperseg=int(strain.duration * strain.sampling_frequency))
    )

distance_prior = PowerLawPrior(30.0, 150.0, 2.0, parameter_names=["d_L"])
prior = CombinePrior(
    [
        UniformPrior(1.18, 1.21, parameter_names=["M_c"]),
        UniformPrior(0.5, 1.0, parameter_names=["q"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s1_z"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s2_z"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
        SinePrior(parameter_names=["iota"]),
        distance_prior,
        UniformPrior(0.0, 2.0 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
    ]
)

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=RippleIMRPhenomD_NRTidalv2(f_ref=20.0),
    trigger_time=gps,
    f_min=20.0,
    f_max=2048.0,
    phase_marginalization=True,
    time_marginalization={"tc_range": (-0.03, 0.03)},
)

jim = Jim(
    likelihood,
    prior,
    likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
    periodic={
        "ra": (0.0, 2.0 * float(jnp.pi)),
        "psi": (0.0, float(jnp.pi)),
    },
    sampler_config=BlackJAXSwiGConfig(
        blocks=[
            ["M_c", "q", "lambda_1", "lambda_2"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["d_L"],
            ["ra", "dec"],
            ["psi"],
        ],
        n_live=512,
        n_delete_frac=0.125,
        num_inner_steps_per_dim=1,
        num_gibbs_sweeps=2,
        termination_dlogz=math.exp(-3.0),
    ),
)

start_time = time.time()
jim.sample()
print(f"Sampling took {(time.time() - start_time) / 60.0:.2f} minutes")
print(jim.get_diagnostics())
