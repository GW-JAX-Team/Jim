import pytest

pytestmark = pytest.mark.integration

import jax.numpy as jnp
from pathlib import Path

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
)
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_detector_preset
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.samplers.config import FlowMCConfig

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_runs():
    gps = 1126259462.4
    fmin = 20.0
    fmax = 1024.0

    detector_preset = get_detector_preset()
    ifos = [detector_preset["H1"], detector_preset["L1"]]

    for ifo in ifos:
        data = Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
        ifo.set_data(data)
        psd = PowerSpectrum.from_file(
            str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
        )
        ifo.set_psd(psd)

    M_c_min, M_c_max = 10.0, 80.0
    q_min, q_max = 0.125, 1.0
    prior = CombinePrior(
        [
            UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
            UniformPrior(q_min, q_max, parameter_names=["q"]),
            UniformPrior(-1.0, 1.0, parameter_names=["s1_z"]),
            UniformPrior(-1.0, 1.0, parameter_names=["s2_z"]),
            PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"]),
            UniformPrior(-0.05, 0.05, parameter_names=["t_c"]),
            UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
            SinePrior(parameter_names=["iota"]),
            UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
            UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
            CosinePrior(parameter_names=["dec"]),
        ]
    )

    likelihood_transforms = [MassRatioToSymmetricMassRatioTransform]

    likelihood = TransientLikelihoodFD(
        ifos,
        waveform=RippleIMRPhenomD(),
        f_min=fmin,
        f_max=fmax,
        trigger_time=gps,
    )

    jim = Jim(
        likelihood,
        prior,
        sampler_config=FlowMCConfig(
            n_chains=2,
            n_local_steps=2,
            n_global_steps=2,
            global_thinning=1,
            n_training_loops=1,
            n_production_loops=1,
            n_epochs=1,
        ),
        likelihood_transforms=likelihood_transforms,
    )

    jim.sample()
    jim.get_samples()
