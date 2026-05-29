import logging
from typing import Sequence

from jimgw.core.jim import Jim
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.samplers.config import FlowMCConfig

logger = logging.getLogger(__name__)

_CLI_CHECKPOINT_INTERVAL = 600.0  # 10 minutes


def _with_checkpoint(sampler_config, output_dir):
    """Return a copy of *sampler_config* with checkpointing wired to *output_dir*.

    The CLI always enables checkpointing so that long runs can be resumed.
    The checkpoint file is always ``{output_dir}/checkpoint.pkl``.
    """
    if isinstance(sampler_config, FlowMCConfig):
        return sampler_config.model_copy(
            update={
                "outdir": str(output_dir),
                "checkpoint_interval": _CLI_CHECKPOINT_INTERVAL,
            }
        )
    # BlackJAX configs (BlackJAXNSAWConfig, BlackJAXNSSConfig, BlackJAXSMCConfig)
    # all carry checkpoint_dir via _CheckpointMixin.
    return sampler_config.model_copy(
        update={
            "checkpoint_dir": output_dir,
            "checkpoint_interval": _CLI_CHECKPOINT_INTERVAL,
        }
    )


def build_jim(
    likelihood,
    prior,
    sample_transforms: Sequence[BijectiveTransform],
    likelihood_transforms: Sequence[NtoMTransform],
    cfg,
):
    """Wire together Jim from the fully-built components."""
    sampler_config = _with_checkpoint(cfg.sampler, cfg.output.dir)
    jim = Jim(
        likelihood=likelihood,
        prior=prior,
        sampler_config=sampler_config,
        sample_transforms=sample_transforms,
        likelihood_transforms=likelihood_transforms,
        seed=cfg.seed,
    )
    logger.info("Built Jim (sampler=%s, seed=%d)", cfg.sampler.type, cfg.seed)
    return jim
