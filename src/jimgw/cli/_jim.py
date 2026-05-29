import logging
from typing import Sequence

from jimgw.core.jim import Jim
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.samplers.config import FlowMCConfig

logger = logging.getLogger(__name__)

_CLI_CHECKPOINT_INTERVAL = 600.0  # 10 minutes


def _with_checkpoint(sampler_config, output_dir):
    """Return a copy of *sampler_config* with CLI checkpoint defaults applied.

    Only fields the user did not explicitly set are filled in, so explicit
    values in the config file are always respected.  The CLI defaults are:
    checkpoint every ``_CLI_CHECKPOINT_INTERVAL`` seconds, writing to
    ``{output_dir}/checkpoint.pkl``.
    """
    explicitly_set = sampler_config.model_fields_set
    if isinstance(sampler_config, FlowMCConfig):
        update = {}
        if "outdir" not in explicitly_set:
            update["outdir"] = str(output_dir)
        if "checkpoint_interval" not in explicitly_set:
            update["checkpoint_interval"] = _CLI_CHECKPOINT_INTERVAL
    else:
        # BlackJAX configs carry checkpoint_dir via _CheckpointMixin.
        update = {}
        if "checkpoint_dir" not in explicitly_set:
            update["checkpoint_dir"] = output_dir
        if "checkpoint_interval" not in explicitly_set:
            update["checkpoint_interval"] = _CLI_CHECKPOINT_INTERVAL
    return sampler_config.model_copy(update=update) if update else sampler_config


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
