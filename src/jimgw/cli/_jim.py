import logging
from typing import Sequence

from jimgw.core.jim import Jim
from jimgw.core.transforms import BijectiveTransform, NtoMTransform

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
    update = {}
    if "checkpoint_dir" not in explicitly_set:
        update["checkpoint_dir"] = output_dir
    if "checkpoint_interval" not in explicitly_set:
        update["checkpoint_interval"] = _CLI_CHECKPOINT_INTERVAL
    if not update:
        return sampler_config
    merged = sampler_config.model_dump() | update
    return sampler_config.__class__.model_validate(merged)


def build_jim(
    likelihood,
    prior,
    sample_transforms: Sequence[BijectiveTransform],
    likelihood_transforms: Sequence[NtoMTransform],
    cfg,
    verbose: bool = False,
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
        verbose=verbose,
    )
    logger.info("Built Jim (sampler=%s, seed=%d)", cfg.sampler.type, cfg.seed)
    return jim
