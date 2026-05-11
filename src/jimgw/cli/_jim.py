import logging
from typing import Sequence

from jimgw.core.jim import Jim
from jimgw.core.transforms import BijectiveTransform, NtoMTransform

logger = logging.getLogger(__name__)


def build_jim(
    likelihood,
    prior,
    sample_transforms: Sequence[BijectiveTransform],
    likelihood_transforms: Sequence[NtoMTransform],
    cfg,
):
    """Wire together Jim from the fully-built components."""
    jim = Jim(
        likelihood=likelihood,
        prior=prior,
        sampler_config=cfg.sampler,
        sample_transforms=sample_transforms,
        likelihood_transforms=likelihood_transforms,
        seed=cfg.seed,
    )
    logger.info("Built Jim (sampler=%s, seed=%d)", cfg.sampler.type, cfg.seed)
    return jim
