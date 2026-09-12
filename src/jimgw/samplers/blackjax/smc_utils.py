"""SMC-specific helpers that aren't part of the sampling algorithm itself.

Currently just the DE reference-ensemble checkpoint trim/restore pair (see
[`jimgw.samplers.blackjax.utils`][] for the checkpoint mechanics these plug
into) — split out from `smc.py` since they're specific to the SMC + DE
combination, not shared with the nested-sampling backends.
"""

from typing import Any

from blackjax.smc import extend_params
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride

from jimgw.samplers.config import BlackJAXSMCConfig


def checkpoint_safe_state(config: BlackJAXSMCConfig, state: Any) -> Any:
    """Drop the DE reference ensemble from ``state`` before checkpointing.

    For ``inner_kernel="DE"`` in the adaptive (inner-kernel-tuning) modes,
    ``state.parameter_override["ensemble"]`` is always an exact copy of
    ``state.sampler_state.particles`` — see
    ``BlackJAXSMCSampler._build_adaptive_inner_kernel_parameters``'s
    ``update_ensemble_parameters``, which sets it to
    ``extend_params({"ensemble": smc_state.particles})`` every step.
    Persisting it doubles the particle population's footprint in every
    checkpoint file for no benefit, since ``restore_checkpoint_ensemble``
    recomputes it from ``sampler_state.particles`` on resume.
    """
    if config.inner_kernel == "DE" and isinstance(state, StateWithParameterOverride):
        return state._replace(
            parameter_override={
                key: value
                for key, value in state.parameter_override.items()
                if key != "ensemble"
            }
        )
    return state


def restore_checkpoint_ensemble(config: BlackJAXSMCConfig, state: Any) -> Any:
    """Reconstruct the DE ensemble dropped by ``checkpoint_safe_state``."""
    if (
        config.inner_kernel == "DE"
        and isinstance(state, StateWithParameterOverride)
        and "ensemble" not in state.parameter_override
    ):
        return state._replace(
            parameter_override={
                **state.parameter_override,
                **extend_params(
                    {"ensemble": state.sampler_state.particles}  # type: ignore[arg-type]
                ),
            }
        )
    return state
