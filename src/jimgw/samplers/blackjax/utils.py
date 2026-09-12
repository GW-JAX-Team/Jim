"""Checkpoint save/load/cleanup helpers shared by the BlackJAX SMC and nested-sampling backends.

``BlackJAXSMCSampler``, ``BlackJAXNSSSampler``, and ``BlackJAXNSAWSampler`` each
run their own JAX while-loop and periodically checkpoint it to
``config.checkpoint_dir/checkpoint.pkl``. The load/save/cleanup mechanics
(compute the path, try to resume, fall back to a fresh state on a corrupt or
foreign checkpoint, gate saves on the configured interval, clean up on
completion) are identical across the three; only the state shape, the extra
per-run bookkeeping (SMC's acceptance/covariance histories vs. NS's ``dead``
point list), and a couple of backend-specific hooks differ. This module holds
that shared mechanics; each sampler supplies the differences via callbacks.
"""

import logging
import pickle
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import jax
from jaxtyping import Key

from jimgw.samplers.base import Sampler
from jimgw.samplers.config import _CheckpointMixin

logger = logging.getLogger(__name__)


def prepare_checkpointing(config: _CheckpointMixin) -> tuple[Optional[Path], float]:
    """Enable the JAX compile cache and return the checkpoint path and start time."""
    checkpoint_path = (
        config.checkpoint_dir / "checkpoint.pkl"
        if config.checkpoint_dir is not None
        else None
    )
    config.configure_jax_cache()
    return checkpoint_path, time.perf_counter()


def load_or_initialize_checkpoint(
    sampler: Sampler,
    checkpoint_path: Optional[Path],
    config: _CheckpointMixin,
    rng_key: Key,
    initial_particles: Any,
    initialize_state: Callable[..., Any],
    initial_extra: dict[str, Any],
    load_extra: Callable[[dict], dict[str, Any]],
    *,
    log_label: str,
    after_load: Optional[Callable[[Any], Any]] = None,
    is_stale: Optional[Callable[[int], bool]] = None,
    stale_message: str = "checkpoint exceeds current schedule",
) -> tuple[Any, Key, int, dict[str, Any]]:
    """Load a compatible checkpoint or initialize a new sampler state.

    ``initialize_state`` builds a fresh state from ``initial_particles``.
    ``initial_extra``/``load_extra`` supply and restore backend-specific
    extras (histories for SMC, the NS ``dead`` point list, ...). ``after_load``
    post-processes a freshly loaded ``state`` — used for device/mesh placement
    (NSS) or other backend-specific reconstruction. ``is_stale`` optionally
    rejects a loaded checkpoint whose ``n_iter`` no longer fits the current
    run (a fixed temperature ladder that has since changed length) and falls
    back to a fresh state.

    Only trusted checkpoint files may be loaded because pickle can execute
    arbitrary code.
    """
    if not (
        checkpoint_path is not None
        and config.checkpoint_interval > 0
        and checkpoint_path.exists()
    ):
        sampler._prev_elapsed = 0.0
        return (
            initialize_state(initial_particles),
            rng_key,
            0,
            dict(initial_extra),
        )

    initial_rng_key = rng_key
    try:
        with open(checkpoint_path, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        sampler._validate_checkpoint(checkpoint)
        state = checkpoint["state"]
        rng_key = checkpoint["rng_key"]
        n_completed_iterations = checkpoint["n_iter"]
        extra = load_extra(checkpoint)
        if is_stale is not None and is_stale(n_completed_iterations):
            logger.warning(
                "%s: %s (n_iter=%d) — starting fresh.",
                log_label,
                stale_message,
                n_completed_iterations,
            )
            rng_key = initial_rng_key
            state = initialize_state(initial_particles)
            n_completed_iterations = 0
            extra = dict(initial_extra)
            sampler._prev_elapsed = 0.0
        else:
            sampler._prev_elapsed = float(checkpoint["elapsed_time"])
            if after_load is not None:
                state = after_load(state)
            logger.info(
                "%s: resumed from checkpoint at n_iter=%d (%s)",
                log_label,
                n_completed_iterations,
                checkpoint_path,
            )
        return state, rng_key, n_completed_iterations, extra
    except (
        OSError,
        EOFError,
        KeyError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        logger.warning(
            "%s: incompatible or corrupt checkpoint at %s (%s) — starting fresh.",
            log_label,
            checkpoint_path,
            error,
        )
        sampler._prev_elapsed = 0.0
        return (
            initialize_state(initial_particles),
            initial_rng_key,
            0,
            dict(initial_extra),
        )


def save_checkpoint_if_due(
    sampler: Sampler,
    config: _CheckpointMixin,
    checkpoint_path: Optional[Path],
    last_checkpoint_write_time: float,
    run_start_time: float,
    state: Any,
    rng_key: Key,
    n_completed_iterations: int,
    extra: dict[str, Any],
    *,
    log_label: str,
    before_save: Optional[
        Callable[[Any, Key, dict[str, Any]], tuple[Any, Key, dict[str, Any]]]
    ] = None,
) -> float:
    """Save a checkpoint when its interval has elapsed.

    ``extra`` is merged into the common checkpoint payload (state, rng_key,
    n_iter, sampler_name, elapsed_time) — use it for backend-specific fields
    such as per-iteration histories or the NS ``dead`` point list.
    ``before_save`` transforms ``(state, rng_key, extra)`` right before they
    are written — e.g. NSS's ``jax.device_get`` on all three. It is called
    only when a checkpoint is actually due, so a host transfer here doesn't
    run every iteration. Returns the last checkpoint time.
    """
    if not (
        checkpoint_path is not None
        and config.checkpoint_interval > 0
        and time.perf_counter() - last_checkpoint_write_time
        >= config.checkpoint_interval
    ):
        return last_checkpoint_write_time
    if before_save is not None:
        state, rng_key, extra = before_save(state, rng_key, extra)
    return config.write_checkpoint(
        {
            "state": state,
            "rng_key": rng_key,
            "n_iter": n_completed_iterations,
            "sampler_name": sampler.sampler_name,
            "elapsed_time": sampler._prev_elapsed
            + (time.perf_counter() - run_start_time),
            **extra,
        },
        log_label,
    )


def remove_checkpoint_and_jax_cache(
    config: _CheckpointMixin, checkpoint_path: Optional[Path]
) -> None:
    """Remove a completed run's checkpoint, JAX cache, and cache setting."""
    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)
    if config.checkpoint_dir is not None:
        shutil.rmtree(config.checkpoint_dir / "jax_cache", ignore_errors=True)
        jax.config.update("jax_compilation_cache_dir", None)
