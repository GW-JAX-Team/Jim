"""Tests for shared BlackJAX checkpoint and compilation-cache helpers."""

import warnings

import jax
import jax.numpy as jnp

from jimgw.samplers.blackjax.utils import remove_checkpoint_and_jax_cache
from jimgw.samplers.config import BlackJAXSMCConfig


def test_sequential_runs_use_their_own_compilation_caches(tmp_path):
    first = BlackJAXSMCConfig(checkpoint_dir=tmp_path / "first")
    second = BlackJAXSMCConfig(checkpoint_dir=tmp_path / "second")
    compiled = jax.jit(lambda x: jnp.sin(x) + 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        first.configure_jax_cache()
        compiled(jnp.arange(7.0)).block_until_ready()
        first_cache = first.checkpoint_dir / "jax_cache"
        assert any(first_cache.iterdir())

        remove_checkpoint_and_jax_cache(first, None)
        assert not first_cache.exists()
        assert jax.config.jax_compilation_cache_dir is None
        # A program compiled now must not try to write into the removed directory.
        compiled(jnp.arange(9.0)).block_until_ready()

        second.configure_jax_cache()
        compiled(jnp.arange(11.0)).block_until_ready()
        second_cache = second.checkpoint_dir / "jax_cache"
        assert any(second_cache.iterdir())


def test_cleaning_up_one_run_preserves_another_runs_cache(tmp_path):
    first = BlackJAXSMCConfig(checkpoint_dir=tmp_path / "first")
    second = BlackJAXSMCConfig(checkpoint_dir=tmp_path / "second")
    compiled = jax.jit(lambda x: jnp.cos(x) + 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        first.configure_jax_cache()
        compiled(jnp.arange(13.0)).block_until_ready()
        second.configure_jax_cache()
        compiled(jnp.arange(17.0)).block_until_ready()
        second_cache = second.checkpoint_dir / "jax_cache"
        assert any(second_cache.iterdir())

        remove_checkpoint_and_jax_cache(first, None)
        assert jax.config.jax_compilation_cache_dir == str(second_cache)
        compiled(jnp.arange(19.0)).block_until_ready()
        assert second_cache.exists()
