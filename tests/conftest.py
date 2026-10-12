import pytest
from jax import config
from jax.experimental.compilation_cache import compilation_cache

config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def restore_compilation_cache_directory():
    """Keep a sampler's temporary cache settings from leaking into later tests."""
    original_dir = config.jax_compilation_cache_dir
    original_min_time = config.jax_persistent_cache_min_compile_time_secs
    yield
    compilation_cache.reset_cache()
    config.update("jax_compilation_cache_dir", original_dir)
    config.update("jax_persistent_cache_min_compile_time_secs", original_min_time)
