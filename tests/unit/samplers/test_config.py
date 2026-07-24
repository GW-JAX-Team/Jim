import warnings

import jax
import numpy as np
import pytest
from pydantic import ValidationError

from jimgw.samplers.config import (
    BaseSamplerConfig,
    BlackJAXNSAWConfig,
    BlackJAXNSSConfig,
    BlackJAXSMCConfig,
    BlackJAXSwiGConfig,
    FlowMCConfig,
    GRWConfig,
    HMCConfig,
    MALAConfig,
    ParallelTemperingConfig,
    SamplerConfig,
)


def test_discriminated_union_dispatch_flowmc():
    cfg = FlowMCConfig.model_validate({"type": "flowmc", "n_chains": 500})
    assert isinstance(cfg, FlowMCConfig)
    assert cfg.n_chains == 500


def test_discriminated_union_dispatch_ns_aw():
    cfg = BlackJAXNSAWConfig.model_validate({"type": "blackjax-ns-aw", "n_live": 2000})
    assert isinstance(cfg, BlackJAXNSAWConfig)
    assert cfg.n_live == 2000


def test_sampler_config_union_from_dict():
    from pydantic import TypeAdapter

    ta = TypeAdapter(SamplerConfig)
    cfg = ta.validate_python({"type": "flowmc"})
    assert isinstance(cfg, FlowMCConfig)

    cfg2 = ta.validate_python({"type": "blackjax-ns-aw"})
    assert isinstance(cfg2, BlackJAXNSAWConfig)

    cfg3 = ta.validate_python({"type": "blackjax-nss"})
    assert isinstance(cfg3, BlackJAXNSSConfig)

    cfg4 = ta.validate_python({"type": "blackjax-smc"})
    assert isinstance(cfg4, BlackJAXSMCConfig)

    cfg5 = ta.validate_python({"type": "blackjax-swig", "blocks": [["x"], ["y"]]})
    assert isinstance(cfg5, BlackJAXSwiGConfig)


def test_swig_blocks_must_be_nonempty_and_unique():
    with pytest.raises(ValidationError, match="at least one"):
        BlackJAXSwiGConfig(blocks=[])
    with pytest.raises(ValidationError, match="empty"):
        BlackJAXSwiGConfig(blocks=[["x"], []])
    with pytest.raises(ValidationError, match="multiple blocks"):
        BlackJAXSwiGConfig(blocks=[["x"], ["x"]])


def test_swig_sampling_defaults():
    config = BlackJAXSwiGConfig(blocks=[["x"]])
    assert config.num_gibbs_sweeps == 2
    assert config.termination_dlogz == pytest.approx(0.1)
    assert config.n_devices == 1


def test_sampler_configs_do_not_advertise_cache_capabilities():
    assert not hasattr(FlowMCConfig(), "cache_blocks")
    assert not hasattr(BlackJAXSwiGConfig(blocks=[["x"]]), "cache_blocks")


@pytest.mark.parametrize(
    ("config_cls", "config_kwargs"),
    [
        (BlackJAXNSAWConfig, {}),
        (BlackJAXNSSConfig, {}),
        (BlackJAXSwiGConfig, {"blocks": [["x"]]}),
    ],
)
@pytest.mark.parametrize("termination_dlogz", [0.0, -1.0])
def test_nested_sampling_termination_dlogz_must_be_positive(
    config_cls, config_kwargs, termination_dlogz
):
    with pytest.raises(ValidationError, match="greater than 0"):
        config_cls(**config_kwargs, termination_dlogz=termination_dlogz)


@pytest.mark.parametrize("config_cls", [BlackJAXNSSConfig, BlackJAXSwiGConfig])
def test_nested_sampler_sharding_requires_divisible_particle_counts(config_cls):
    kwargs = {"blocks": [["x"]]} if config_cls is BlackJAXSwiGConfig else {}
    with pytest.raises(ValidationError, match="n_live must be divisible by n_devices"):
        config_cls(n_live=10, n_delete_frac=0.4, n_devices=4, **kwargs)
    with pytest.raises(
        ValidationError, match="n_delete must be divisible by n_devices"
    ):
        config_cls(n_live=12, n_delete_frac=0.25, n_devices=2, **kwargs)


@pytest.mark.parametrize("config_cls", [BlackJAXNSSConfig, BlackJAXSwiGConfig])
def test_nested_sampler_sharding_requires_positive_device_count(config_cls):
    kwargs = {"blocks": [["x"]]} if config_cls is BlackJAXSwiGConfig else {}
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        config_cls(n_devices=0, **kwargs)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        FlowMCConfig(unknown_field=42)


def test_n_delete_frac_validator():
    with pytest.raises(ValidationError):
        BlackJAXNSAWConfig(n_delete_frac=0.0)
    with pytest.raises(ValidationError):
        BlackJAXNSAWConfig(n_delete_frac=1.0)
    cfg = BlackJAXNSAWConfig(n_delete_frac=0.5)
    assert cfg.n_delete_frac == 0.5


def test_base_config_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        BaseSamplerConfig[str](type="test", unknown_field=True)


def test_base_config_requires_sampler_type():
    with pytest.raises(ValidationError, match="type"):
        BaseSamplerConfig[str]()


# ---------------------------------------------------------------------------
# B1: FlowMC kernel/PT warning validator
# ---------------------------------------------------------------------------


def test_flowmc_pt_off_by_default():
    cfg = FlowMCConfig()
    assert cfg.parallel_tempering is None


def test_flowmc_pt_on_with_config():
    cfg = FlowMCConfig(parallel_tempering=ParallelTemperingConfig(n_temperatures=3))
    assert cfg.parallel_tempering is not None
    assert cfg.parallel_tempering.n_temperatures == 3


def test_flowmc_pt_on_with_true():
    cfg = FlowMCConfig(parallel_tempering=True)
    assert cfg.parallel_tempering is not None
    assert cfg.parallel_tempering.n_temperatures == 5  # default


def test_flowmc_pt_on_with_dict():
    cfg = FlowMCConfig(parallel_tempering={"n_temperatures": 8})
    assert cfg.parallel_tempering is not None
    assert cfg.parallel_tempering.n_temperatures == 8


def test_flowmc_pt_off_with_false():
    cfg = FlowMCConfig(parallel_tempering=False)
    assert cfg.parallel_tempering is None


def test_flowmc_pt_off_with_none():
    cfg = FlowMCConfig(parallel_tempering=None)
    assert cfg.parallel_tempering is None


def test_flowmc_irrelevant_kernel_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FlowMCConfig(local_kernel="MALA", hmc=HMCConfig(step_size=0.5))
    assert any("hmc" in str(warning.message).lower() for warning in w)


def test_flowmc_irrelevant_parallel_tempering_warns():
    # No warning expected: passing PT config enables it, nothing is ignored.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FlowMCConfig(parallel_tempering=ParallelTemperingConfig(n_temperatures=10))
    pt_warnings = [x for x in w if "parallel_tempering" in str(x.message).lower()]
    assert len(pt_warnings) == 0


def test_flowmc_no_spurious_warning_when_kernel_matches():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FlowMCConfig(local_kernel="HMC", hmc=HMCConfig(step_size=0.5))
    kernel_warnings = [x for x in w if "hmc" in str(x.message).lower()]
    assert len(kernel_warnings) == 0


# ---------------------------------------------------------------------------
# B2: BlackJAXSMCConfig temperature ladder validator
# ---------------------------------------------------------------------------


def test_smc_config_defaults():
    cfg = BlackJAXSMCConfig()
    assert cfg.persistent_sampling is True
    assert cfg.temperature_ladder is None


def test_smc_temperature_ladder_valid():
    cfg = BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 1.0])
    assert cfg.temperature_ladder == [0.0, 0.5, 1.0]


def test_smc_temperature_ladder_must_start_at_zero():
    with pytest.raises(ValidationError, match=r"start at 0\.0"):
        BlackJAXSMCConfig(temperature_ladder=[0.1, 0.5, 1.0])


def test_smc_temperature_ladder_must_end_at_one():
    with pytest.raises(ValidationError, match=r"end at 1\.0"):
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 0.9])


def test_smc_temperature_ladder_must_be_increasing():
    with pytest.raises(ValidationError, match="increasing"):
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.8, 0.5, 1.0])


def test_smc_temperature_ladder_warns_ess():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 1.0], target_ess=5000)
    assert any("ESS" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# B2b: BlackJAXSMCConfig ESS XOR validator
# ---------------------------------------------------------------------------


def test_smc_default_ess_fraction():
    cfg = BlackJAXSMCConfig()
    assert cfg.target_ess_fraction == 0.9
    assert cfg.target_ess is None


def test_smc_ess_fraction_set():
    cfg = BlackJAXSMCConfig(target_ess_fraction=0.3)
    assert cfg.target_ess_fraction == 0.3
    assert cfg.target_ess is None


def test_smc_absolute_ess_set():
    cfg = BlackJAXSMCConfig(target_ess=1000)
    assert cfg.target_ess == 1000
    assert cfg.target_ess_fraction is None


def test_smc_both_ess_raises():
    with pytest.raises(ValidationError, match="exactly one"):
        BlackJAXSMCConfig(target_ess_fraction=0.9, target_ess=1000)


def test_smc_fraction_zero_raises():
    with pytest.raises(ValidationError):
        BlackJAXSMCConfig(target_ess_fraction=0.0)


def test_smc_fraction_above_one_in_persistent_ok():
    cfg = BlackJAXSMCConfig(target_ess_fraction=1.5, persistent_sampling=True)
    assert cfg.target_ess_fraction == 1.5


def test_smc_fraction_above_one_in_tempered_raises():
    with pytest.raises(ValidationError, match=r"1\.0"):
        BlackJAXSMCConfig(target_ess_fraction=1.5, persistent_sampling=False)


def test_smc_absolute_ess_above_n_particles_in_tempered_raises():
    with pytest.raises(ValidationError):
        BlackJAXSMCConfig(target_ess=5000, n_particles=2000, persistent_sampling=False)


def test_smc_absolute_ess_above_n_particles_in_persistent_ok():
    cfg = BlackJAXSMCConfig(target_ess=5000, n_particles=2000, persistent_sampling=True)
    assert cfg.target_ess == 5000


def test_smc_fraction_warns_with_fixed_ladder():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 1.0], target_ess_fraction=0.3)
    assert any(
        "target_ess_fraction" in str(x.message) or "ESS" in str(x.message) for x in w
    )


# ---------------------------------------------------------------------------
# C: New kernel sub-config features (array step sizes, condition_matrix)
# ---------------------------------------------------------------------------


def test_mala_step_size_scalar():
    cfg = MALAConfig(step_size=1e-2)
    assert cfg.step_size == 1e-2


def test_mala_step_size_array():
    arr = np.array([1e-2, 2e-2, 3e-2])
    cfg = MALAConfig(step_size=arr)
    np.testing.assert_array_equal(cfg.step_size, arr)


def test_grw_step_size_array():
    arr = np.array([5e-3, 1e-2])
    cfg = GRWConfig(step_size=arr)
    np.testing.assert_array_equal(cfg.step_size, arr)


def test_hmc_condition_matrix_scalar():
    cfg = HMCConfig(condition_matrix=2.0)
    assert cfg.condition_matrix == 2.0


def test_hmc_condition_matrix_array():
    arr = np.array([1.0, 2.0, 0.5])
    cfg = HMCConfig(condition_matrix=arr)
    np.testing.assert_array_equal(cfg.condition_matrix, arr)


def test_hmc_defaults():
    cfg = HMCConfig()
    assert cfg.step_size == 2e-3
    assert cfg.condition_matrix == 1.0
    assert cfg.n_leapfrog_steps == 10


# ---------------------------------------------------------------------------
# D: Config classes no longer have a periodic field
# ---------------------------------------------------------------------------


def test_flowmc_config_has_no_periodic_field():
    assert not hasattr(FlowMCConfig(), "periodic")


def test_blackjax_ns_aw_config_has_no_periodic_field():
    assert not hasattr(BlackJAXNSAWConfig(), "periodic")


def test_blackjax_nss_config_has_no_periodic_field():
    assert not hasattr(BlackJAXNSSConfig(), "periodic")


def test_blackjax_smc_config_has_no_periodic_field():
    assert not hasattr(BlackJAXSMCConfig(), "periodic")


def test_flowmc_config_rejects_periodic_field():
    with pytest.raises(ValidationError):
        FlowMCConfig(periodic={"phase_c": (0.0, 6.28)})


def test_smc_resolve_target_ess_fraction():
    cfg = BlackJAXSMCConfig(target_ess_fraction=0.4)
    assert cfg._resolve_target_ess_fraction() == pytest.approx(0.4)

    cfg2 = BlackJAXSMCConfig(target_ess=1000, n_particles=2000)
    assert cfg2._resolve_target_ess_fraction() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# E: checkpoint_dir / checkpoint_interval validators
# ---------------------------------------------------------------------------


def test_checkpoint_dir_accepts_string(tmp_path):
    from pathlib import Path

    cfg = BlackJAXNSAWConfig(checkpoint_dir=str(tmp_path))
    assert cfg.checkpoint_dir == tmp_path
    assert isinstance(cfg.checkpoint_dir, Path)


def test_checkpoint_interval_negative_raises():
    with pytest.raises(ValidationError):
        FlowMCConfig(checkpoint_interval=-1.0)


def test_checkpoint_interval_without_dir_raises():
    with pytest.raises(ValidationError, match="checkpoint_dir must be set"):
        BlackJAXNSAWConfig(checkpoint_interval=600.0)


def test_checkpoint_interval_with_dir_ok(tmp_path):
    cfg = BlackJAXNSAWConfig(checkpoint_dir=tmp_path, checkpoint_interval=600.0)
    assert cfg.checkpoint_dir == tmp_path
    assert cfg.checkpoint_interval == 600.0


# ---------------------------------------------------------------------------
# F: configure_jax_cache
# ---------------------------------------------------------------------------


def test_configure_jax_cache_sets_dir(tmp_path):
    original = getattr(jax.config, "jax_compilation_cache_dir", None)
    try:
        BlackJAXNSAWConfig(
            checkpoint_dir=tmp_path, checkpoint_interval=60.0
        ).configure_jax_cache()
        assert (tmp_path / "jax_cache").is_dir()
        assert getattr(jax.config, "jax_compilation_cache_dir", None) == str(
            tmp_path / "jax_cache"
        )
    finally:
        jax.config.update("jax_compilation_cache_dir", original)


def test_configure_jax_cache_noop_when_no_dir():
    original = getattr(jax.config, "jax_compilation_cache_dir", None)
    try:
        BlackJAXNSAWConfig().configure_jax_cache()
        assert getattr(jax.config, "jax_compilation_cache_dir", None) == original
    finally:
        jax.config.update("jax_compilation_cache_dir", original)
