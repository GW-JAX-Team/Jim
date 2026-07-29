"""Unit tests for waveform-cache block-dependency inference."""

from types import SimpleNamespace

import pytest

from jimgw.core.single_event.blocked_likelihood import (
    _build_rebuild_required_by_block,
    _infer_waveform_sampling_dependencies,
    _validate_parameter_blocks,
)
from jimgw.core.transforms import (
    BijectiveTransform,
    ConditionalBijectiveTransform,
    NtoMTransform,
)


def _fake_likelihood(
    waveform_parameter_names: tuple[str, ...],
    fixed_parameters: dict | None = None,
    waveform_caches_distance: bool = False,
) -> SimpleNamespace:
    """Minimal duck-typed stand-in exposing only what the module reads."""
    return SimpleNamespace(
        waveform=SimpleNamespace(parameter_names=waveform_parameter_names),
        fixed_parameters=fixed_parameters or {},
        waveform_caches_distance=waveform_caches_distance,
    )


class TestValidateParameterBlocks:
    def test_exact_partition_is_accepted(self):
        _validate_parameter_blocks([["a"], ["b", "c"]], parameter_names=("a", "b", "c"))

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="not sampling parameters"):
            _validate_parameter_blocks([["a"], ["z"]], parameter_names=("a", "b"))

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="do not cover"):
            _validate_parameter_blocks([["a"]], parameter_names=("a", "b"))


class TestInferWaveformSamplingDependencies:
    def test_identity_with_no_transforms(self):
        likelihood = _fake_likelihood(waveform_parameter_names=("a", "c"))
        deps = _infer_waveform_sampling_dependencies(
            likelihood,
            parameter_names=("a", "b"),
            sample_transforms=[],
            likelihood_transforms=[],
        )
        assert deps == {"a"}

    def test_sample_transform_propagates_back_to_sampling_name(self):
        # Sampling space has "a_scaled" (prior space has "a"); the waveform
        # consumes prior-space "a", so the dependency must resolve to the
        # *sampling*-space name "a_scaled", not "a".
        likelihood = _fake_likelihood(waveform_parameter_names=("a", "b"))
        sample_transform = BijectiveTransform(name_mapping=(["a"], ["a_scaled"]))
        deps = _infer_waveform_sampling_dependencies(
            likelihood,
            parameter_names=("a_scaled", "b"),
            sample_transforms=[sample_transform],
            likelihood_transforms=[],
        )
        assert deps == {"a_scaled", "b"}

    def test_likelihood_transform_propagates_forward(self):
        # "q" (sampling==prior space) maps to "eta" (likelihood space) via a
        # likelihood transform; the waveform consumes "eta" but the inferred
        # dependency must be reported in terms of the sampling name "q".
        likelihood = _fake_likelihood(waveform_parameter_names=("eta",))
        likelihood_transform = NtoMTransform(name_mapping=(["q"], ["eta"]))
        deps = _infer_waveform_sampling_dependencies(
            likelihood,
            parameter_names=("q",),
            sample_transforms=[],
            likelihood_transforms=[likelihood_transform],
        )
        assert deps == {"q"}

    def test_conditional_names_contribute_as_inputs(self):
        # A ConditionalBijectiveTransform's conditional_names are additional
        # inputs (not consumed), so they must feed into the produced name's
        # dependency set even though they aren't in name_mapping[0].
        likelihood = _fake_likelihood(waveform_parameter_names=("x",))
        conditional_transform = ConditionalBijectiveTransform(
            name_mapping=(["x"], ["y"]),
            conditional_names=["z"],
        )
        deps = _infer_waveform_sampling_dependencies(
            likelihood,
            parameter_names=("y", "z"),
            sample_transforms=[conditional_transform],
            likelihood_transforms=[],
        )
        assert deps == {"y", "z"}

    def test_d_l_is_skipped_when_waveform_caches_distance(self):
        # d_L amplitude is factored out of the cache, so it must never be
        # reported as a waveform dependency even when it is itself a
        # sampling-space parameter with an identity dependency.
        likelihood = _fake_likelihood(
            waveform_parameter_names=("d_L", "a"), waveform_caches_distance=True
        )
        deps = _infer_waveform_sampling_dependencies(
            likelihood,
            parameter_names=("d_L", "a"),
            sample_transforms=[],
            likelihood_transforms=[],
        )
        assert deps == {"a"}

    def test_d_l_is_a_dependency_when_waveform_does_not_cache_distance(self):
        # Without analytic distance scaling, the cache is a full evaluation at
        # the given d_L, so d_L is a dependency like any other parameter.
        likelihood = _fake_likelihood(
            waveform_parameter_names=("d_L", "a"), waveform_caches_distance=False
        )
        deps = _infer_waveform_sampling_dependencies(
            likelihood,
            parameter_names=("d_L", "a"),
            sample_transforms=[],
            likelihood_transforms=[],
        )
        assert deps == {"d_L", "a"}

    def test_constant_fixed_parameter_contributes_nothing(self):
        likelihood = _fake_likelihood(
            waveform_parameter_names=("c", "a"),
            fixed_parameters={"c": 30.0},
        )
        deps = _infer_waveform_sampling_dependencies(
            likelihood,
            parameter_names=("a",),
            sample_transforms=[],
            likelihood_transforms=[],
        )
        assert deps == {"a"}

    def test_callable_fixed_parameter_depends_on_everything(self):
        likelihood = _fake_likelihood(
            waveform_parameter_names=("c",),
            fixed_parameters={"c": lambda params: params},
        )
        deps = _infer_waveform_sampling_dependencies(
            likelihood,
            parameter_names=("a", "b"),
            sample_transforms=[],
            likelihood_transforms=[],
        )
        assert deps == {"a", "b"}

    def test_unresolvable_name_is_skipped(self):
        # A waveform input that traces back to neither a sampling parameter
        # nor a fixed parameter (e.g. event metadata like trigger_time/gmst,
        # injected later by _prepare_parameters) contributes no dependency.
        likelihood = _fake_likelihood(waveform_parameter_names=("trigger_time", "a"))
        deps = _infer_waveform_sampling_dependencies(
            likelihood,
            parameter_names=("a",),
            sample_transforms=[],
            likelihood_transforms=[],
        )
        assert deps == {"a"}


class TestBuildRebuildRequiredByBlock:
    def test_marks_blocks_by_waveform_dependency(self):
        likelihood = _fake_likelihood(waveform_parameter_names=("a",))
        result = _build_rebuild_required_by_block(
            likelihood,
            blocks=[["a"], ["b"]],
            parameter_names=("a", "b"),
            sample_transforms=[],
            likelihood_transforms=[],
        )
        assert result == {(0,): True, (1,): False}

    def test_mixed_block_requires_rebuild(self):
        likelihood = _fake_likelihood(waveform_parameter_names=("a",))
        result = _build_rebuild_required_by_block(
            likelihood,
            blocks=[["a", "b"]],
            parameter_names=("a", "b"),
            sample_transforms=[],
            likelihood_transforms=[],
        )
        assert result == {(0, 1): True}
