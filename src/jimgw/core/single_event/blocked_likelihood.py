"""Waveform-cache support for single-event likelihoods."""

from collections.abc import Sequence

from ripplegw.interfaces import DistanceScaledWaveform

from jimgw.core.single_event.likelihood import SingleEventLikelihood
from jimgw.core.transforms import (
    BijectiveTransform,
    ConditionalBijectiveTransform,
    NtoMTransform,
)


def _validate_parameter_blocks(
    blocks: Sequence[Sequence[str]],
    *,
    parameter_names: tuple[str, ...],
) -> None:
    """Validate that proposal blocks exactly cover valid sampling parameters."""
    flattened_names = [name for block in blocks for name in block]
    unknown_names = sorted(set(flattened_names) - set(parameter_names))
    missing_names = sorted(set(parameter_names) - set(flattened_names))
    if unknown_names:
        raise ValueError(
            "Block parameter(s) "
            f"{unknown_names} are not sampling parameters {parameter_names}."
        )
    if missing_names:
        raise ValueError(
            f"Parameter blocks do not cover sampling parameter(s) {missing_names}."
        )


def _build_rebuild_required_by_block(
    likelihood: SingleEventLikelihood,
    blocks: Sequence[Sequence[str]],
    *,
    parameter_names: tuple[str, ...],
    sample_transforms: Sequence[BijectiveTransform],
    likelihood_transforms: Sequence[NtoMTransform],
) -> dict[tuple[int, ...], bool]:
    """Map each proposal block to whether it requires rebuilding."""
    waveform_sampling_dependencies = _infer_waveform_sampling_dependencies(
        likelihood,
        parameter_names,
        sample_transforms,
        likelihood_transforms,
    )
    return {
        tuple(parameter_names.index(name) for name in block): bool(
            set(block).intersection(waveform_sampling_dependencies)
        )
        for block in blocks
    }


def _infer_waveform_sampling_dependencies(
    likelihood: SingleEventLikelihood,
    parameter_names: tuple[str, ...],
    sample_transforms: Sequence[BijectiveTransform],
    likelihood_transforms: Sequence[NtoMTransform],
) -> set[str]:
    """Map waveform inputs back to sampling-space parameters."""
    all_sampling_parameters = set(parameter_names)
    dependencies = {name: {name} for name in parameter_names}

    for transforms, reverse in (
        (reversed(sample_transforms), True),
        (iter(likelihood_transforms), False),
    ):
        for transform in transforms:
            from_names, to_names = transform.name_mapping
            consumed_names = to_names if reverse else from_names
            produced_names = from_names if reverse else to_names
            conditional_names = (
                transform.conditional_names
                if isinstance(transform, ConditionalBijectiveTransform)
                else ()
            )
            input_names = (*consumed_names, *conditional_names)
            input_dependencies: set[str] = set()
            for name in input_names:
                input_dependencies.update(
                    dependencies.get(name, all_sampling_parameters)
                )
            for name in consumed_names:
                dependencies.pop(name, None)
            for name in produced_names:
                dependencies[name] = input_dependencies.copy()

    waveform_sampling_dependencies: set[str] = set()
    waveform_caches_distance = isinstance(likelihood.waveform, DistanceScaledWaveform)
    for parameter_name in likelihood.waveform.parameter_names:
        if parameter_name == "d_L" and waveform_caches_distance:
            # Cached polarizations factor out inverse-distance amplitude.
            continue
        if parameter_name in likelihood.fixed_parameters:
            if callable(likelihood.fixed_parameters[parameter_name]):
                waveform_sampling_dependencies.update(all_sampling_parameters)
            continue
        sampling_dependencies = dependencies.get(parameter_name)
        if sampling_dependencies is not None:
            waveform_sampling_dependencies.update(sampling_dependencies)
    return waveform_sampling_dependencies
