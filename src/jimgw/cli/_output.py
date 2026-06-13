import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import tomli_w
import jax.numpy as jnp

from jimgw.cli._transforms import to_likelihood_space
from jimgw.core.single_event.detector import GroundBased2G
from jimgw.core.transforms import NtoMTransform

logger = logging.getLogger(__name__)


def write_outputs(jim, cfg) -> None:
    """Write samples, diagnostics, and optional corner plot under output.dir."""

    out_dir: Path = cfg.output.dir

    if out_dir.exists():
        if not cfg.output.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}. "
                "Set output.overwrite = true to allow overwriting."
            )
        shutil.rmtree(out_dir)
        logger.info("Removed existing output directory: %s", out_dir)
    out_dir.mkdir(parents=True)

    # Samples
    samples = jim.get_samples(n_samples=cfg.output.n_samples)
    samples_path = out_dir / "samples.npz"
    np.savez(samples_path, **{k: np.asarray(v) for k, v in samples.items()})  # type: ignore[call-overload]
    logger.info(
        "Saved %d samples to %s", next(iter(samples.values())).shape[0], samples_path
    )

    # Diagnostics
    diagnostics = jim.get_diagnostics()
    scalar_diag = {k: v for k, v in diagnostics.items() if np.asarray(v).ndim == 0}
    array_diag = {k: v for k, v in diagnostics.items() if np.asarray(v).ndim > 0}

    if scalar_diag:
        diag_json = out_dir / "diagnostics.json"
        with open(diag_json, "w") as f:
            json.dump({k: float(v) for k, v in scalar_diag.items()}, f, indent=2)
        logger.info("Saved scalar diagnostics to %s", diag_json)

    if array_diag:
        diag_npz = out_dir / "diagnostics.npz"
        np.savez(diag_npz, **{k: np.asarray(v) for k, v in array_diag.items()})  # type: ignore[call-overload]
        logger.info("Saved array diagnostics to %s", diag_npz)

    # Resolved config
    cfg_path = out_dir / "config.final.toml"
    dumped = cfg.model_dump(mode="json", exclude_none=True)
    if dumped.get("sampler", {}).get("type") == "flowmc":
        active = dumped["sampler"]["local_kernel"].lower()
        for kernel in ("mala", "hmc", "grw"):
            if kernel != active:
                dumped["sampler"].pop(kernel, None)
    with open(cfg_path, "wb") as f:
        tomli_w.dump(dumped, f)
    logger.info("Saved resolved config to %s", cfg_path)

    # Corner plot
    if cfg.output.save_corner:
        truths = (
            _injection_truths_in_prior_space(
                cfg.data.injection_parameters,
                jim.likelihood_transforms,
                cfg.waveform.f_ref,
                trigger_time=cfg.data.trigger_time,
                ifos=list(jim.likelihood.detectors),
                time_frame=cfg.sampling.time_frame,
            )
            if cfg.data.type == "injection"
            else None
        )
        _save_corner(out_dir, samples, cfg.output.corner_parameters, truths)


def _injection_truths_in_prior_space(
    injection_parameters: dict[str, float],
    likelihood_transforms: list[NtoMTransform],
    waveform_f_ref: float,
    trigger_time: float,
    ifos: list[GroundBased2G],
    time_frame: str,
) -> Optional[dict[str, float]]:
    """Convert injection parameters to prior space for corner plot truth markers.

    injection_parameters may be in any supported parametrization (J-frame spins,
    spherical spins, q/eta, azimuth/zenith, t_det, etc.).  We convert to
    likelihood space first, then reverse the likelihood transforms to land in
    prior space — the same space that jim.get_samples() returns.
    """
    p: dict = to_likelihood_space(
        injection_parameters,
        waveform_f_ref,
        trigger_time=trigger_time,
        ifos=ifos,
        time_frame=time_frame,
    )
    p = {k: jnp.float64(v) for k, v in p.items()}

    for transform in reversed(likelihood_transforms):
        # All currently supported likelihood transforms have a backward method
        # but this may not always be the case.
        if hasattr(transform, "backward"):
            p = transform.backward(p)  # type: ignore[attr-defined]
        else:
            logger.warning(
                "Likelihood transform %s does not have a backward method — "
                "cannot convert truths to prior space",
                transform,
            )
            return None

    return {k: float(v) for k, v in p.items()}


def _save_corner(
    out_dir: Path,
    samples: dict,
    param_names: Optional[list[str]] = None,
    truths: Optional[dict[str, float]] = None,
) -> None:
    try:
        import corner  # type: ignore[import]
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError:
        logger.warning("corner or matplotlib not available — skipping corner plot")
        return

    labels = list(samples.keys())
    if param_names:
        filtered = [p for p in param_names if p in samples]
        if filtered:
            labels = filtered
    data = np.column_stack([np.asarray(samples[p]) for p in labels])

    # Limit number of samples for corner plot to avoid excessive memory usage and slow plotting.
    n_corner = 5000
    if data.shape[0] > n_corner:
        data = data[:n_corner]

    truth_values = [truths.get(p) for p in labels] if truths else None

    fig = corner.corner(data, labels=labels, truths=truth_values)
    corner_path = out_dir / "corner.png"
    fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved corner plot to %s", corner_path)
