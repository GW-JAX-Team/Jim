import json
import logging
import shutil
from pathlib import Path

import numpy as np
import tomli_w

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
        _save_corner(out_dir, samples, cfg.output.corner_parameters)


def _save_corner(out_dir: Path, samples: dict, param_names: list[str] | None) -> None:
    try:
        import corner  # type: ignore[import]
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError:
        logger.warning("corner or matplotlib not available — skipping corner plot")
        return

    if param_names:
        labels = [p for p in param_names if p in samples]
        if labels:
            data = np.column_stack([np.asarray(samples[p]) for p in labels])
        else:
            labels = list(samples.keys())
            data = np.column_stack([np.asarray(samples[p]) for p in labels])
    else:
        labels = list(samples.keys())
        data = np.column_stack([np.asarray(samples[p]) for p in labels])

    fig = corner.corner(data, labels=labels)
    corner_path = out_dir / "corner.png"
    fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved corner plot to %s", corner_path)
