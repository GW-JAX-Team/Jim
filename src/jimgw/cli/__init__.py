import logging
import tomllib
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from jimgw.cli._config import PipelineConfig

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="jim-run",
    add_completion=False,
    help="Run a jimgw parameter-estimation pipeline from a TOML config file.",
)

_INIT_TEMPLATE = """\
seed = 0

[data]
type = "gwosc"
detectors = ["H1", "L1"]
trigger_time = 1126259462.4
duration = 4.0
post_trigger_duration = 2.0
psd_duration = 1024.0

[waveform]
approximant = "IMRPhenomXAS"
f_ref = 20.0

[prior]
M_c     = { type = "uniform",   min = 10.0,  max = 80.0  }
q       = { type = "uniform",   min = 0.125, max = 1.0   }
s1_z    = { type = "uniform",   min = -0.99, max = 0.99  }
s2_z    = { type = "uniform",   min = -0.99, max = 0.99  }
iota    = { type = "sine" }
d_L     = { type = "power_law", min = 1.0,   max = 2000.0, alpha = 2.0 }
t_c     = { type = "uniform",   min = -0.1,  max = 0.1   }
phase_c = { type = "uniform",   min = 0.0,   max = 6.283185307179586 }  # 2π
psi     = { type = "uniform",   min = 0.0,   max = 3.141592653589793 }  # π
ra      = { type = "uniform",   min = 0.0,   max = 6.283185307179586 }  # 2π
dec     = { type = "cosine" }

[likelihood]
f_min = 20.0
f_max = 1024.0

# Production defaults — for a quick test try: n_chains=100, n_global_steps=100, n_production_loops=2
[sampler]
type = "flowmc"
n_chains = 1000
n_local_steps = 100
n_global_steps = 1000
n_training_loops = 50
n_production_loops = 10
n_NFproposal_batch_size = 100
global_thinning = 100

[output]
dir = "output/my_run"
# save_corner requires the 'corner' package: pip install corner
save_corner = false
"""


@app.command()
def run(
    config: Optional[Path] = typer.Argument(None, help="Path to the TOML config file."),
    init: Optional[Path] = typer.Option(
        None,
        "--init",
        help="Write a minimal GW150914-style template config to PATH and exit.",
        metavar="PATH",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
) -> None:
    """Run a jimgw parameter-estimation pipeline from CONFIG."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(name)s | %(message)s")
    logging.getLogger("jimgw").setLevel(level)

    if init is not None:
        if init.exists():
            typer.echo(
                f"Error: {init} already exists. Choose a different path.", err=True
            )
            raise typer.Exit(code=2)
        try:
            init.parent.mkdir(parents=True, exist_ok=True)
            init.write_text(_INIT_TEMPLATE)
        except OSError as exc:
            typer.echo(f"Error: could not write template to {init}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(f"Template config written to {init}")
        raise typer.Exit()

    if config is None:
        typer.echo(
            "Error: provide a CONFIG file or use --init to create a template.", err=True
        )
        raise typer.Exit(code=2)

    if not config.exists():
        typer.echo(f"Error: config file not found: {config}", err=True)
        raise typer.Exit(code=2)

    try:
        with open(config, "rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        typer.echo(f"Error reading config {config}:\n{exc}", err=True)
        raise typer.Exit(code=2) from exc

    logger.info("Loaded config from %s", config)

    try:
        cfg = PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        typer.echo(f"Config validation error:\n{exc}", err=True)
        raise typer.Exit(code=2) from exc

    _log_config_summary(cfg)

    out_dir = cfg.output.dir
    if out_dir.exists() and not cfg.output.overwrite:
        typer.echo(
            f"Error: output directory already exists: {out_dir}. "
            "Set output.overwrite = true to allow overwriting.",
            err=True,
        )
        raise typer.Exit(code=2)

    import jax

    jax.config.update("jax_enable_x64", True)

    from jimgw.cli._data import build_data
    from jimgw.cli._jim import build_jim
    from jimgw.cli._likelihood import build_likelihood
    from jimgw.cli._output import write_outputs
    from jimgw.cli._prior import adapt_prior_for_ns_time, build_prior
    from jimgw.cli._transforms import (
        infer_likelihood_transforms,
        infer_sample_transforms,
    )
    from jimgw.cli._waveform import build_waveform

    trigger_time: float = cfg.data.trigger_time

    # Stage 2: waveform
    waveform = build_waveform(cfg.waveform)

    # Stage 3: data — injection runs receive the already-built waveform
    ifos = build_data(
        cfg.data,
        f_min=cfg.likelihood.f_min,
        f_max=cfg.likelihood.f_max,
        waveform=waveform,
        time_frame=cfg.sampling.time_frame,
    )

    # NS-AW requires all sampling-space parameters in [0, 1].
    # Must run before build_prior so the built prior and
    # prior_params already reflect the substitution.
    if cfg.sampler.type == "blackjax-ns-aw":
        modified_prior = adapt_prior_for_ns_time(cfg.prior, cfg.sampling)
        if modified_prior is not None:
            cfg.prior = modified_prior

    # Stage 4: prior
    prior = build_prior(cfg.prior)

    # Stage 5: transform inference
    prior_params = frozenset(prior.parameter_names)
    ns_aw = cfg.sampler.type == "blackjax-ns-aw"
    sample_transforms = infer_sample_transforms(
        prior_params,
        trigger_time,
        ifos,
        cfg.sampling,
        unit_cube=ns_aw,
        prior_cfg=cfg.prior,
    )
    likelihood_transforms = infer_likelihood_transforms(
        prior_params,
        trigger_time,
        ifos,
        cfg.sampling,
        cfg.waveform.f_ref,
        phase_marginalization=cfg.likelihood.phase_marginalization,
    )

    # Stage 6: likelihood
    likelihood = build_likelihood(
        cfg.likelihood,
        ifos,
        waveform,
        trigger_time,
        cfg.waveform.f_ref,
        prior=prior,
        likelihood_transforms=likelihood_transforms,
        data_cfg=cfg.data,
        time_frame=cfg.sampling.time_frame,
    )

    # Stage 7: build Jim + run sampler
    jim = build_jim(
        likelihood,
        prior,
        sample_transforms,
        likelihood_transforms,
        cfg,
        verbose=verbose,
    )

    try:
        jim.sample()
    except Exception as exc:
        logger.error("Sampling failed: %s", exc)
        raise typer.Exit(code=3) from exc
    logger.info("Sampling complete.")

    # Stage 8: write outputs
    try:
        write_outputs(jim, cfg)
    except FileExistsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _log_config_summary(cfg: PipelineConfig) -> None:
    logger.info("seed: %d", cfg.seed)
    logger.info(
        "data: type=%s, detectors=%s",
        cfg.data.type,
        cfg.data.detectors,
    )
    logger.info(
        "waveform: %s (f_ref=%.1f Hz)", cfg.waveform.approximant, cfg.waveform.f_ref
    )
    param_names = list(cfg.prior.root.keys())
    logger.info("prior: %d parameter(s): %s", len(param_names), param_names)
    logger.info("sampler: type=%s", cfg.sampler.type)
    logger.info("output: %s", cfg.output.dir)
