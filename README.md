# Jim 🚬

### A JAX-based gravitational-wave inference toolkit

[![docs](https://img.shields.io/badge/docs-online-blue)](https://gw-jax-team.github.io/jim/) [![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/GW-JAX-Team/jim/blob/main/LICENSE) [![coverage](https://img.shields.io/coveralls/github/GW-JAX-Team/jim/main)](https://coveralls.io/github/GW-JAX-Team/jim?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/jim/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/jim/main)

Jim is a JAX-based toolkit for Bayesian parameter estimation of gravitational-wave sources. It pairs differentiable waveform models from [ripple](https://github.com/GW-JAX-Team/ripple) with GPU-accelerated JAX-based samplers, enabling massively parallel inference.

**Supported samplers:**

- [flowMC](https://github.com/GW-JAX-Team/flowMC) — normalizing-flow-enhanced MCMC with optional parallel tempering.
- [BlackJAX NS-AW](https://github.com/mrosep/blackjax_ns_gw) — nested sampling described in [Prathaban et al. 2025 (arXiv:2509.04336)](https://arxiv.org/abs/2509.04336).
- [BlackJAX NSS](https://github.com/handley-lab/blackjax) — nested slice sampling.
- [BlackJAX SMC](https://github.com/blackjax-devs/blackjax) — sequential Monte Carlo with optional adaptive tempering and persistent sampling.

For a quick introduction, see the [Quick Start guide](https://gw-jax-team.github.io/jim/stable/quickstart/).

> [!WARNING]
> Jim has not yet reached v1.0.0 and the API may change. Use at your own risk. Consider pinning to a specific version if you need API stability.

## Installation

The simplest way to install Jim is through pip:

```bash
pip install JimGW
```

This will install the latest stable release and its dependencies.
Jim is built on [JAX](https://github.com/jax-ml/jax).
By default, this installs the CPU version of JAX.
If you have an NVIDIA GPU, install the CUDA-enabled version:

```bash
pip install JimGW[cuda]
```

If you want to install the latest version of Jim, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/jim.git
cd jim
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.

## Attribution

If you use Jim in your research, please cite the accompanying paper:

```bibtex
@article{Wong:2023lgb,
    author = "Wong, Kaze W. K. and Isi, Maximiliano and Edwards, Thomas D. P.",
    title = "{Fast Gravitational-wave Parameter Estimation without Compromises}",
    eprint = "2302.05333",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.3847/1538-4357/acf5cd",
    journal = "Astrophys. J.",
    volume = "958",
    number = "2",
    pages = "129",
    year = "2023"
}
```
