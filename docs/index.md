# Jim 🚬

## A JAX-based gravitational-wave inference toolkit

Jim is a JAX-based toolkit for Bayesian parameter estimation of gravitational-wave sources. It pairs differentiable waveform models from [ripple](https://github.com/GW-JAX-Team/ripple) with GPU-accelerated JAX-based samplers, enabling massively parallel inference.

**Supported samplers:**

- [flowMC](https://github.com/GW-JAX-Team/flowMC) — normalizing-flow-enhanced MCMC with optional parallel tempering.
- [BlackJAX NS-AW](https://github.com/mrosep/blackjax_ns_gw) — nested sampling described in [Prathaban et al. 2025 (arXiv:2509.04336)](https://arxiv.org/abs/2509.04336).
- [BlackJAX NSS](https://github.com/handley-lab/blackjax) — nested slice sampling.
- [BlackJAX SMC](https://github.com/blackjax-devs/blackjax) — sequential Monte Carlo with optional adaptive tempering and persistent sampling.

!!! warning
    Jim has not yet reached v1.0.0 and the API may change. Use at your own risk. Consider pinning to a specific version if you need API stability.

## Documentation

- **[Installation](installation.md)** — How to install Jim
- **[Quick Start](quickstart.md)** — A basic example to get started
- **[Tutorials](tutorials/index.md)** — Hands-on worked examples
- **[Guides](guides/index.md)** — In-depth coverage of each module
- **[FAQ](FAQ.md)** — Answers to common questions
