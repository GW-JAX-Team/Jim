# Installation

The simplest way to install Jim is through pip:

```bash
pip install JimGW
```

This will install the latest stable release and its dependencies.
Jim is built on [JAX](https://github.com/google/jax).
By default, this installs the CPU version of JAX.
If you have an NVIDIA GPU, install the CUDA-enabled version:

```bash
pip install "JimGW[cuda]"
```

If you want to install the latest version of Jim, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/Jim.git
cd Jim
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.

## BlackJAX nested samplers

Jim's BlackJAX nested-sampling backends depend on modules not yet released on PyPI.
The features are merged into the upstream [blackjax-devs/blackjax](https://github.com/blackjax-devs/blackjax) main branch but await a versioned release.

### With uv (recommended)

Clone the repository if you haven't already, then sync the nested-sampling group:

```bash
git clone https://github.com/GW-JAX-Team/Jim.git
cd Jim
uv sync --group nested-sampling
```

This installs BlackJAX from `blackjax-devs/blackjax` as declared in `[tool.uv.sources]`.

### From source (pip)

```bash
pip install "git+https://github.com/blackjax-devs/blackjax.git"
```

> **Note:** `pip install jimgw[nested-sampling]` will not work until the required features land in a PyPI release. Use one of the commands above instead.
