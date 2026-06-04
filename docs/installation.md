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
git clone https://github.com/GW-JAX-Team/jim.git
cd jim
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.

## BlackJAX nested samplers

Jim's BlackJAX nested-sampling backends depend on modules not yet released on PyPI.
They are distributed via a maintained fork and must be installed separately.

### With uv (recommended)

```bash
uv sync --group nested-sampling
```

This installs the `blackjax` fork declared in `[tool.uv.sources]`. The fork is resolved automatically — no manual git clone needed.

### From source (pip)

```bash
pip install "git+https://github.com/GW-JAX-Team/blackjax.git@jim"
```

> **Note:** `pip install jimgw[nested-sampling]` will not work until the required features land in a PyPI release. Use one of the commands above instead.
