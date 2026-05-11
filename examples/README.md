# Examples

This directory contains two kinds of examples.

## CLI configs (`configs/`)

TOML config files for `jim-run` — the recommended way to run an analysis:

```bash
jim-run configs/GW150914_flowmc.toml
```

See [`configs/README.md`](configs/README.md) for details.

## Python scripts

Standalone scripts that use the programmatic API directly. Run with:

```bash
python GW150914_flowMC.py
```

These scripts require Jim and its dependencies to be installed. Each script fetches data from GWOSC on first run, so an internet connection is needed.
