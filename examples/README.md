# Examples

This directory contains two kinds of examples.

## CLI configs (`cli/`)

TOML config files for `jim-run` — the recommended way to run an analysis:

```bash
jim-run cli/GW150914_flowmc.toml
```

See [`cli/README.md`](cli/README.md) for details.

## Python scripts

Standalone scripts that use the programmatic API directly. Run with:

```bash
python GW150914_flowMC.py
```

These scripts require Jim and its dependencies to be installed. Each script fetches data from GWOSC on first run, so an internet connection is needed.
