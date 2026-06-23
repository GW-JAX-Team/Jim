# Example configs

These TOML files are ready-to-run configs for `jim-run`. Each one targets a specific event or scenario and can be used as a starting point for your own analysis.

## Running a config

```bash
jim-run GW150914_flowmc.toml
```

Outputs are written to the directory specified by `output.dir` in the config. Run `jim-run --help` for available options.

For a detailed walkthrough of what each config section does, see the [GW150914 tutorial](https://gw-jax-team.github.io/Jim/stable/tutorials/gw150914_cli/) or the [CLI Config Reference](https://gw-jax-team.github.io/Jim/stable/guides/cli/).
