# Test Fixtures

This directory contains cached data fixtures for unit tests. These fixtures are pre-generated from GWOSC data to speed up tests and avoid repeated network calls.

## Fixture Files

- `GW150914_strain_H1.npz` - 4s strain data for H1 detector around GW150914
- `GW150914_strain_L1.npz` - 4s strain data for L1 detector around GW150914
- `GW150914_psd_H1.npz` - Pre-computed PSD for H1 (from 4096s data)
- `GW150914_psd_L1.npz` - Pre-computed PSD for L1 (from 4096s data)

## Regenerating Fixtures

If you need to regenerate the fixtures (e.g., after changing data format), run:

```bash
cd tests/fixtures
uv run python generate_fixtures.py
```

This will fetch fresh data from GWOSC and save it using the `Data.to_file()` and `PowerSpectrum.to_file()` methods.

**Note:** Fixture generation requires network access to GWOSC.

## File Format

All fixtures are stored in NumPy's `.npz` format using `jnp.savez()`. They contain:

**Strain files:**

- `td`: Time-domain data array
- `dt`: Time step (sampling interval)
- `start_time`: GPS time of first sample
- `name`: Detector name (e.g., "H1")

**PSD files:**

- `values`: PSD values array
- `frequencies`: Frequency array
- `name`: Detector name (e.g., "H1")
