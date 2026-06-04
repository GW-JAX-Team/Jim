#!/usr/bin/env python
"""Cross-validate JAX GMST/UTC implementation against LAL and bilby.

This script verifies the JAX implementation of the conversion from GPS time
to UTC date and GMST values against the LAL and bilby implementations, all
the way up to year 2500 (and potentially beyond this).

The verification is done in two parts:
1. Without JIT, which gives an EXACT match with bilby values.
2. With JIT enabled, which gives an error of order 1e-10 s.

The GPS times are generated from 1980-01-06 to 2500-12-31.
Note that the LAL implementation, limited by overflow errors, can only compute
up to the year 2038. The remaining years till 2500 (and beyond) are way beyond
practical purposes, and are purely for algorithmic comparison with bilby.

Requires:
    - bilby
"""

import time
from calendar import timegm
import jax
import jax.numpy as jnp
import numpy as np

from tests.utils import check_bilby_available

# Check if bilby and LAL are available
try:
    check_bilby_available()
    from lal import GPSToUTC, GreenwichMeanSiderealTime as LAL_gmst
    from bilby_cython.time import (
        gps_time_to_utc,
        greenwich_mean_sidereal_time as bilby_gmst,
    )

    BILBY_AVAILABLE = True
except ImportError as e:
    print(f"Error: {e}")
    print("This script requires bilby.")
    exit(1)

jax.config.update("jax_platforms", "cpu")

from jimgw.core.single_event.time_utils import (
    gps_to_utc_date,
    greenwich_mean_sidereal_time as jim_gmst,
)

# Configuration
SIZE = 10_000_000  # Total number of samples
# SIZE = 10_000  # Uncomment for quick testing
CHUNKS = 8  # Number of chunks for processing


def compute_seconds_from_utc_date(hour: int, min: int, sec: int) -> int:
    """Convert UTC time components to total seconds."""
    return sec + min * 60 + hour * 3600


print("=" * 50)
print("Generating GPS times and computing LAL/bilby values")
print("=" * 50)

# The test range is the designed time range of the Jim UTC date implementation
start_gps = timegm(time.strptime("1980-01-06", "%Y-%m-%d"))
end_gps = timegm(time.strptime("2500-12-31", "%Y-%m-%d"))
print(f"GPS time range: {start_gps} to {end_gps}")
print(f"Generating {SIZE} samples...")

gps_times = np.geomspace(start_gps, end_gps, SIZE, dtype=np.int64)
none_tuple = tuple([0] * 9)

# Prepare data from LAL and bilby
start_time = time.time()
results = []
for i, gps_time in enumerate(gps_times):
    if i % 1_000_000 == 0 and i > 0:
        elapsed = time.time() - start_time
        eta = elapsed / i * (SIZE - i)
        print(f"  Processed {i}/{SIZE} samples ({elapsed:.1f}s, ETA: {eta:.1f}s)")

    bilby_utc = gps_time_to_utc(gps_time)
    bilby_sec = compute_seconds_from_utc_date(
        bilby_utc.hour, bilby_utc.minute, bilby_utc.second
    )
    bilby_gmst_val = bilby_gmst(gps_time)

    try:
        # Note: LAL implementation has an upper limit of year 2038
        # https://lscsoft.docs.ligo.org/lalsuite/lal/_x_l_a_l_civil_time_8c_source.html#l00276
        lal_utc = GPSToUTC(gps_time)
        lal_sec = compute_seconds_from_utc_date(lal_utc[3], lal_utc[4], lal_utc[5])
        lal_gmst_val = LAL_gmst(gps_time)
    except (OverflowError, RuntimeError):
        lal_utc = none_tuple
        lal_sec = 0
        lal_gmst_val = 0.0

    results.append(
        (
            gps_time,
            (lal_utc[0], lal_utc[1], lal_utc[2], lal_sec, lal_gmst_val),
            (bilby_utc.year, bilby_utc.month, bilby_utc.day, bilby_sec, bilby_gmst_val),
        )
    )

print(f"Data generation completed in {time.time() - start_time:.2f} seconds")

# Convert to structured array
computed_times = np.array(
    results,
    dtype=[
        ("gps_time", np.int64),
        (
            "lal",
            [
                ("year", np.int32),
                ("month", np.int32),
                ("day", np.int32),
                ("sec", np.int32),
                ("gmst", np.float64),
            ],
        ),
        (
            "bilby",
            [
                ("year", np.int64),
                ("month", np.int64),
                ("day", np.int64),
                ("sec", np.int64),
                ("gmst", np.float64),
            ],
        ),
    ],
)

# Convert to JAX arrays for verification
computed_times = jnp.array(computed_times)

print("\n" + "=" * 50)
print("CROSS-VALIDATION: Jim vs LAL/bilby")
print("=" * 50)

with jax.disable_jit():
    print("\n" + "=" * 50)
    print("Part 1: WITHOUT JIT (Exact Match Expected)")
    print("=" * 50)
    start = 0
    for end in jnp.linspace(0, SIZE + 1, CHUNKS, dtype=jnp.int32)[1:]:
        print(f"\nChunk: indices {start} to {end}")
        _computed_times = computed_times[start:end]
        print(f"  Processing {_computed_times.size} samples...")

        start_time = time.time()
        gps_times_chunk = jnp.asarray(_computed_times["gps_time"])
        utc_dates = jax.vmap(gps_to_utc_date)(gps_times_chunk)
        gmst_vals = jax.vmap(jim_gmst)(gps_times_chunk)
        print(f"  Jim computation: {time.time() - start_time:.4f}s")

        print("\n  LAL vs bilby comparison:")
        for item in ("year", "month", "day", "sec", "gmst"):
            is_agree = jnp.where(
                _computed_times["lal"]["year"] != 0,
                (_computed_times["lal"][item] == _computed_times["bilby"][item]),
                True,
            ).all()
            print(f"    {item:6s}: {'✓ MATCH' if is_agree else '✗ DIFFER'}")

        print("\n  Jim vs bilby comparison (exact):")
        for key, jim_val in zip(
            ("year", "month", "day", "sec", "gmst"), (*utc_dates, gmst_vals)
        ):
            is_agree = (jim_val == _computed_times["bilby"][key]).all()
            print(f"    {key:6s}: {'✓ MATCH' if is_agree else '✗ DIFFER'}")
        start = end

print("\n" + "=" * 50)
print("Part 2: WITH JIT (Numerical Tolerance Expected)")
print("=" * 50)
RTOL = 1e-16
ATOL = 4e-10
print(f"Tolerance: rtol={RTOL}, atol={ATOL}")

start = 0
for end in jnp.linspace(0, SIZE + 1, CHUNKS, dtype=jnp.int32)[1:]:
    print(f"\nChunk: indices {start} to {end}")
    _computed_times = computed_times[start:end]
    print(f"  Processing {_computed_times.size} samples...")

    start_time = time.time()
    gps_times_chunk = jnp.asarray(_computed_times["gps_time"])
    utc_dates = jax.vmap(gps_to_utc_date)(gps_times_chunk)
    gmst_vals = jax.vmap(jim_gmst)(gps_times_chunk)
    print(f"  Jim computation: {time.time() - start_time:.4f}s")

    print("\n  Jim vs bilby comparison (with tolerance):")
    for key, jim_val in zip(
        ("year", "month", "day", "sec", "gmst"), (*utc_dates, gmst_vals)
    ):
        is_agree = jnp.allclose(
            jim_val, _computed_times["bilby"][key], rtol=RTOL, atol=ATOL
        )
        print(f"    {key:6s}: {'✓ MATCH' if is_agree else '✗ DIFFER'}")
    start = end

print("\n" + "=" * 50)
print("VERIFICATION COMPLETE")
print("=" * 50)
print("""
Notes:
- LAL implementation is limited to year 2038 due to overflow
- Years beyond 2038 are compared only with bilby (algorithmic test)
- Without JIT: Exact match expected for all fields
- With JIT: Numerical differences ~1e-10s are acceptable for GMST/sec
""")
