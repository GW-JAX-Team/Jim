"""Script to generate test fixtures from GWOSC data.

This script fetches real GW150914 data and saves it as fixtures for testing.
Run this once to create the fixture files.
"""

from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import get_H1, get_L1

# GW150914 parameters
gps = 1126259462.4
start = gps - 2  # 4s analysis segment
end = gps + 2
psd_start = gps - 2048  # 4096s for PSD estimation
psd_end = gps + 2048

print("Fetching GW150914 data from GWOSC...")

# Generate fixtures for H1 and L1
for get_ifo, name in [(get_H1, "H1"), (get_L1, "L1")]:
    print(f"\nProcessing {name}...")

    # Fetch 4s analysis segment
    print(f"  Fetching 4s strain data...")
    strain_data = Data.from_gwosc(name, start, end)
    strain_data.to_file(f"GW150914_strain_{name}.npz")
    print(f"  Saved GW150914_strain_{name}.npz")

    # Fetch 4096s for PSD estimation
    print(f"  Fetching 4096s data for PSD estimation...")
    psd_data = Data.from_gwosc(name, psd_start, psd_end)

    # Compute PSD using the 4s data's duration and sampling frequency
    psd_fftlength = strain_data.duration * strain_data.sampling_frequency
    psd = psd_data.to_psd(nperseg=psd_fftlength)
    psd.to_file(f"GW150914_psd_{name}.npz")
    print(f"  Saved GW150914_psd_{name}.npz")

print("\nFixture generation complete!")
print("Files created:")
print("  - GW150914_strain_H1.npz")
print("  - GW150914_strain_L1.npz")
print("  - GW150914_psd_H1.npz")
print("  - GW150914_psd_L1.npz")
