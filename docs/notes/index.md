# Technical Notes

These notes give the details behind Jim's results: conventions and implementation choices that matter when you compare Jim with another code, but that the guides and how-tos do not need.

- **[Detector Response Conventions](detector_response.md)** — The phase convention of the finite-arm-length response and how to convert from the other common one.
- **[Detector Presets](detector_presets.md)** — How the Einstein Telescope and Cosmic Explorer geometries are built.
- **[Likelihood Details](likelihood_details.md)** — Conventions of the data and the inner product, frequency ranges, time marginalisation, and how the fast likelihoods choose their bins and bands.
- **[Signal-Duration Check](signal_duration.md)** — How `inject_signal` decides whether a signal fits the segment.
- **[Sampler Details](samplers.md)** — How the SwiG sampler reuses the waveform.
