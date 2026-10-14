# Sampler Details

## SwiG waveform cache

SwiG updates one block of parameters at a time.
The likelihood gives the sampler a cache of the generated waveform polarizations.
A block whose parameters change the waveform refreshes the cache, so the waveform is generated again.
A block that only changes the projection onto the detectors reuses the cache.

When the waveform model supports analytic distance scaling, the luminosity distance is factored out of the waveform, so a `d_L` block reuses the cache.
Otherwise a `d_L` block refreshes it like any other waveform parameter.

The likelihood decides which parameters the waveform depends on after the sample transforms, the likelihood transforms, the fixed parameters and the analytic marginalisation have been applied.
