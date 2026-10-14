# Likelihood Details

## Fourier and noise-weighting conventions

The frequency-domain data are the one-sided discrete Fourier transform of the windowed time series, multiplied by the time step $\Delta t$.
The window is not renormalised.
The PSD $S_n(f)$ is one-sided, in $\mathrm{Hz}^{-1}$, and $\Delta f = 1/T$ for a segment of duration $T$.
The noise-weighted inner product is

$$(a|b) = 4\,\Delta f \sum_k \mathrm{Re}\left[\frac{a^*(f_k)\,b(f_k)}{S_n(f_k)}\right],$$

summed over the frequency range of the likelihood.
For each detector the log-likelihood is $(d|h) - \tfrac{1}{2}(h|h)$, without the term $-\tfrac{1}{2}(d|d)$ that does not depend on the parameters, and the detectors are summed.

## Frequency ranges

The Nyquist frequency of the data is computed as $1/(2\Delta t)$.
For some sampling rates (744 Hz, for example) $1/(1/f_s)$ is not exactly $f_s$, so a range that ends at $f_s/2$ is accepted if it equals the Nyquist frequency to a relative $10^{-12}$, both for the data and for the likelihood.
When `f_max` is not given, the likelihood ends at the upper end of the frequency range of the data, or at the highest frequency of the data if that is lower, so the last bin is not lost to that rounding.

Whitened data (`Detector.whitened_frequency_domain_data` and `whitened_time_domain_data`) use the frequency range of the data, not that of a likelihood; for data without a range, set `data.frequency_bounds` to limit them.

## Time marginalisation

With `time_marginalization`, the likelihood is averaged over a uniform prior on $t_c$ in `tc_range`.
The average is a sum over the samples of the FFT time grid that lie inside the range, each weighted by the time spacing $1/(N\,\Delta f)$ of the FFT grid of length $N$, divided by the width $W$ of `tc_range`:

$$\ln \bar L = \ln \left[\frac{1}{W}\sum_{t_k \in \text{tc\_range}} \frac{L(t_k)}{N\,\Delta f}\right].$$

The evidence of a run with time marginalisation includes the prior on $t_c$, normalised over `tc_range`, so it is comparable with a run that samples $t_c$ with a uniform prior on the same range.
The posterior of the other parameters does not depend on the normalisation.

## Heterodyning (Relative binning)

`HeterodynedTransientLikelihoodFD` places the bin edges at equal steps of the phase of the reference waveform.
That phase is modelled as the sum of five power laws in frequency, $(f/f_*)^{\gamma}$ with $\gamma = -5/3, -2/3, 1, 5/3, 7/3$ ([Zackay et al. 2018](https://arxiv.org/abs/1806.08792)), which stand for the Newtonian chirp, the spin-orbit term, the coalescence time and the merger-ringdown.
Each term is scaled so that it changes by at most $2\pi$ rad over the frequency range, and the cumulative sum is divided into equal steps.
With `n_bins` the number of steps is given; with `epsilon` (default 0.5 rad) it is the total phase divided by `epsilon`, rounded down, and at least 1.
Bins where the reference waveform is zero, usually beyond the merger frequency, are removed.

## Multi-banding

`MultibandedTransientLikelihoodFD` follows [Morisaki 2021](https://arxiv.org/abs/2104.07813).
The bands have durations $T, T/2, T/4, \ldots$, where $T$ is the duration of the data.
The first band starts at the lowest `f_min` of the detectors and the last band ends at the highest `f_max`, as in bilby.

The start of each next band is the lowest frequency for which, with the 22-mode time to merger $\tau(f)$ of the reference chirp mass (rescaled by `highest_mode`) and $L$ the `accuracy_factor`,

- the signal still fits in the shorter duration: $\tau + L\sqrt{-\mathrm{d}\tau/\mathrm{d}f} < T_{\text{band}} - t_{\text{offset}}$, and
- the transition from the previous band is smooth: $f - 1/\sqrt{-\mathrm{d}\tau/\mathrm{d}f} > f_{\text{previous}}$.

It is found by bisection.
`time_offset` is the time from the earliest arrival time that the prior allows (allowing for the light travel time across the Earth) to the end of the segment, and `delta_f_end` is the frequency scale of the taper at the high-frequency end.
Both are inferred from the prior on $t_c$ when it is given: with $t_{\text{end}}$ the time from the trigger to the end of the data and $R_\oplus/c$ the light travel time across the Earth, `time_offset` is $t_{\text{end}} - t_{c,\min} + R_\oplus/c$ and `delta_f_end` is $100 / (t_{\text{end}} - t_{c,\max} - R_\oplus/c)$.
Without such a prior they default to 2.12 s and 53 Hz, which assume that the data end 2 s after the trigger; for other segments they must be given.
