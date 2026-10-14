# Signal-Duration Check

`Detector.inject_signal` checks, before it changes anything, that the signal fits the data segment around its peak.
Parts of a signal that fall outside the segment would wrap around to the other end, because the injection is done in the frequency domain.
If the signal does not fit, `inject_signal` raises a `ValueError`; with `allow_wraparound=True` it injects anyway and logs a warning.

The check does not evaluate the waveform.
It estimates how long the signal lasts before and after its peak from the chirp mass, the symmetric mass ratio and the spins, and compares that with the position of the peak in the segment, `trigger_time - start_time + t_c`.
The signal fits if

$$\text{peak} \ge T_{\text{before}} \quad\text{and}\quad \text{peak} + T_{\text{after}} \le \text{duration}.$$

## The estimate

With $M$ the total mass in seconds ($GM/c^3$) and $f_{\text{start}}$ the frequency at which the signal enters the frequency range of the data,

$$T_{\text{before}} = T_{\text{chirp}}(f_{\text{start}}) + 200\,M + \frac{1}{f_{\text{start}}} + \frac{R_\oplus}{c}, \qquad T_{\text{after}} = 1000\,M + \frac{1}{f_{\text{start}}} + \frac{R_\oplus}{c}.$$

- **Chirp time.** The 2PN inspiral time from `XLALSimInspiralChirpTimeBound` in LALSimulation, which drops the negative terms so that it never underestimates:
  $$T_{\text{chirp}} = \frac{5M}{256\,\eta\,v^8}\left[1 + \left(\frac{743}{252} + \frac{11}{3}\eta\right)v^2 + \frac{226}{15}\chi\,v^3 + \left(\frac{3058673}{508032} + \frac{5429}{504}\eta + \frac{617}{72}\eta^2\right)v^4\right], \quad v = (\pi M f_{\text{start}})^{1/3}.$$
  Here $\chi$ is the larger of the two spin magnitudes, taken from the spin components the waveform declares (`s1_x`, `s1_y`, `s1_z`, `s2_x`, `s2_y`, `s2_z`), or 1 if it declares none.
  For aligned-spin waveforms this is LAL's value; for precessing ones the full spin vector is used instead of its $z$ component, which can only make the estimate longer.
- **Starting frequency.** $f_{\text{start}} = 2 f_{\min} / m_{\max}$, with $f_{\min}$ the lower end of the frequency range of the data and $m_{\max}$ the `highest_mode` of the waveform (2 for waveforms with only the 22 mode, 4 for the higher-mode ones).
  A harmonic with azimuthal number $m$ reaches $f_{\min}$ when the orbital frequency is $f_{\min}/m$, which is earlier for larger $m$.
- **Merger and ringdown.** $200\,M$ before the peak and $1000\,M$ after it.
  LAL's merger bound is $54\pi M \approx 170\,M$, and its ringdown bound (11 e-folds) goes from $127\,M$ at zero final spin to $704\,M$ at 0.998.
- **Padding.** One cycle at $f_{\text{start}}$, and the light travel time from the geocentre to a detector, at most $R_\oplus/c \approx 21$ ms.

For a binary with chirp mass 28 and symmetric mass ratio 0.24 observed from 20 Hz, this gives about 1.3 s before the peak for no spin and 1.5 s for maximal spin, and about 0.4 s after it.

## What it does not cover

- It estimates the physical duration of the signal, not the tails that the sharp cut at $f_{\min}$ gives in the time domain.
- Tides, which shorten the inspiral, and eccentricity are not modelled.
- A waveform that does not declare `M_c` and `eta` cannot be checked: `inject_signal` then raises, or warns with `allow_wraparound=True`.

## Comparison with bilby

`bilby.gw.detector.get_safe_signal_duration` uses the same LAL chirp-time bound, with the actual spins and a default lower frequency of 10 Hz.
It returns one segment length, the chirp time rounded up to a power of two with a floor of 4 s, and adds neither the merger, the ringdown nor any padding.
Its check in `inject_signal_into_gwpy_timeseries` constructs a `ValueError` but never raises it.
