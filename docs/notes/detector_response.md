# Detector Response Conventions

## Finite-arm-length phase convention

The finite-arm-length response multiplies each arm's antenna pattern by a transfer function ([Baral et al. 2023](https://arxiv.org/abs/2304.09889), Eq. 7), normalised so that it tends to one at low frequency.
Its overall phase depends on which instant of the light's round trip in the arm the detector output is assigned to, and different codes make different choices.
Jim uses this one:

- The gravitational wave is referenced at the detector vertex, as for `GroundBased2G`.
- The output is assigned to the time the light reaches the **end mirror**, i.e. half-way through its round trip, not to the time it returns to the beam splitter.

With $x = fL/c$ for arm length $L$, $y = \mathbf{k}\cdot\mathbf{n}$ the projection of the propagation direction $\mathbf{k}$ onto the arm $\mathbf{n}$, and Jim's Fourier convention, in which a time delay $\tau$ multiplies the frequency-domain signal by $e^{-2\pi i f \tau}$, the transfer function is

$$T(f) = \frac{1}{2}\left[e^{-i\pi x(1+y)}\,\mathrm{sinc}\,x(1-y) + e^{i\pi x(1-y)}\,\mathrm{sinc}\,x(1+y)\right],$$

with $\mathrm{sinc}\,z = \sin(\pi z)/(\pi z)$.
At low frequency $T \approx 1 - i\pi x y$, so it adds no direction-independent time delay to the long-wavelength response: injections made with or without the finite-arm-length response share the same time reference.

Codes that assign the output to the light's return at the beam splitter, following the round-trip construction of [Rakhmanov et al. (2008)](https://arxiv.org/abs/0808.3805), use

$$T_{\text{return}}(f) = e^{-2\pi i f L/c}\, T(f),$$

which delays the signal by $L/c$: about 33 μs for the 10 km arms of ET and 133 μs for the 40 km arms of CE.
For one detector, or a network with equal arm lengths, the difference is absorbed into $t_c$.
For a network with unequal arm lengths, such as ET with CE, it changes the relative arrival times by about 100 μs and can shift the inferred sky position.
Injecting and recovering with `GroundBased3G` is consistent; when the data come from another code, check which convention it uses and apply the factor above if they differ.
