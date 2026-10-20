# Sampler Details

## SwiG waveform cache

SwiG updates one block of parameters at a time.
The likelihood gives the sampler a cache of the generated waveform polarizations.
A block whose parameters change the waveform refreshes the cache, so the waveform is generated again.
A block that only changes the projection onto the detectors reuses the cache.

When the waveform model supports analytic distance scaling, the luminosity distance is factored out of the waveform, so a `d_L` block reuses the cache.
Otherwise a `d_L` block refreshes it like any other waveform parameter.

The likelihood decides which parameters the waveform depends on after the sample transforms, the likelihood transforms, the fixed parameters and the analytic marginalisation have been applied.

## SMC mutation

At every temperature, the SMC sampler moves all particles together, in coordinates fitted to them.
The steps below run once per temperature, inside the jitted SMC step.

### Coordinates

Each angle that wraps around is cut in the middle of the widest arc that holds no particle.
The angle is rotated so that the cut sits at the ends of its period, then mapped to the real line with a logit.
A mode that straddles the ends of the period therefore stays in one piece, and two modes half a period apart are kept apart by the cut.
The cut is fixed during the temperature step, so the map is an exact change of coordinates.

With a flow, flowMC's masked-coupling rational-quadratic spline is then trained on the particles with `NFModel.train_jit`: the inputs are rescaled by their mean and standard deviation, and the flow trains for `n_epochs` with flowMC's optimizer (clipped AdamW).
Every temperature step trains a newly initialised flow with a fresh optimizer.
The kernel then moves the particles in the flow's latent space.

Every proposal is mapped back to the sampling space and evaluated with the posterior.
The acceptance ratio includes the log-Jacobian of the map, so the particles target the tempered posterior exactly whatever the quality of the flow; a poor flow only lowers the acceptance rate.

### t-pCN

t-pCN (a preconditioned Crank-Nicolson move with a Student-t reference) first fits a Student-t distribution to the particles.
For 24 degrees of freedom $\nu$ between 1 and 1000, an expectation-maximization (EM) fit gives the centre $\mu$ and scale matrix $\Sigma = L L^\top$; the $\nu$ with the highest likelihood is kept.
A particle $z$ then moves to

$$z' = \mu + \sqrt{1 - \rho^2}\,(z - \mu) + \rho \sqrt{W}\, L \epsilon,$$

with $\epsilon$ standard normal and $W$ inverse-gamma with shape $(\nu + d)/2$ and scale $(\nu + \delta)/2$, where $\delta$ is the squared Mahalanobis distance of $z$ from $\mu$ and $d$ the dimension.
This move leaves the fitted Student-t distribution unchanged, so its density ratio replaces the proposal ratio: $\tfrac{1}{2}(\nu + d)\,[\log(1 + \delta'/\nu) - \log(1 + \delta/\nu)]$ is added to the log acceptance ratio.
Where the particles follow the Student-t fit, large moves are accepted.

The step size $\rho$ starts at $\min(2.38/\sqrt{d}, 0.99)$, which is also its cap.
After step $i$ it moves by $(i + 1)^{-3/4}$ times the difference between the mean acceptance probability and `target_acceptance_rate`, and it carries over to the next temperature step.

A temperature step ends once the particles' mean log posterior has not improved for $n_\text{patience}\,(\rho_\text{max}/\rho)^2$ steps, with $n_\text{patience}$ = `n_patience_per_dim` $\times\, d$, or after `n_max_per_dim` $\times\, d$ steps.
The factor $(\rho_\text{max}/\rho)^2$ waits longer while small steps make slow progress.

### Gaussian random walk

The random walk proposes $z + \sqrt{s}\, L \epsilon$, with $L L^\top$ the covariance of the particles at the current temperature, for `n_mcmc_steps_per_dim` $\times\, d$ steps.
After each temperature step, $s$ is multiplied by $\exp[\text{gain}\,(\text{acceptance} - \text{target})]$.

### Likelihood evaluations

Each temperature step evaluates the posterior of all $N$ particles once at the start and once per MCMC step.
BlackJAX evaluates the likelihood once more per step: persistent sampling re-evaluates the moved particles, and tempered SMC uses them to reweight.
The adaptive tempered mode also evaluates the particles to choose the next temperature, and persistent sampling evaluates the initial particles.
With $S_t$ MCMC steps at temperature step $t$, `n_likelihood_evaluations` is therefore $N\,[1 + \sum_t (S_t + 2)]$ for the persistent modes, $N \sum_t (S_t + 3)$ for the adaptive tempered mode, and $N \sum_t (S_t + 2)$ for the fixed tempered mode.

### Credits

The t-pCN move, its step-size adaptation and the plateau stopping rule follow minipcn (M. J. Williams), Grumitt et al. 2024 ([arXiv:2407.07781](https://arxiv.org/abs/2407.07781)) and pocoMC (Karamanis et al.).
