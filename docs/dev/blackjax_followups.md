# BlackJAX PyPI migration

BlackJAX 1.6 includes the nested-sampling APIs Jim requires: the nested slice
sampler, `blackjax.ns.{base,adaptive,utils}`, and
`blackjax.ns.utils.finalise`.

## Completed

- Jim requires `blackjax>=1.6` from PyPI and the lockfile resolves that release.
- The temporary Git source and `nested-sampling` dependency group were removed.
- NS-AW and NSS import their BlackJAX modules directly; the version-specific
  import guards were removed.
- Installation docs, CI, and pre-commit now use the standard dependency.

## Outstanding external cleanup

- Delete the temporary `GW-JAX-Team/blackjax` fork once an organization admin
  explicitly authorizes that irreversible action.
