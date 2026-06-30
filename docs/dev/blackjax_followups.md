# BlackJAX PyPI follow-ups

## What we're waiting on

A new PyPI release of [blackjax-devs/blackjax](https://github.com/blackjax-devs/blackjax) that includes:

- Nested slice sampler (`blackjax.nss`, `blackjax.ns.{base,adaptive,utils}`).
- `blackjax.ns.utils.finalise` (used by both NS-AW and NSS at run time).

These modules are already merged into `blackjax-devs/blackjax` main.
Jim's `[tool.uv.sources]` currently pins to that branch directly.

## What to undo when a PyPI release ships

### Packaging

- Remove the `[tool.uv.sources]` block in `pyproject.toml` pointing `blackjax = { git = "https://github.com/blackjax-devs/blackjax.git" }`.
- Drop the `[dependency-groups] nested-sampling` PEP 735 group entirely.
- Bump the `blackjax>=1.6` pin to whatever release first contains all NS features.
- Remove the BlackJAX fork owned by GW JAX Team.

### Inline imports → module top

**`src/jimgw/samplers/blackjax/ns_aw.py`**

| Line | Import                                   |
|------|------------------------------------------|
| 144  | `from blackjax.ns.utils import finalise` |

**`src/jimgw/samplers/blackjax/nss.py`**

| Line | Import                                   |
|------|------------------------------------------|
| 107  | `from blackjax.ns.utils import finalise` |

### Docs

- `docs/installation.md`:
  - Drop the "BlackJAX nested samplers" section entirely.
  - `pip install jimgw` is the canonical install instruction.
- `docs/guides/samplers.md`:
  - Drop the "BlackJAX samplers" install preamble section.
  - Restore `pip install jimgw` as the canonical install instruction.
- `README.md`: any BlackJAX install caveats can be deleted.

### CI

- `.github/workflows/CI.yml`: remove `--group nested-sampling` from the test job. The `[dependency-groups] nested-sampling` block in `pyproject.toml` can be deleted at the same time.

### Tests

- `tests/unit/samplers/test_blackjax_*.py`: the `pytest.importorskip("blackjax")` lines can stay as defense-in-depth but are no longer load-bearing.
