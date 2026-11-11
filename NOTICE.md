# Third-party notices

## bilby and bilby-cython

`src/jimgw/core/single_event/transform_utils.py` and `polarization.py` include
code adapted from [bilby](https://github.com/bilby-dev/bilby).
Copyright (c) 2018 Paul D. Lasky.

`src/jimgw/core/single_event/transform_utils.py` and `time_utils.py` include
code adapted from [bilby-cython](https://github.com/bilby-dev/bilby_cython).
Copyright (c) 2022 Colm Talbot.

Both are MIT licensed; the MIT permission notice is in [LICENSE](LICENSE).

## Acceptance-walk kernel

`src/jimgw/samplers/blackjax/ns/_acceptance_walk_kernel.py` is adapted and
modified from the kernel by Metha Prathaban, David Yallup, James Alvey, Ming
Yang, Will Templeton and Will Handley, deposited as
`gpu_gw_zenodo/src/custom_ns_kernels/acceptance_walk_kernel.py` in
[Gravitational-wave inference at GPU speed](https://doi.org/10.5281/zenodo.17012011)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The same file
is at
<https://github.com/mrosep/blackjax_ns_gw/blob/aa2dd39d0b1cc43bbb393f8212b5d2c6056eb2d9/src/custom_kernels/acceptance_walk.py>.
