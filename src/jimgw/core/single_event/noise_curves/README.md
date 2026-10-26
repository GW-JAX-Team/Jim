# Default noise curves

Jim uses these design-sensitivity amplitude spectral densities (ASDs) when no PSD or ASD file is given, for example in `Detector.load_and_set_psd()` and for injections run with `jim-run`.
Each file has two columns, frequency in Hz and strain ASD in $\mathrm{Hz}^{-1/2}$, and starts with a `#` header naming its source.

| Detector | File | Curve | Source | Frequency range |
| --- | --- | --- | --- | --- |
| H1, L1 | `aLIGO_design_asd.txt` | Advanced LIGO design | LIGO-T1800044-v5, L. Barsotti, P. Fritschel, M. Evans, S. Gras (2018), file `aLIGODesign.txt`. <https://dcc.ligo.org/LIGO-T1800044/public> | 5 - 5000 Hz |
| V1 | `AdV_design_asd.txt` | Advanced Virgo design | LIGO-P1200087-v42, LIGO Scientific Collaboration and Virgo Collaboration, Living Rev. Relativ. 19, 1 (2016), [arXiv:1304.0670](https://arxiv.org/abs/1304.0670), file `fig1_adv_sensitivity.txt`. <https://dcc.ligo.org/LIGO-P1200087-v42/public> | 10 - 10000 Hz |
| ET1, ET2, ET3 | `ET_10km_asd.txt` | Einstein Telescope 10 km baseline (LF+HF) | ETO Design Task Force baseline, F. Sorrentino et al., ET-0482A-25 (2025), <https://apps.et-gw.eu/tds/?r=19915>. Data: <https://gitlab.et-gw.eu/pub/et-sensitivity-curves>, commit `66177ccf4626c77fee243ef21367ac934b2c0ccc`, `10_km/{HF,LF}/data/total.txt` | 1 - 10000 Hz |
| ET_Sar, ET_EMR | `ET_15km_asd.txt` | Einstein Telescope 15 km baseline (LF+HF) | As above, `15_km/{HF,LF}/data/total.txt` | 1 - 10000 Hz |
| CE | `CE_40km_asd.txt` | Cosmic Explorer baseline 40 km | CE-T2000017-v8, Cosmic Explorer Project, file `ce_strain.zip`, `cosmic_explorer_strain.txt`. <https://dcc.cosmicexplorer.org/CE-T2000017/public>. Detector designs: [arXiv:2201.10668](https://arxiv.org/abs/2201.10668), [arXiv:2109.09882](https://arxiv.org/abs/2109.09882) | 5 - 5000 Hz |

The data rows of the LIGO, Virgo and CE files are unchanged from the upstream files; only the header was added.
The Virgo file keeps the frequency column and the `Design` column of the eight columns in `fig1_adv_sensitivity.txt`.
Each ET file combines the HF and LF interferometers of one detector, $\mathrm{ASD} = (\mathrm{ASD}_\mathrm{HF}^{-2} + \mathrm{ASD}_\mathrm{LF}^{-2})^{-1/2}$, from the third upstream column (strain ASD with the pygwinc correction for a signal 15 degrees off zenith).

## Notes

- The curves were retrieved on 2026-10-14, and the ET curves on 2026-10-19.
- The Cosmic Explorer curve is calculated for a source 15 degrees off normal incidence (upstream README).
- The ET curves are for one L-shaped detector with 90 degrees between its arms. The 60 degree arms of the triangle are in the detector response of ET1-ET3, so do not rescale the PSD for them.
- Upstream marks the ET curves as a work-in-progress snapshot (January 2026). `tests/cross_validation/test_noise_curves.py` checks the packaged files against the latest upstream version.
- Frequencies outside a file's range are filled with the value at its nearest end, so check that the frequency range of your data is inside the range above.
- The documents above state no licence or redistribution terms. The curves belong to their authors and collaborations and are included here with attribution; please cite the source documents when you use them.
- To use a different curve, pass `psd_file` or `asd_file` to `load_and_set_psd`, or `psd_files` / `asd_files` in a `jim-run` config.
