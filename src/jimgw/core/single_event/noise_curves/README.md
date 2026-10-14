# Default noise curves

Jim uses these design-sensitivity amplitude spectral densities (ASDs) when no PSD or ASD file is given, for example in `Detector.load_and_set_psd()` and for injections run with `jim-run`.
Each file has two columns, frequency in Hz and strain ASD in $\mathrm{Hz}^{-1/2}$, and starts with a `#` header naming its source.

| Detector | File | Curve | Source | Frequency range |
| --- | --- | --- | --- | --- |
| H1, L1 | `aLIGO_design_asd.txt` | Advanced LIGO design | LIGO-T1800044-v5, L. Barsotti, P. Fritschel, M. Evans, S. Gras (2018), file `aLIGODesign.txt`. <https://dcc.ligo.org/LIGO-T1800044/public> | 5 - 5000 Hz |
| V1 | `AdV_design_asd.txt` | Advanced Virgo design | LIGO-P1200087-v42, LIGO Scientific Collaboration and Virgo Collaboration, Living Rev. Relativ. 19, 1 (2016), [arXiv:1304.0670](https://arxiv.org/abs/1304.0670), file `fig1_adv_sensitivity.txt`. <https://dcc.ligo.org/LIGO-P1200087-v42/public> | 10 - 10000 Hz |
| ET1, ET2, ET3 | `ET_D_asd.txt` | Einstein Telescope ET-D | LIGO-T1500293-v13, M. Evans, R. Sturani, S. Vitale, E. Hall (2020), file `curves_Jan_2020/et_d.txt`. <https://dcc.ligo.org/LIGO-T1500293/public>. ET-D design: S. Hild et al., Class. Quantum Grav. 28, 094013 (2011), [arXiv:1012.0908](https://arxiv.org/abs/1012.0908) | 1 - 10000 Hz |
| CE | `CE_40km_asd.txt` | Cosmic Explorer baseline 40 km | CE-T2000017-v8, Cosmic Explorer Project, file `ce_strain.zip`, `cosmic_explorer_strain.txt`. <https://dcc.cosmicexplorer.org/CE-T2000017/public>. Detector designs: [arXiv:2201.10668](https://arxiv.org/abs/2201.10668), [arXiv:2109.09882](https://arxiv.org/abs/2109.09882) | 5 - 5000 Hz |

SHA-256 of the upstream files these were made from:

| File | Upstream file | SHA-256 |
| --- | --- | --- |
| `aLIGO_design_asd.txt` | `aLIGODesign.txt` | `b90556e379ddb19e8784d3e3a1a45e7899a8337a9e8f10c7fdb644b93b64ec87` |
| `AdV_design_asd.txt` | `fig1_adv_sensitivity.txt` | `70ee22b94b0f183e32ade5ded8dfe1b5564dcbe1757d867ddd8b1f66ae2360fe` |
| `ET_D_asd.txt` | `curves_Jan_2020/et_d.txt` | `6060d0c500108841f25bc4d16feb47eaca791597e1d0dfb7752531dbe5f74736` |
| `CE_40km_asd.txt` | `ce_strain/cosmic_explorer_strain.txt` | `41a4aebd4846c3293dbb4a4ce8fcf93b698b016ae65924229b22580784b6cc08` |

The data rows are unchanged from the upstream files; only the header was added.
The Virgo file keeps the frequency column and the `Design` column of the eight columns in `fig1_adv_sensitivity.txt`.

## Notes

- The curves were retrieved on 2026-10-14.
- The Cosmic Explorer curve is calculated for a source 15 degrees off normal incidence (upstream README).
- `LIGO-T1500293` is titled "unofficial sensitivity curves"; it is the public file we found that holds the ET-D curve.
- Frequencies outside a file's range are filled with the value at its nearest end, so check that the frequency range of your data is inside the range above.
- The documents above state no licence or redistribution terms. The curves belong to their authors and collaborations and are included here with attribution; please cite the source documents when you use them.
- To use a different curve, pass `psd_file` or `asd_file` to `load_and_set_psd`, or `psd_files` / `asd_files` in a `jim-run` config.
