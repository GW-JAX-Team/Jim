# Detector Presets

`get_H1()`, `get_L1()` and `get_V1()` return single detectors with the site coordinates, arm azimuths and tilts written in `jimgw/core/single_event/detector.py`.
`get_ET()`, `get_ET_2L()` and `get_CE()` return `GroundBased3G` detectors, which respond like `GroundBased2G` unless `earth_rotation` or `finite_arm_length` is switched on (see the [data guide](../guides/data.md#next-generation-detectors)).

## Einstein Telescope (Triangular detectors)

`get_ET()` returns three detectors, `ET1`, `ET2` and `ET3`, with 10 km arms and 60 degrees between the two arms of each.
`ET1` is at the Euregio Meuse-Rhine site (about 50.723 N, 5.921 E, at zero elevation), the point of `ET_EMR` below, with its x-arm at the azimuth of the Virgo x-arm.
`ET2` and `ET3` are the other two corners of an equilateral triangle with 10 km sides.
Each corner is found by moving 10 km from the previous one in the direction of its x-arm, along a great circle (haversine formula) on a sphere whose radius is the WGS-84 radius at that latitude.
The arms are rotated by 240 degrees from one corner to the next.

## Einstein Telescope (2 L-shaped detectors)

`get_ET_2L()` returns two L-shaped detectors with 15 km arms and 90 degrees between the two arms of each: `ET_Sar` in Sardinia (about 40.517 N, 9.417 E) and `ET_EMR` in the Euregio Meuse-Rhine (about 50.723 N, 5.921 E), both at zero elevation.
The sites and the way the angle is measured follow [Branchesi et al. (2023)](https://arxiv.org/abs/2303.15923).
The arms of `ET_Sar` have the azimuths of the Virgo arms.
Those of `ET_EMR` are rotated from them by `alignment_angle`, in degrees: 0 for parallel arms and 45 for the misaligned case of that paper.
The default is 42.5.
The rotation is a difference of azimuths, each measured against the local axes of its site rather than against the great circle between the sites.
`load_and_set_psd()` sets the 15 km ET baseline ASD for them; see [Default design-sensitivity ASD](../guides/data.md#default-design-sensitivity-asd).

## Cosmic Explorer

`get_CE()` returns one detector with 40 km arms at the LIGO Hanford site, with the coordinates, elevation and arm tilts of H1.
