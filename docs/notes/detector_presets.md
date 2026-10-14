# Detector Presets

`get_H1()`, `get_L1()` and `get_V1()` return single interferometers with the site coordinates, arm azimuths and tilts written in `jimgw/core/single_event/detector.py`.
`get_ET()` and `get_CE()` return `GroundBased3G` detectors, which respond like `GroundBased2G` unless `earth_rotation` or `finite_arm_length` is switched on (see the [data guide](../guides/data.md#next-generation-detectors)).

## Einstein Telescope

`get_ET()` returns three interferometers, `ET1`, `ET2` and `ET3`, with 10 km arms and 60 degrees between the two arms of each.
`ET1` is at the Virgo site, with its x-arm at the azimuth of the Virgo x-arm.
`ET2` and `ET3` are the other two corners of an equilateral triangle with 10 km sides.
Each corner is found by moving 10 km from the previous one in the direction of its x-arm, along a great circle (haversine formula) on a sphere whose radius is the WGS-84 radius at that latitude.
The arms are rotated by 240 degrees from one corner to the next.

## Cosmic Explorer

`get_CE()` returns one interferometer with 40 km arms at the LIGO Hanford site, with the coordinates, elevation and arm tilts of H1.
