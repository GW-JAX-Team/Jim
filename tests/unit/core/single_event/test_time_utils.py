import jax

from jimgw.core.single_event.time_utils import (
    cached_greenwich_mean_sidereal_time,
    greenwich_mean_sidereal_time,
)

GPS_TIME = 1126259462.4


class TestCachedGreenwichMeanSiderealTime:
    def test_matches_the_uncached_function(self):
        cached = cached_greenwich_mean_sidereal_time(GPS_TIME)
        assert float(cached) == float(greenwich_mean_sidereal_time(GPS_TIME))

    def test_is_computed_once_per_gps_time(self):
        first = cached_greenwich_mean_sidereal_time(GPS_TIME)
        assert cached_greenwich_mean_sidereal_time(GPS_TIME) is first

    def test_is_a_constant_inside_jit(self):
        # A plain greenwich_mean_sidereal_time call adds about 150 equations to a
        # traced program. The first call must happen while tracing, so empty the cache.
        cached_greenwich_mean_sidereal_time.cache_clear()
        traced = jax.make_jaxpr(
            lambda x: x * cached_greenwich_mean_sidereal_time(GPS_TIME)
        )(1.0)
        assert len(traced.jaxpr.eqns) < 10
