"""Cross-validate Jim's default Tukey window against bilby."""

import numpy as np
import pytest

from jimgw.core.single_event.data import DEFAULT_TUKEY_ROLL_OFF, Data

bilby = pytest.importorskip("bilby")


@pytest.mark.parametrize("duration", [4, 128])
def test_default_tukey_window_matches_bilby(duration: int) -> None:
    """Jim and bilby calculate alpha identically for any duration."""
    sampling_frequency = 16
    strain = np.ones(duration * sampling_frequency)

    jim_data = Data(td=strain, delta_t=1 / sampling_frequency)

    bilby_data = bilby.gw.detector.InterferometerStrainData(
        roll_off=DEFAULT_TUKEY_ROLL_OFF
    )
    bilby_data.set_from_time_domain_strain(
        time_domain_strain=strain,
        sampling_frequency=sampling_frequency,
        duration=duration,
    )

    expected_alpha = 2 * bilby_data.roll_off / duration
    assert bilby_data.alpha == pytest.approx(expected_alpha)
    np.testing.assert_allclose(jim_data.window, bilby_data.time_domain_window())
