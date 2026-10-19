"""Cross-validation of the packaged ET noise curves against the latest upstream release.

    Jim                                     Reference
    --------------------------------------  ------------------------------------------------
    noise_curves/ET_10km_asd.txt            ET sensitivity curves, main, 10_km/{HF,LF}
    noise_curves/ET_15km_asd.txt            ET sensitivity curves, main, 15_km/{HF,LF}

The reference is https://gitlab.et-gw.eu/pub/et-sensitivity-curves; the
comparison skips when it cannot be reached.
"""

import io
import urllib.error
import urllib.parse
import urllib.request
from importlib import resources

import numpy as np
import pytest

from jimgw.core.single_event.detector import default_asd_files

ET_CURVES_URL = (
    "https://gitlab.et-gw.eu/api/v4/projects/pub%2Fet-sensitivity-curves"
    "/repository/files/{path}/raw?ref=main"
)


def fetch_upstream(path: str) -> np.ndarray:
    url = ET_CURVES_URL.format(path=urllib.parse.quote(path, safe=""))
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return np.loadtxt(io.BytesIO(response.read()))
    except urllib.error.URLError as error:
        pytest.skip(f"ET sensitivity curves unreachable: {error}")


@pytest.mark.slow
class TestETUpstream:
    @pytest.mark.parametrize(
        "name, arm_length", [("ET1", "10_km"), ("ET_Sar", "15_km")]
    )
    def test_packaged_curve_matches_upstream_main(self, name, arm_length):
        hf = fetch_upstream(f"{arm_length}/HF/data/total.txt")
        lf = fetch_upstream(f"{arm_length}/LF/data/total.txt")
        expected = (hf[:, 2] ** -2 + lf[:, 2] ** -2) ** -0.5

        curve = resources.files("jimgw.core.single_event") / "noise_curves"
        frequencies, asd = np.genfromtxt(
            str(curve / default_asd_files[name]), unpack=True
        )
        assert np.array_equal(hf[:, 0], lf[:, 0]), "HF and LF grids differ upstream"
        np.testing.assert_array_equal(frequencies, hf[:, 0])
        np.testing.assert_allclose(
            asd,
            expected,
            rtol=1e-15,
            err_msg="Upstream ET curve changed; update noise_curves/ (see its README).",
        )
