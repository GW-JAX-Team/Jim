from beartype import beartype as typechecker
from jaxtyping import jaxtyped

from jimgw.core.prior import (
    PowerLawPrior,
)


@jaxtyped(typechecker=typechecker)
class UniformComponentChirpMassPrior(PowerLawPrior):
    """Prior for chirp mass induced by a uniform distribution over component masses.

    When both component masses are drawn uniformly, the chirp mass follows a
    power-law distribution with exponent ``alpha = 1``:

    .. math::

        p(\\mathcal{M}_c) \\propto \\mathcal{M}_c, \\quad
        \\mathcal{M}_c \\in [x_{\\min}, x_{\\max})
    """

    def __repr__(self):
        return f"UniformInComponentsChirpMassPrior(xmin={self.xmin}, xmax={self.xmax}, naming={self.parameter_names})"

    def __init__(self, xmin: float, xmax: float) -> None:
        """
        Args:
            xmin (float): Minimum chirp mass.
            xmax (float): Maximum chirp mass.
        """
        super().__init__(xmin, xmax, 1.0, ["M_c"])
