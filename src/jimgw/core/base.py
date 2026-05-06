from abc import ABC, abstractmethod

from jaxtyping import Float


class Data(ABC):
    """Abstract base class for data containers."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def fetch(self) -> None:
        """Fetch or load the data into the container."""
        raise NotImplementedError


class LikelihoodBase(ABC):
    """Abstract base class for likelihoods.

    Handles two main components: the data and the model.
    Subclasses must implement `evaluate`.
    """

    _model: object
    _data: object

    @property
    def model(self) -> object:
        """The model used by the likelihood."""
        return self._model

    @property
    def data(self) -> object:
        """The data used by the likelihood."""
        return self._data

    @abstractmethod
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Evaluate the log-likelihood for a given set of parameters.

        Args:
            params: Dictionary mapping parameter names to values.
            data: Auxiliary data (typically empty for pre-loaded data).

        Returns:
            Log-likelihood value.
        """
        raise NotImplementedError
