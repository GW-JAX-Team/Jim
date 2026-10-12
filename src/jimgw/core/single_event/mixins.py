class _Freezable:
    """Objects that become immutable for likelihood evaluation."""

    _frozen: bool = False

    @property
    def is_frozen(self) -> bool:
        """Whether this object has been frozen."""
        return self._frozen

    def _check_mutable(self) -> None:
        if self.is_frozen:
            raise RuntimeError(
                f"{type(self).__name__} is frozen and cannot be modified."
            )

    def __setattr__(self, name: str, value: object) -> None:
        self._check_mutable()
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        self._check_mutable()
        object.__delattr__(self, name)

    def _freeze(self) -> None:
        if not self.is_frozen:
            self._frozen = True
