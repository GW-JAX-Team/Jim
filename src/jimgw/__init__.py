from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jimgw")
except PackageNotFoundError:
    __version__ = "unknown"
