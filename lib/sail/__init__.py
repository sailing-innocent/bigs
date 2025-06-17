__all__ = ["sail", "__version__", "__doc__", "__name__", "add"]

from .module import SailModule

sail = SailModule.get_instance()
# Attributes
# __version__ = sailtorch["__version__"]
# __doc__ = sailtorch["__doc__"]
__name__ = "sail"
# Functions
add = sail("add")
