import sys
import importlib
import logging

logger = logging.getLogger("ModuleBase")
logger.setLevel(logging.INFO)

# Add console handler if not already configured
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Prevent duplicate messages from parent loggers
    logger.propagate = False


class ModuleBase:
    _instance = None
    _name: str
    _module_name: str
    _cwd: str
    _module = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModuleBase, cls).__new__(cls)
        return cls._instance

    def __init__(self, name: str, module_name: str, cwd: str):
        if not hasattr(self, "_initialized"):
            # print("ModuleBase init: initializing")
            self._name = name
            self._module_name = module_name
            self._cwd = cwd
            logger.info(
                "loading module {} from {}".format(self._module_name, self._cwd)
            )
            self.load_module()
            self._initialized = True

    def load_module(self):
        sys.path.append(self._cwd)
        try:
            self._module = importlib.import_module(self._module_name)
            logger.info("module {} loaded".format(self._module_name))
        except ImportError:
            logger.error("module {} not found".format(self._module_name))
            return None

    def lazy_get_attr(self, name: str):
        assert self._module is not None, f"module {self._module_name} is not loaded"
        return getattr(self._module, name)

    def lazy_call_func(self, name: str):
        assert self._module is not None, f"module {self._module_name} is not loaded"

        def _call(*args, **kwargs):
            logger.info("calling {} in module {}".format(name, self._module_name))
            return getattr(self._module, name)(*args, **kwargs)

        return _call
