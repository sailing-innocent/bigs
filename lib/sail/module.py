from .module_base import ModuleBase
import os

# -------------------------------------------------
# torch should be imported before sail in order to add TORCH relevant DLL path to PATH
import torch
# -------------------------------------------------


class SailModule(ModuleBase):
    def __init__(self):
        name = "sail"  # API Name
        module_name = "sailc"  #  DLL Name
        cwd = os.path.dirname(os.path.abspath(__file__))
        # cwd = os.path.join(cwd, "../release")
        cwd = os.path.join(cwd, "bin/release")
        assert torch.__version__ >= "1.8.0", "torch version should be >= 1.8.0"
        super().__init__(name, module_name, cwd)

    def __getitem__(self, key: str):
        return self.lazy_get_attr(key)

    def __call__(self, key: str):
        return self.lazy_call_func(key)

    @staticmethod
    def get_instance():
        if SailModule._instance is None:
            SailModule._instance = SailModule()
        return SailModule._instance
