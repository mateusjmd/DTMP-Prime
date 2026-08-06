"""DTMP-Prime"""

__version__ = "0.1.0"

# Instala os aliases de módulo legados (Train_models, Utilite, ...) no sys.meta_path.
from . import compat as _compat

_compat.install_legacy_module_aliases()
