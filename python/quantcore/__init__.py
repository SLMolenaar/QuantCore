"""QuantCore - High-performance backtesting engine"""
import sys
from pathlib import Path

_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from . import _core
except ImportError:
    try:
        import _core
    except ImportError as e:
        raise ImportError(
            f"Failed to import C++ extension.\n"
            f"Module should be at: {_current_dir / '_core.cp312-win_amd64.pyd'}\n"
            f"Did you build it? Run: python build_module.py"
        ) from e

hello = _core.hello

__version__ = "0.1.0"
__all__ = ['hello']