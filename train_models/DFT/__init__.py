"""
DFT functionals - use the shared module at project root.

The unified implementations are located at:
    /dft_functionals/PBE.py
    /dft_functionals/SVWN3.py

Import directly from dft_functionals:
    from dft_functionals import PBE, SVWN3
"""

import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from dft_functionals import PBE, SVWN3

__all__ = ["PBE", "SVWN3"]
