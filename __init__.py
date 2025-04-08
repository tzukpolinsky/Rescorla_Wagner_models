import os
import sys
from pathlib import Path


def _setup_import_paths():
    package_dir = Path(__file__).parent.absolute()
    if str(package_dir.parent) not in sys.path:
        sys.path.insert(0, str(package_dir.parent))
        print(f"Added {package_dir.parent} to Python path")


# Run this when the package is imported
_setup_import_paths()