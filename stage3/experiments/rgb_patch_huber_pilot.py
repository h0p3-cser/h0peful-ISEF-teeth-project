"""Standalone entry point for the version 1.0 three-view physical-reference pilot."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from qlf_stage3.patch_pilot.cli import main


if __name__=="__main__":
    main()
