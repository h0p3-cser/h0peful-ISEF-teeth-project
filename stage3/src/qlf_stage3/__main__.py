"""Allow the package to be executed with ``python -m qlf_stage3``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
