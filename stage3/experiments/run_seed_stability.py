"""Source-checkout shortcut for the installed `qlf-stage3 stability` command."""
from datetime import datetime
from pathlib import Path
import os
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = str(REPOSITORY/"src")
sys.path.insert(0,SOURCE)

from qlf_stage3.stability_cli import main


if __name__ == "__main__":
    # The installed CLI needs no path override. This wrapper also works in a source checkout.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None,[SOURCE,os.environ.get("PYTHONPATH")]))
    supplied = sys.argv[1:]
    defaults = ["--data", str(REPOSITORY/"data/study"), "--features",
                str(REPOSITORY/"runs/study/measurements/training-features-255-thr30.csv")]
    if not any(v == "--out" or v.startswith("--out=") for v in supplied):
        defaults.extend(["--out",str(REPOSITORY/"runs"/("stability_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")))])
    raise SystemExit(main(defaults+supplied))
