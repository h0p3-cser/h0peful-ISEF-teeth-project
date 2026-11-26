"""Location of the local experimental inputs."""
import os
from pathlib import Path

PROJECT = Path(os.environ.get("QLF_DATA_ROOT", ".")).expanduser().resolve()
LOCAL_VIEWS = PROJECT / "student-views.json"
