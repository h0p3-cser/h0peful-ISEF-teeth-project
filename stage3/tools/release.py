"""Build the wheel, sdist and public GitHub-ready source ZIP, then checksum them."""
import argparse
from hashlib import sha256
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile


def build(out, no_isolation):
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf8"))["project"]["version"]
    prefix = f"qlf-stage3-{version}"
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".stage3-release-", dir=out) as work:
        work = Path(work)
        cmd = [sys.executable, "-m", "build", "--outdir", str(work)]
        if no_isolation:
            cmd.append("--no-isolation")
        subprocess.run([*cmd, str(root)], check=True)
        sdist, = work.glob("*.tar.gz")
        wheel, = work.glob("*.whl")
        archive = work / (prefix + "-source.zip")
        names = set()
        with tarfile.open(sdist, "r:gz") as source, ZipFile(archive, "w", ZIP_DEFLATED) as target:
            for member in source.getmembers():
                if member.isdir():
                    continue
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or len(parts) < 2 or ".." in parts or parts[0] != sdist.name[:-7]:
                    raise ValueError(f"Unexpected source entry: {member.name}")
                name = "/".join(parts[1:])
                if parts[1] in {"data", "runs", ".venv", "dist", "build", ".git"} or name.endswith((".pth", ".joblib")):
                    raise ValueError(f"Private or generated file in source distribution: {name}")
                if set(parts) & {"PKG-INFO", "qlf_stage3.egg-info"} or name == "setup.cfg":
                    continue
                names.add(name)
                target.writestr(prefix + "/" + name, source.extractfile(member).read())
        required = {"README.md", "LICENSE", "NOTICE.md", "CONTRIBUTING.md", "CITATION.cff",
                    ".github/workflows/ci.yml", "pyproject.toml", "tools/release.py",
                    "tests/check_endpoint.py", "src/qlf_stage3/config/endpoint-reference.json",
                    "src/qlf_stage3/config/rgb-patch-huber-protocol-v1.md"}
        if not required.issubset(names):
            raise ValueError(f"Missing public source files: {sorted(required - names)}")
        sums = work / (prefix + "-SHA256SUMS.txt")
        sums.write_text("".join(f"{sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
                                 for p in (archive, sdist, wheel)), encoding="ascii")
        for path in (archive, sdist, wheel, sums):
            target = out / path.name
            path.replace(target)
            print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "dist")
    parser.add_argument("--no-isolation", action="store_true", help="Use the current build/setuptools/wheel environment")
    args = parser.parse_args()
    build(args.out.resolve(), args.no_isolation)
