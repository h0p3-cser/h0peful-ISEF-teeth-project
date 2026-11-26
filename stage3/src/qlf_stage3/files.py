"""Portable study layout, locked model protocol and file integrity."""
import hashlib
import json
import os
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
PROTOCOL = PACKAGE / 'config' / 'protocol.json'
FEATURE_CSV = 'training-features-255-thr30.csv'
SOURCE_TABLE = 'QLF_GPTadvice/QLF_GPTadvice_S0_S1_S2_C04.csv'


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def relative_path(path, base):
    """Cross-volume Windows runs need an absolute path, relocatable via CLI overrides."""
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def dataset_info(data):
    root = Path(data).expanduser().resolve()
    description = root / 'dataset.json'
    if not description.is_file():
        raise ValueError(f'Missing {description}. Use prepare-data for reviewed inputs or demo-data for the public example.')
    descriptor = load(description)
    if descriptor.get('schema') != 1 or descriptor.get('kind') not in ('study','synthetic'):
        raise ValueError(f'Unsupported dataset descriptor: {description}')
    return root, descriptor


def config(data, features):
    root, descriptor = dataset_info(data)
    return dict(load(PROTOCOL), dataset=root, kind=descriptor['kind'], author_workspace=root / 'workspace',
                capture_catalog=root / 'catalog', feature_csv=Path(features).resolve(), _path=PROTOCOL)


def outside_inputs(out, *paths):
    out = Path(out).expanduser().resolve()
    if any(out.is_relative_to(Path(path).resolve()) for path in paths):
        raise ValueError('The output directory must be outside the input data and measurement directories.')
    return out
