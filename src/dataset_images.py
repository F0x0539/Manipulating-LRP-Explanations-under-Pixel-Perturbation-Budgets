"""Resolve the fixed manifest against downloaded validation JPEGs or saved PNGs."""

import hashlib
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/imagenet/manifest.json'


def entries(manifest_path=MANIFEST):
    return json.loads(Path(manifest_path).read_text()).get('images', [])


def source_names(item):
    name = Path(item['source_path']).name
    return (name, name.split('_n', 1)[0] + '.JPEG')


def verify_image(path, item):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest in (item['sha256'], item['source_sha256']):
        return
    # Accept byte-different JPEG metadata only if the decoded RGB pixels match.
    with Image.open(io.BytesIO(data)) as image:
        output = io.BytesIO()
        image.convert('RGB').save(output, format='PNG')
    if hashlib.sha256(output.getvalue()).hexdigest() != item['sha256']:
        raise ValueError(f'Image does not match the fixed manifest: {path}')


def resolve_image(value, image_root=None, manifest_path=MANIFEST):
    path = Path(value).expanduser()
    records = entries(manifest_path)
    item = next((r for r in records if path.name in (Path(r['file']).name, *source_names(r))), None)
    candidates = [path] if path.is_absolute() else [ROOT / path, path]
    if item:
        base = Path(image_root).expanduser() if image_root else Path(manifest_path).parent / 'images'
        candidates.extend(base / name for name in (Path(item['file']).name, *source_names(item)))
        synset = Path(item['source_path']).stem.rsplit('_', 1)[-1]
        candidates.extend(base / synset / name for name in source_names(item))
    for candidate in candidates:
        if candidate.is_file():
            if item:
                verify_image(candidate, item)
            return candidate.resolve()
    raise FileNotFoundError(f'Image not found: {value}. Obtain the ImageNet validation images '
                            'listed in data/imagenet/manifest.json and use --imagenet-dir PATH.')


def recorded_image(value):
    path = Path(value).resolve()
    item = next((r for r in entries() if path.name in (Path(r['file']).name, *source_names(r))), None)
    if item:
        return str(Path('data/imagenet') / item['file'])
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
