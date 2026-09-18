"""Download the original 257/193 train/test split and verify SHA-256."""
from pathlib import Path
import hashlib
import json
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent

def main():
    spec = json.loads((ROOT / 'dataset_manifest.json').read_text())
    archive = ROOT / 'data' / spec['filename']
    destination = ROOT / 'data' / 'balls_128'
    archive.parent.mkdir(exist_ok=True)
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    if not archive.exists() or digest(archive) != spec['sha256']:
        temp = archive.with_suffix('.part')
        print('Downloading', spec['url'], flush=True)
        urllib.request.urlretrieve(spec['url'], temp)
        if digest(temp) != spec['sha256']:
            raise RuntimeError('Dataset checksum mismatch; rerun to download again.')
        temp.replace(archive)
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                raise RuntimeError('Unexpected archive entry: ' + member.name)
        bundle.extractall(destination)
    for split, count in spec['counts'].items():
        samples = [p for p in (destination / split).iterdir() if p.is_dir() and p.name.isdigit()]
        if len(samples) != count:
            raise RuntimeError(f'{split}: expected {count} clips, found {len(samples)}')
    print('Verified dataset ready:', destination)

if __name__ == '__main__':
    main()
