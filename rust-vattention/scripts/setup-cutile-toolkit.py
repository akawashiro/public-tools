#!/usr/bin/env python3
"""Install the pinned Tile IR compiler and headers under target (no system changes).

Python 3.12+. Downloads NVIDIA CUDA redistributables and verifies every archive
against the release manifest's SHA256. cuRAND contributes headers only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = '13.2.0'
BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
COMPONENTS = ('cuda_cudart', 'cuda_crt', 'cuda_nvcc', 'cuda_cccl',
              'cuda_tileiras', 'libnvvm', 'libcurand')
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--prefix', type=Path, default=ROOT/'target/cuda-toolkit-13.2')
a = p.parse_args()
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('This setup script currently packages Linux x86_64 only.')
cache = ROOT/'target/cutile-downloads'
cache.mkdir(parents=True, exist_ok=True)
a.prefix.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(BASE+f'redistrib_{VERSION}.json', timeout=60) as response:
    manifest_bytes = response.read()
manifest = json.loads(manifest_bytes)
for component in COMPONENTS:
    spec = manifest[component]['linux-x86_64']
    archive = cache/Path(spec['relative_path']).name
    if not archive.exists():
        print('Download', component, flush=True)
        partial = archive.with_suffix(archive.suffix+'.partial')
        urllib.request.urlretrieve(BASE+spec['relative_path'], partial)
        partial.rename(archive)
    with archive.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != spec['sha256']:
        raise SystemExit(f'Checksum mismatch: {archive}; remove it and retry.')
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = member.name.split('/', 1)
            if len(parts) != 2 or not parts[1]:
                continue
            member.name = parts[1]
            if component == 'libcurand' and not member.name.startswith('include/'):
                continue
            tar.extract(member, a.prefix, filter='data')
    print('Ready', component, flush=True)
(a.prefix/'redistrib-manifest.json').write_bytes(manifest_bytes)
print(f'CUDA_TOOLKIT_PATH={a.prefix.resolve()}')
