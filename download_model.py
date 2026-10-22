#!/usr/bin/env python3
# Copyright 2026 alphaBell. Licensed under Apache-2.0.
"""Download and verify the alphaBell FP8 checkpoint from GitHub Releases."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request

RELEASE = 'https://github.com/alphabell-research/alphabell-qwen-open/releases/download/v0.1.0-fp8'

def safe_name(name):
    if not name or name in {'.', '..'} or Path(name).name != name or '\\' in name:
        raise ValueError('Unexpected release filename')
    return name

def url(name):
    return RELEASE + '/' + urllib.parse.quote(safe_name(name), safe='')

def safe_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '\\' in name or any(part in {'.', '..'} for part in path.parts) or str(path) != name:
        raise ValueError('Unexpected checkpoint path')
    return path

def digest_file(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('alphaBell-Qwen3.8-27B-Open-FP8'))
    parser.add_argument('--metadata-only', action='store_true', help='Download configuration and documentation only')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url('download-manifest.json'), timeout=60) as response:
        manifest = json.load(response)
    if manifest.get('schema_version') != 1 or manifest.get('tag') != 'v0.1.0-fp8':
        raise ValueError('Unexpected release manifest')
    files = [item for item in manifest['files']
             if not args.metadata_only or not item['name'].endswith('.safetensors')]
    print(f"Downloading {len(files)} files, {sum(item['bytes'] for item in files) / 1e9:.2f} GB", flush=True)
    for item in files:
        target = args.output / safe_path(item['name'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == item['bytes'] and digest_file(target) == item['sha256']:
            print(f'Verified {target.name}', flush=True)
            continue
        temporary = target.with_name(target.name + '.download')
        total = hashlib.sha256()
        count = 0
        with temporary.open('wb') as stream:
            for part in item['parts']:
                print(f"Downloading {part['asset']}", flush=True)
                checksum = hashlib.sha256()
                length = 0
                with urllib.request.urlopen(url(part['asset']), timeout=120) as response:
                    while block := response.read(8 * 1024 * 1024):
                        stream.write(block)
                        checksum.update(block)
                        total.update(block)
                        length += len(block)
                        count += len(block)
                if length != part['bytes'] or checksum.hexdigest() != part['sha256']:
                    raise RuntimeError(f"Download verification failed: {part['asset']}")
        if count != item['bytes'] or total.hexdigest() != item['sha256']:
            raise RuntimeError(f'Checkpoint file verification failed: {target.name}')
        temporary.replace(target)
        print(f'Verified {target.name}', flush=True)
    print(f'Files verified in {args.output.resolve()}', flush=True)

if __name__ == '__main__':
    main()
