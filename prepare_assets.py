"""One-time online setup. Prediction never invokes this script."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import concurrent.futures
import time

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
os.environ['HF_HOME'] = str(ASSETS / 'hf-cache')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'


def download_weights(target):
    """Bounded HTTP range requests avoid long single-stream CDN timeouts."""
    import requests
    url = 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors'
    head = requests.get(url, allow_redirects=False, timeout=60)
    head.raise_for_status()
    size = int(head.headers['X-Linked-Size'])
    expected = head.headers['X-Linked-Etag'].strip('"')
    if target.exists() and target.stat().st_size == size:
        with target.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() == expected:
                return
    chunk_size = 8 * 1024 * 1024
    partial = target.with_suffix('.partial')
    done_path = target.with_suffix('.progress.json')
    completed = set(json.loads(done_path.read_text()) if done_path.exists() and partial.exists() else [])
    if not partial.exists():
        with partial.open('wb') as f:
            f.truncate(size)
    direct = head.headers['Location']
    def fetch(index):
        start, end = index * chunk_size, min(size, (index+1)*chunk_size)-1
        for attempt in range(4):
            try:
                response = requests.get(direct, headers={'Range': f'bytes={start}-{end}'}, timeout=(20, 90))
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}' or len(response.content) != end-start+1:
                    raise RuntimeError('CDN returned an unexpected range.')
                return index, start, response.content
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    count = (size + chunk_size-1)//chunk_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool, partial.open('r+b') as f:
        jobs = [pool.submit(fetch, i) for i in range(count) if i not in completed]
        for future in concurrent.futures.as_completed(jobs):
            index, offset, data = future.result()
            f.seek(offset)
            f.write(data)
            f.flush()
            completed.add(index)
            done_path.write_text(json.dumps(sorted(completed)))
            if len(completed) % 10 == 0 or len(completed) == count:
                print(f'Model download: {len(completed)*100//count}%', flush=True)
    with partial.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
            done_path.unlink(missing_ok=True)
            raise RuntimeError('Model checksum mismatch; rerun setup to download again.')
    partial.replace(target)
    done_path.unlink(missing_ok=True)


def download_boundary():
    import requests
    ASSETS.mkdir(exist_ok=True)
    print('Downloading Kerala boundary...', flush=True)
    api = 'https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/'
    response = requests.get(api, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    url = metadata['gjDownloadURL'].replace('github.com/', 'media.githubusercontent.com/media/').replace('/raw/', '/')
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    features = [f for f in response.json()['features']
                if f['properties'].get('shapeName', '').casefold() == 'kerala']
    if len(features) != 1:
        raise RuntimeError('Expected exactly one Kerala boundary; refusing an ambiguous boundary.')
    (ASSETS / 'kerala.geojson').write_text(json.dumps({'type':'FeatureCollection', 'features':features}), encoding='utf-8')
    (ASSETS / 'boundary-source.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    from huggingface_hub import snapshot_download
    ASSETS.mkdir(exist_ok=True)
    print('Downloading the CLIP backbone (about 1.7 GB on first setup)...', flush=True)
    snapshot_download(
        repo_id='openai/clip-vit-large-patch14',
        local_dir=ASSETS / 'clip', revision='32bd64288804d66eefd0ccbe215aa642df71cc41',
        allow_patterns=['config.json', 'preprocessor_config.json', 'tokenizer.json',
                        'tokenizer_config.json', 'special_tokens_map.json',
                        'vocab.json', 'merges.txt'],
    )
    download_weights(ASSETS / 'clip/model.safetensors')
    # GeoCLIP's small trained heads and worldwide GPS gallery ship in its wheel.
    dist = importlib.metadata.distribution('geoclip')
    source = Path(dist.locate_file('geoclip/model'))
    for folder in ['weights', 'gps_gallery']:
        shutil.copytree(source / folder, ASSETS / folder, dirs_exist_ok=True)
    if not (ASSETS / 'kerala.geojson').exists():
        download_boundary()
    files = {}
    for p in ASSETS.rglob('*'):
        if p.is_file() and '.cache' not in p.parts and 'hf-cache' not in p.parts and p.name != 'manifest.json':
            digest = hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            files[str(p.relative_to(ASSETS)).replace('\\', '/')] = {'bytes': p.stat().st_size, 'sha256': digest.hexdigest()}
    (ASSETS / 'manifest.json').write_text(json.dumps({'geoclip': importlib.metadata.version('geoclip'), 'files': files}, indent=2), encoding='utf-8')
    print('Assets saved locally. Run python cli.py doctor, then launch the app.', flush=True)


if __name__ == '__main__':
    main()
