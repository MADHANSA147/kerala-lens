import argparse
import hashlib
import json
from pathlib import Path
import sys


def doctor(verify=False):
    from kerala_geo.core import ASSETS, KeralaBoundary
    import importlib.metadata
    import torch
    info = {'python': sys.version.split()[0], 'cuda_available': torch.cuda.is_available(),
            'packages': {p: importlib.metadata.version(p) for p in ['geoclip','torch','transformers','shapely','Pillow','scikit-learn']}}
    manifest = json.loads((ASSETS / 'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = ASSETS / relative
        if not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Missing or incomplete asset: {relative}')
        if verify:
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(4*1024*1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected['sha256']:
                raise ValueError(f'Asset checksum failed: {relative}')
    boundary = KeralaBoundary()
    checks = {'Kochi': boundary.contains(9.9312,76.2673), 'Thiruvananthapuram': boundary.contains(8.5241,76.9366),
              'Chennai_outside': not boundary.contains(13.0827,80.2707), 'Bengaluru_outside': not boundary.contains(12.9716,77.5946)}
    if not all(checks.values()):
        raise ValueError(f'Boundary sanity checks failed: {checks}')
    info.update({'asset_checks': 'sha256' if verify else 'file sizes', 'boundary_checks': checks,
                 'accuracy': 'Not measured unless you have trained/evaluated on your own labelled data'})
    return info


def main():
    parser = argparse.ArgumentParser(description='Offline GeoCLIP Kerala classifier')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('predict')
    p.add_argument('image')
    p.add_argument('--output', type=Path)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    d = commands.add_parser('doctor')
    d.add_argument('--verify', action='store_true')
    for name in ('train','evaluate'):
        p = commands.add_parser(name)
        p.add_argument('manifest')
        if name == 'evaluate':
            p.add_argument('--output', type=Path, default=Path('evaluation.json'))
    args = parser.parse_args()
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    try:
        if args.command == 'doctor':
            result = doctor(args.verify)
        elif args.command == 'predict':
            from kerala_geo.model import Predictor
            result = Predictor(device=args.device, progress=progress).predict(args.image)
        else:
            from kerala_geo.training import train, evaluate
            result = (train if args.command == 'train' else evaluate)(args.manifest, progress)
        text = json.dumps(result, indent=2)
        if getattr(args, 'output', None):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding='utf-8')
        print(text)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
