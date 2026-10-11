"""Reproduce the reviewed pilot manifest without modifying original downloads."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    data = ROOT / 'data'
    excluded = set(json.loads((data/'curation.json').read_text())['excluded_pageids'])
    rows, seen = [], set()
    for source in ('pilot','pilot-validation-extra'):
        with (data/source/'dataset.csv').open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                pid = int(Path(row['path']).stem)
                if pid in seen or pid in excluded:
                    continue
                seen.add(pid)
                row['path'] = source + '/' + row['path']
                rows.append(row)
    with (data/'scenes.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['path','label','group','split','latitude','longitude'])
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((r['split'],r['label']) for r in rows)
    summary = {'images':len(rows),'groups':len({r['group'] for r in rows}),
               'counts':{f'{a}/{b}':c for (a,b),c in counts.items()},
               'scope':'Public scene-focused pilot; not a general everyday-photo or indoor benchmark.'}
    (data/'scene-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
