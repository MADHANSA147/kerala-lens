"""Online, licensed Commons pilot collection; never used during prediction.

Coordinates are community-provided weak labels, not independently verified truth.
Cities are assigned to splits before image selection. No model results influence selection.
"""
import csv
import html
import io
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image

from kerala_geo.core import ROOT, KeralaBoundary, image_digest

CENTERS = [
    ('Kochi',9.98,76.28,'train',1), ('Kozhikode',11.26,75.78,'train',1),
    ('Thrissur',10.52,76.21,'train',1), ('Kollam',8.89,76.60,'train',1),
    ('Palakkad',10.78,76.65,'train',1), ('Alappuzha',9.50,76.34,'train',1),
    ('Munnar',10.09,77.06,'train',1), ('Kannur',11.87,75.37,'train',1),
    ('Kottayam',9.59,76.52,'val',1), ('Kalpetta',11.61,76.08,'val',1),
    ('Thiruvananthapuram',8.52,76.94,'test',1), ('Kasaragod',12.50,74.99,'test',1),
    ('Pathanamthitta',9.26,76.78,'test',1),
    ('Mangaluru',12.91,74.85,'train',0), ('Coimbatore',11.01,76.96,'train',0),
    ('Mysuru',12.30,76.65,'train',0), ('Madurai',9.92,78.12,'train',0),
    ('Panaji',15.50,73.83,'train',0), ('Chennai',13.08,80.27,'train',0),
    ('Colombo',6.927,79.861,'train',0), ('Kandy',7.29,80.64,'train',0),
    ('Udupi',13.34,74.75,'val',0), ('Ooty',11.41,76.70,'val',0),
    ('Nagercoil',8.18,77.43,'test',0), ('Tirunelveli',8.71,77.76,'test',0),
    ('Shivamogga',13.93,75.57,'test',0),
]
API = 'https://commons.wikimedia.org/w/api.php'
EXCLUDE = re.compile(r'butterfly|moth\b|insect|spider|dragonfly|damselfly|beetle|flower|bird\b|portrait|map\b|diagram|logo|flag\b|stamp\b|scan\b|drawing|painting|herbarium|specimen|snake|lizard|fungus|mushroom|museum.*object', re.I)


def clean(value):
    return html.unescape(re.sub('<[^>]+>', '', value or '')).strip()


def main(folder=None, centers=None, limit=16):
    folder = Path(folder) if folder else ROOT / 'data/pilot'
    images = folder / 'images'
    cache = folder / 'source-cache'
    images.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    boundary = KeralaBoundary()
    session = requests.Session()
    session.headers['User-Agent'] = 'KeralaGeoResearchPrototype/0.1 (educational geolocation; Python requests)'
    def query(name, params):
        path = cache / (name + '.json')
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        for attempt in range(3):
            try:
                response = session.get(API, params={'action':'query','format':'json', **params}, timeout=60)
                response.raise_for_status()
                data = response.json()
                if 'error' in data:
                    raise RuntimeError(str(data['error']))
                path.write_text(json.dumps(data), encoding='utf-8')
                time.sleep(.3)
                return data
            except (requests.RequestException, RuntimeError):
                if attempt == 2:
                    raise
                time.sleep(3)
    rows, attribution, failures = [], [], []
    used_ids, digests = set(), set()
    for city, lat, lon, split, label in (centers or CENTERS):
        print(f'Collecting {city} ({split})...', flush=True)
        listing = query(city, {'list':'geosearch','gscoord':f'{lat}|{lon}','gsradius':10000,'gslimit':500,'gsnamespace':6})
        candidates = [r for r in listing.get('query',{}).get('geosearch',[])
                      if r['title'].lower().endswith(('.jpg','.jpeg','.png')) and not EXCLUDE.search(r['title'])
                      and r['pageid'] not in used_ids
                      and int(boundary.contains(r['lat'],r['lon'])) == label]
        random.Random(42).shuffle(candidates)
        selected, authors, tiles = 0, Counter(), Counter()
        for offset in range(0, min(300,len(candidates)), 50):
            batch = candidates[offset:offset+50]
            details = query(f'{city}-info-{offset}', {'prop':'imageinfo','pageids':'|'.join(str(r['pageid']) for r in batch),
                            'iiprop':'url|size|extmetadata','iiurlwidth':640})
            pages = details.get('query',{}).get('pages',{})
            for candidate in batch:
                pid = candidate['pageid']
                information = pages.get(str(pid),{}).get('imageinfo',[])
                if not information:
                    continue
                info = information[0]
                meta = info.get('extmetadata',{})
                field = lambda key: clean(meta.get(key,{}).get('value',''))
                licence = field('LicenseShortName')
                if not (licence.startswith(('CC BY', 'CC0', 'Public domain', 'PDM'))):
                    continue
                author = field('Artist')
                if not author or authors[author] >= 4 or min(info.get('width',0),info.get('height',0)) < 224:
                    continue
                tile = (round(candidate['lat']/.003),round(candidate['lon']/.003))
                if tiles[tile] >= 2:
                    continue
                path = images / f'{pid}.jpg'
                try:
                    if not path.exists():
                        response = session.get(info.get('thumburl',info['url']), timeout=45)
                        response.raise_for_status()
                        with Image.open(io.BytesIO(response.content)) as image:
                            if image.width * image.height > 40_000_000:
                                continue
                            image.convert('RGB').save(path, quality=92)
                        time.sleep(.12)
                    digest = image_digest(path)
                    if digest in digests:
                        continue
                except (requests.RequestException,OSError,ValueError) as exc:
                    failures.append({'pageid':pid,'error':str(exc)[:200]})
                    continue
                used_ids.add(pid)
                digests.add(digest)
                authors[author] += 1
                tiles[tile] += 1
                rows.append({'path':f'images/{pid}.jpg','label':'kerala' if label else 'non_kerala',
                             'group':city, 'split':split, 'latitude':candidate['lat'],'longitude':candidate['lon']})
                attribution.append({'pageid':pid,'title':candidate['title'],'author':author,'license':licence,
                                    'license_url':field('LicenseUrl'), 'source':info['descriptionurl'],
                                    'download_url':info.get('thumburl',info['url']), 'change':'Resized thumbnail converted to JPEG',
                                    'sha256_pixels':digest, 'city':city, 'split':split})
                selected += 1
                if selected >= limit:
                    break
            if selected >= limit:
                break
        print(f'  {selected} photos retained', flush=True)
        with (folder/'dataset.csv').open('w',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['path','label','group','split','latitude','longitude'])
            writer.writeheader()
            writer.writerows(rows)
        (folder/'attribution.json').write_text(json.dumps(attribution,indent=2,ensure_ascii=False),encoding='utf-8')
        (folder/'collection-errors.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    counts = Counter((r['split'],r['label']) for r in rows)
    summary = {'counts':{f'{s}/{l}':v for (s,l),v in counts.items()},
               'limitations':['Community geotags are weak labels, not audited ground truth.',
                              'Public landmark/tourist photos differ from everyday homes, roads and interiors.',
                              'Outside-Kerala samples cover selected South Asian cities, not the whole world.',
                              'Pretrained model training overlap cannot be excluded.',
                              'City splits prevent overlap for the new classifier only; photographers may overlap.']}
    (folder/'collection-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2), flush=True)


if __name__ == '__main__':
    main()
