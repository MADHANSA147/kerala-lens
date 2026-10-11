"""Import images from user-supplied ZIPs without guessing geographic labels."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archives', nargs='+', type=Path)
    args = parser.parse_args()
    folder = ROOT/'data/user-unlabelled'
    (folder/'images').mkdir(parents=True,exist_ok=True)
    records, errors, gps_found, coordinate_tags = [], [], 0, 0
    for archive in args.archives:
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                extension=Path(member.filename).suffix.lower()
                if member.is_dir() or extension not in ('.jpg','.jpeg','.png','.webp'):
                    continue
                if member.file_size > 30_000_000:
                    errors.append({'archive':archive.name,'member':member.filename,'error':'Image file exceeds 30 MB import limit'})
                    continue
                try:
                    data=z.read(member)
                    with Image.open(io.BytesIO(data)) as im:
                        if im.width*im.height > 40_000_000 or min(im.size)<32:
                            raise ValueError('Image dimensions outside supported limits')
                        gps_fields=im.getexif().get_ifd(34853)
                        gps=bool(gps_fields)
                        has_coordinates=all(k in gps_fields for k in (1,2,3,4))
                        coordinate_tags += int(has_coordinates)
                        gps_found += int(gps)
                        oriented=ImageOps.exif_transpose(im).convert('RGB')
                        pixels=hashlib.sha256(str(oriented.size).encode()+oriented.tobytes()).hexdigest()
                        width,height=oriented.size
                    digest=hashlib.sha256(data).hexdigest()
                    path=folder/'images'/(digest+extension)
                    if not path.exists():
                        path.write_bytes(data)
                    records.append({'path':'images/'+path.name,'label':'unknown','group':'unverified',
                                    'split':'unlabelled','archive':archive.name,'original_name':member.filename,
                                    'width':width,'height':height,'gps_metadata_present':gps,
                                    'gps_coordinates_present':has_coordinates,
                                    'sha256_file':digest,'sha256_pixels':pixels})
                except (OSError,ValueError,Image.DecompressionBombError,zipfile.BadZipFile) as exc:
                    errors.append({'archive':archive.name,'member':member.filename,'error':str(exc)})
    (folder/'inventory.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    (folder/'import-errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    fields=['path','label','group','split']
    with (folder/'labels-to-verify.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in records)
    summary={'files_imported':len(records),'unique_pixel_images':len({r['sha256_pixels'] for r in records}),
             'files_with_gps_metadata':gps_found,'files_with_coordinate_tags':coordinate_tags,
             'errors':len(errors),'used_for_geographic_training':False,
             'reason':'Capture locations are unverified. Cultural objects and clothing do not establish the capture location.'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
