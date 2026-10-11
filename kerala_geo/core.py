import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from shapely.geometry import Point, shape
from shapely.ops import transform, unary_union

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
LABELS = {0: 'Outside Kerala', 1: 'Kerala', -1: 'Uncertain'}


def read_image(source):
    """Apply camera orientation, ignore GPS EXIF, reject unreasonable inputs."""
    try:
        with Image.open(source) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError('Image exceeds 40 megapixels. Resize it first.')
            if min(image.size) < 32:
                raise ValueError('Image is too small. Use an image at least 32 x 32 pixels.')
            image.load()
            return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Cannot read this image. Choose a valid JPEG, PNG, or WebP file.') from exc


def image_digest(source):
    image = read_image(source)
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def haversine_km(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, [lat1, lon1, lat2, lon2])
    h = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


class KeralaBoundary:
    def __init__(self, path=ASSETS / 'kerala.geojson'):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        features = data['features'] if data['type'] == 'FeatureCollection' else [data]
        self.geometry = unary_union([shape(f['geometry']) for f in features])
        if self.geometry.is_empty or not self.geometry.is_valid:
            raise ValueError('Kerala boundary is empty or invalid. Re-run asset setup.')
        # Local equirectangular approximation, used only for a conservative warning.
        self.project = lambda x, y, z=None: (np.asarray(x)*111.195*math.cos(math.radians(10.5)), np.asarray(y)*111.195)
        self.projected_boundary = transform(self.project, self.geometry.boundary)

    def contains(self, latitude, longitude):
        if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError('Invalid latitude or longitude.')
        return bool(self.geometry.covers(Point(longitude, latitude)))

    def distance_km(self, latitude, longitude):
        point = Point(*self.project(longitude, latitude))
        return float(self.projected_boundary.distance(point))


def baseline_decision(candidates):
    """Explicit, uncalibrated abstention heuristic; not an accuracy guarantee."""
    if len(candidates) < 5:
        return -1, 'Too few candidate locations.'
    top = candidates[:5]
    votes = [c['in_kerala'] for c in top]
    if len(set(votes)) != 1:
        return -1, 'Likely locations fall both inside and outside Kerala.'
    if min(c['boundary_distance_km'] for c in top) < 20:
        return -1, 'A likely location is within about 20 km of the Kerala boundary.'
    spread = max(float(haversine_km(top[0]['latitude'], top[0]['longitude'], c['latitude'], c['longitude'])) for c in top)
    if spread > 250:
        return -1, 'Likely locations are spread more than 250 km apart.'
    return int(votes[0]), 'The top five locations agree. This is an unvalidated visual estimate.'


def threshold_labels(scores, low, high):
    scores = np.asarray(scores)
    if not 0 <= low < high <= 1:
        raise ValueError('Decision thresholds must satisfy 0 <= low < high <= 1.')
    return np.where(scores >= high, 1, np.where(scores <= low, 0, -1))
