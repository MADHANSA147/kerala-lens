import hashlib
import json
import os
from pathlib import Path
import tempfile

from .core import ASSETS, ROOT, LABELS, KeralaBoundary, baseline_decision, read_image, threshold_labels

# Set before importing Transformers/Hugging Face; inference never downloads.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HOME'] = str(ASSETS / 'hf-cache')

import numpy as np
import torch
import torch.nn.functional as F


class Predictor:
    def __init__(self, device='cpu', progress=None, use_classifier=True):
        self.progress = progress or (lambda message: None)
        if device not in ('cpu', 'cuda'):
            raise ValueError('Device must be cpu or cuda.')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Select CPU instead.')
        required = ['clip/config.json', 'clip/preprocessor_config.json', 'kerala.geojson',
                    'weights/image_encoder_mlp_weights.pth', 'weights/location_encoder_weights.pth',
                    'weights/logit_scale_weights.pth', 'gps_gallery/coordinates_100K.csv', 'manifest.json']
        missing = [p for p in required if not (ASSETS / p).is_file()]
        if missing:
            raise RuntimeError('Offline assets are missing. Run Setup.cmd with internet access first. Missing: ' + ', '.join(missing))
        self.device = device
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.boundary = KeralaBoundary()
        self.fingerprint = hashlib.sha256((ASSETS / 'manifest.json').read_bytes()).hexdigest()
        self.progress('Loading the local model...')
        from geoclip import GeoCLIP
        self.model = GeoCLIP(from_pretrained=False, clip_path=str(ASSETS / 'clip'))
        # This app only uses image features. Release CLIP's unused text branch on low-RAM PCs.
        del self.model.image_encoder.CLIP.text_model
        del self.model.image_encoder.CLIP.text_projection
        self.model.image_encoder.mlp.load_state_dict(torch.load(ASSETS / 'weights/image_encoder_mlp_weights.pth', map_location='cpu', weights_only=True))
        self.model.location_encoder.load_state_dict(torch.load(ASSETS / 'weights/location_encoder_weights.pth', map_location='cpu', weights_only=True))
        self.model.logit_scale = torch.nn.Parameter(torch.load(ASSETS / 'weights/logit_scale_weights.pth', map_location='cpu', weights_only=True))
        import csv
        with (ASSETS / 'gps_gallery/coordinates_100K.csv').open(newline='') as f:
            reader = csv.reader(f)
            next(reader)
            gps = np.array([[float(r[0]), float(r[1])] for r in reader], dtype=np.float32)
        if gps.ndim != 2 or gps.shape[1] != 2 or not np.isfinite(gps).all() or (np.abs(gps[:, 0]) > 90).any() or (np.abs(gps[:, 1]) > 180).any():
            raise ValueError('Invalid GPS gallery.')
        self.model.gps_gallery = torch.from_numpy(gps)
        self.model.to(device).eval()
        self.gps = gps
        self.locations = None
        self.classifier = None
        self.classifier_path = ROOT / 'trained' / 'classifier.json'
        if use_classifier and self.classifier_path.exists():
            data = json.loads(self.classifier_path.read_text(encoding='utf-8'))
            if data['model_fingerprint'] != self.fingerprint:
                raise ValueError('Trained classifier belongs to different model assets. Retrain it or move trained/classifier.json aside.')
            feature_type = data.get('feature_type', 'clip')
            sizes = {'clip':768, 'geoclip':512, 'combined':1280}
            coef = np.asarray(data['coef'],dtype=np.float64)
            if feature_type not in sizes or coef.shape != (sizes[feature_type],) or not np.isfinite(coef).all() or not np.isfinite(data['intercept']):
                raise ValueError('Invalid trained classifier coefficients.')
            threshold_labels([.5],data['low'],data['high'])
            self.classifier = data

    @torch.inference_mode()
    def features(self, source):
        image = read_image(source)
        pixels = self.model.image_encoder.preprocess_image(image).to(self.device)
        from geoclip.model.image_encoder import _extract_image_features
        raw = _extract_image_features(self.model.image_encoder.CLIP.get_image_features(pixel_values=pixels))
        geo = F.normalize(self.model.image_encoder.mlp(raw), dim=-1)
        return F.normalize(raw, dim=-1).cpu().numpy()[0], geo

    @torch.inference_mode()
    def prepare_gallery(self):
        if self.locations is not None:
            return
        folder = ROOT / 'cache'
        folder.mkdir(exist_ok=True)
        target = folder / ('gps-' + self.fingerprint[:16] + '.npy')
        if target.exists():
            try:
                data = np.load(target, allow_pickle=False)
                if data.shape == (len(self.gps), 512) and np.isfinite(data).all():
                    self.locations = torch.from_numpy(data).to(self.device)
                    return
            except (OSError,ValueError,EOFError):
                pass
            target.unlink()
        output = []
        for start in range(0, len(self.gps), 512):
            if start % 5120 == 0:
                self.progress(f'Preparing offline location index: {start*100//len(self.gps)}%')
            location = self.model.location_encoder(self.model.gps_gallery[start:start+512])
            output.append(F.normalize(location, dim=-1).cpu())
        self.locations = torch.cat(output).to(self.device)
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.npy', delete=False) as f:
            temporary = Path(f.name)
            np.save(f, self.locations.cpu().numpy())
        temporary.replace(target)

    @torch.inference_mode()
    def predict(self, source):
        self.progress('Reading visual clues...')
        raw, geo = self.features(source)
        self.prepare_gallery()
        scores = (geo @ self.locations.T)[0]
        values, indices = torch.topk(scores, min(10, len(scores)))
        candidates = []
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
            lat, lon = map(float, self.gps[index])
            candidates.append({'latitude': lat, 'longitude': lon, 'similarity': round(value, 5),
                               'in_kerala': self.boundary.contains(lat, lon),
                               'boundary_distance_km': round(self.boundary.distance_km(lat, lon), 2)})
        decision, reason = baseline_decision(candidates)
        mode = 'Pretrained GeoCLIP: accuracy on Kerala photos has not been measured'
        classifier_score = None
        exploratory_guess = None
        if self.classifier is not None:
            c = self.classifier
            geo_features = geo.cpu().numpy()[0]
            feature_type = c.get('feature_type', 'clip')
            vector = raw if feature_type == 'clip' else geo_features if feature_type == 'geoclip' else np.concatenate([raw, geo_features]) / np.sqrt(2)
            logit = float(vector @ np.asarray(c['coef'], dtype=np.float64) + c['intercept'])
            classifier_score = float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))
            exploratory_guess = LABELS[int(classifier_score >= .5)]
            decision = int(threshold_labels([classifier_score], c['low'], c['high'])[0])
            reason = (f'Exploratory guess: {exploratory_guess}. Evidence is insufficient for a firm result.' if decision == -1 else
                      'Decision from the Kerala classifier; coordinates are a separate GeoCLIP estimate.')
            mode = 'Pilot model trained on public scene photos. Reliability on your photos is not established.'
        return {'label': LABELS[decision], 'decision': decision, 'reason': reason, 'mode': mode,
                'coordinates_are_approximate': True, 'confidence_percent': None,
                'classifier_score_uncalibrated': classifier_score,
                'exploratory_guess': exploratory_guess,
                'candidates': candidates, 'model_fingerprint': self.fingerprint}
