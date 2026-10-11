"""Run after setup: real inference while every socket connection is forbidden."""
import json
import csv
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


attempts = []


def forbidden(*args,**kwargs):
    attempts.append(True)
    raise AssertionError('Inference attempted network access')


socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.socket.sendto = forbidden
socket.getaddrinfo = forbidden
socket.gethostbyname = forbidden

from kerala_geo.model import Predictor
from kerala_geo.core import read_image
import torch
import torch.nn.functional as F

started = time.perf_counter()
model = Predictor(progress=lambda message: print(message,flush=True))
result = model.predict(sys.argv[1])
first_seconds = time.perf_counter()-started
started = time.perf_counter()
repeat = model.predict(sys.argv[1])
second_seconds = time.perf_counter()-started
assert result == repeat, 'Repeated predictions were not deterministic'
assert result['label'] in ('Kerala','Outside Kerala','Uncertain')
assert len(result['candidates']) == 10
assert result['confidence_percent'] is None
# Check that the cached inference path agrees with GeoCLIP's official forward method.
with torch.inference_mode():
    pixels=model.model.image_encoder.preprocess_image(read_image(sys.argv[1])).to(model.device)
    reference=model.model(pixels,model.model.gps_gallery[:128])
    image_features=F.normalize(model.model.image_encoder(pixels),dim=-1)
    cached=model.model.logit_scale.exp()*(image_features @ model.locations[:128].T)
    torch.testing.assert_close(reference,cached,rtol=1e-4,atol=2e-5)
assert not attempts, 'A library attempted network access, even if it caught the error'
comparison=False
saved=ROOT/'trained/test-predictions.csv'
if saved.exists():
    with saved.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if Path(row['path']).resolve() == Path(sys.argv[1]).resolve():
                assert result['decision'] == int(row['prediction'])
                assert abs(result['classifier_score_uncalibrated']-float(row['classifier_score'])) < 1e-6
                comparison=True
report = {'network_connections_allowed':False,'repeat_prediction_identical':True,
          'network_attempts':len(attempts),
          'matches_saved_test_prediction':comparison,
          'matches_official_geoclip_forward':True,
          'first_prediction_seconds':first_seconds,'warm_prediction_seconds':second_seconds,'result':result}
(ROOT/'offline-test.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
