"""Loopback-only browser interface. Inference and image processing stay local."""
import io
import json
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder='web', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
_lock = threading.Lock()
_predictor = None


@app.before_request
def local_only():
    if request.host not in ('127.0.0.1:7860', 'localhost:7860', 'localhost'):
        return jsonify(error='Open this app using http://127.0.0.1:7860'), 403
    origin = request.headers.get('Origin')
    if origin and origin != request.host_url.rstrip('/'):
        return jsonify(error='Requests must come from this local app.'), 403


@app.after_request
def response_headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    return response


@app.get('/')
def home():
    return send_from_directory(ROOT / 'web', 'index.html')


@app.get('/api/report')
def report():
    data = json.loads((ROOT / 'trained/report.json').read_text(encoding='utf-8'))
    return jsonify({k: data[k] for k in ('held_out_test', 'held_out_test_forced_binary',
        'held_out_geoclip_top1_boundary_baseline', 'held_out_geoclip_coordinate_error')})


@app.errorhandler(413)
def too_large(error):
    return jsonify(error='Choose an image smaller than 30 MB.'), 413


@app.post('/api/predict')
def predict():
    global _predictor
    if 'image' not in request.files:
        return jsonify(error='Choose a photo first.'), 400
    if not _lock.acquire(blocking=False):
        return jsonify(error='An image is already being analysed. Please try again shortly.'), 409
    try:
        from kerala_geo.core import read_image
        source = io.BytesIO(request.files['image'].read())
        try:
            read_image(source)
        except (ValueError, OSError) as error:
            return jsonify(error='This image could not be read. Use a valid photo, at least 32 pixels wide and high, up to 40 megapixels.'), 400
        source.seek(0)
        if _predictor is None:
            from kerala_geo.model import Predictor
            _predictor = Predictor()
        return jsonify(_predictor.predict(source))
    except Exception:
        app.logger.exception('Local prediction failed')
        return jsonify(error='The local model could not finish. Check the server window for details, then try again.'), 500
    finally:
        _lock.release()


if __name__ == '__main__':
    import sys
    if '--open' in sys.argv:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:7860')).start()
    app.run(host='127.0.0.1', port=7860, debug=False, threaded=True, use_reloader=False)
