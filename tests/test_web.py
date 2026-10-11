import io
import unittest
from unittest.mock import Mock
from PIL import Image
import web_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()
        self.old = web_app._predictor

    def tearDown(self):
        web_app._predictor = self.old

    def test_page_and_report(self):
        with self.client.get('/') as response:
            self.assertEqual(response.status_code, 200)
        response = self.client.get('/api/report')
        self.assertEqual(response.json['held_out_test']['images'], 53)
        self.assertNotIn('dataset', response.json)

    def test_missing_and_invalid_upload(self):
        self.assertEqual(self.client.post('/api/predict').status_code, 400)
        response = self.client.post('/api/predict', data={'image': (io.BytesIO(b'bad'), 'bad.jpg')})
        self.assertEqual(response.status_code, 400)

    def test_foreign_origin_and_host(self):
        self.assertEqual(self.client.post('/api/predict', headers={'Origin':'https://example.com'}).status_code, 403)
        self.assertEqual(self.client.get('/', headers={'Host':'example.com'}).status_code, 403)

    def test_valid_upload_and_busy(self):
        def photo():
            buf = io.BytesIO(); Image.new('RGB', (64,64)).save(buf, format='PNG'); buf.seek(0)
            return {'image': (buf, 'photo.png')}
        web_app._predictor = Mock()
        def read(source):
            self.assertEqual(Image.open(source).size, (64,64))
            return {'label':'Uncertain', 'candidates':[]}
        web_app._predictor.predict.side_effect = read
        self.assertEqual(self.client.post('/api/predict', data=photo()).json['label'], 'Uncertain')
        web_app._lock.acquire()
        try:
            self.assertEqual(self.client.post('/api/predict', data=photo()).status_code, 409)
        finally:
            web_app._lock.release()

    def test_size_limit(self):
        before = web_app.app.config['MAX_CONTENT_LENGTH']
        web_app.app.config['MAX_CONTENT_LENGTH'] = 10
        try:
            self.assertEqual(self.client.post('/api/predict', data={'image': (io.BytesIO(b'a'*100),'a.jpg')}).status_code, 413)
        finally:
            web_app.app.config['MAX_CONTENT_LENGTH'] = before


if __name__ == '__main__':
    unittest.main()
