import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from kerala_geo.core import KeralaBoundary, baseline_decision, haversine_km, image_digest, read_image, threshold_labels
from kerala_geo.training import choose_thresholds, metrics, read_dataset, validate_training_counts


class CoreTests(unittest.TestCase):
    def test_boundary_holes_and_coordinate_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'boundary.json'
            path.write_text(json.dumps({'type':'Feature','geometry':{'type':'Polygon','coordinates':[
                [[75,8],[78,8],[78,12],[75,12],[75,8]],
                [[76,9],[77,9],[77,10],[76,10],[76,9]]]},'properties':{}}))
            b = KeralaBoundary(path)
            self.assertTrue(b.contains(11,76))
            self.assertFalse(b.contains(9.5,76.5))
            self.assertFalse(b.contains(76,11))
            self.assertTrue(b.contains(8,75))
            self.assertAlmostEqual(b.distance_km(8,75),0)
            for lat,lon in [(float('nan'),76),(91,76),(10,181)]:
                with self.assertRaises(ValueError):
                    b.contains(lat,lon)

    def test_haversine(self):
        self.assertAlmostEqual(float(haversine_km(0,0,0,1)),111.195,places=2)
        self.assertAlmostEqual(float(haversine_km(12,75,12,75)),0)

    def test_mixed_candidates_abstain(self):
        candidates = [{'in_kerala':True,'latitude':10.,'longitude':76.,'boundary_distance_km':30.} for _ in range(5)]
        self.assertEqual(baseline_decision(candidates)[0],1)
        candidates[1]['in_kerala'] = False
        self.assertEqual(baseline_decision(candidates)[0],-1)

    def test_boundary_and_spread_abstain(self):
        candidates = [{'in_kerala':False,'latitude':20.,'longitude':80.,'boundary_distance_km':300.} for _ in range(5)]
        self.assertEqual(baseline_decision(candidates)[0],0)
        candidates[0]['boundary_distance_km'] = 5.
        self.assertEqual(baseline_decision(candidates)[0],-1)
        candidates[0]['boundary_distance_km'] = 300.
        candidates[1]['latitude'] = 40.
        self.assertEqual(baseline_decision(candidates)[0],-1)

    def test_image_validation_and_orientation(self):
        with self.assertRaises(ValueError):
            read_image(io.BytesIO(b'not an image'))
        tiny = io.BytesIO()
        Image.new('RGB',(10,10)).save(tiny,format='PNG')
        tiny.seek(0)
        with self.assertRaises(ValueError):
            read_image(tiny)
        photo = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB',(80,40)).save(photo,format='JPEG',exif=exif)
        photo.seek(0)
        self.assertEqual(read_image(photo).size,(40,80))

    def test_metrics_include_abstentions(self):
        report = metrics([0,0,1,1],[0,-1,0,1])
        self.assertEqual(report['coverage'],.75)
        self.assertEqual(report['accuracy_counting_uncertain_as_incorrect'],.5)
        self.assertAlmostEqual(report['accuracy_on_accepted'],2/3)
        self.assertEqual(report['classes']['kerala']['recall_including_abstentions'],.5)
        self.assertIsNone(metrics([0,1],[-1,-1])['accuracy_on_accepted'])

    def test_threshold_learning_and_no_support(self):
        y = np.array([0]*10+[1]*10)
        scores = np.array([.1]*10+[.9]*10)
        low,high = choose_thresholds(y,scores)
        self.assertEqual((low,high),(.1,.9))
        self.assertEqual(threshold_labels([.1,.5,.9],low,high).tolist(),[0,-1,1])
        self.assertEqual(choose_thresholds([0,1],[.2,.8]),(0.,1.))
        with self.assertRaises(ValueError):
            threshold_labels([.5],.8,.2)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        Image.new('RGB',(64,64),'red').save(self.folder/'a.png')
        Image.new('RGB',(64,64),'blue').save(self.folder/'b.png')

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self,rows):
        path = self.folder/'dataset.csv'
        with path.open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['path','label','group','split'])
            writer.writerows(rows)
        return path

    def test_valid_dataset(self):
        rows=read_dataset(self.manifest([['a.png','kerala','place-a','train'],['b.png','non_kerala','place-b','test']]))
        self.assertEqual([r['label'] for r in rows],[1,0])

    def test_group_leakage(self):
        with self.assertRaisesRegex(ValueError,'multiple splits'):
            read_dataset(self.manifest([['a.png','kerala','same-place','train'],['b.png','kerala','same-place','test']]))

    def test_duplicate_pixels_across_names(self):
        Image.new('RGB',(64,64),'red').save(self.folder/'copy.png')
        self.assertEqual(image_digest(self.folder/'a.png'),image_digest(self.folder/'copy.png'))
        with self.assertRaisesRegex(ValueError,'duplicate image pixels'):
            read_dataset(self.manifest([['a.png','kerala','place-a','train'],['copy.png','kerala','place-b','test']]))

    def test_bad_label(self):
        with self.assertRaisesRegex(ValueError,'label must'):
            read_dataset(self.manifest([['a.png','yes','place-a','train']]))

    def test_insufficient_training(self):
        with self.assertRaisesRegex(ValueError,'need at least'):
            validate_training_counts([])

    def test_geographic_leakage_even_with_different_group_names(self):
        path=self.folder/'geographic.csv'
        path.write_text('path,label,group,split,latitude,longitude\na.png,kerala,one,train,10,76\nb.png,kerala,two,test,10.001,76\n')
        with self.assertRaisesRegex(ValueError,'within 1 km'):
            read_dataset(path)


if __name__ == '__main__':
    unittest.main()
