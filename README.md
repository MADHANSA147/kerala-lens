# Kerala Lens — offline visual geolocation

Choose a photo to receive a Kerala / Outside Kerala / Uncertain estimate and approximate GPS candidates. Photos stay on your computer. The app does not use embedded GPS metadata to make predictions.

**No image-geolocation system is error-free.** Model accuracy must be read together with dataset size, coverage, and test conditions. This project includes real evaluation tools; it does not promise a specific accuracy before measurement.

**Current measured result:** 73.6% exploratory binary accuracy on 53 scene photos from held-out towns, versus 60.4% for GeoCLIP's top-coordinate boundary check. Conservative mode returned a class for only 4/53 photos. This is a working research prototype, **not yet a dependable high-accuracy location verifier**. See [RESULTS.md](RESULTS.md).

## Open the app

For the **local website**, double-click **Launch Web.cmd**, then open **http://127.0.0.1:7860** in your browser. Keep the server window open while using the website; close it to stop the server. It only listens on this computer. Choose or drop a photo, click **Analyse photo**, and expand **Suggested coordinates** to inspect the estimates. **Download result** saves the prediction as JSON. The page includes the measured accuracy and uncertainty rate. No external fonts, maps, or services are used.

The first web prediction loads the model; subsequent predictions reuse it. Only one photo is processed at a time. Uploaded images are processed in memory and are not saved by the web app. If the site is already running, open the address directly instead of launching a second server.

On this computer, double-click **Launch.cmd** in this folder after setup has completed. Choose an image, then **Analyse photo**. Save results as JSON if needed.

The first prediction loads the model and builds a local coordinate index. Later predictions reuse it. CPU operation is supported. A compatible NVIDIA PyTorch installation can use `--device cuda` through the command line.

The coordinate index is already prepared on this computer. The app's **View accuracy report** button explains measured performance. An exploratory guess may be shown alongside an uncertain result; it should not be treated as a confident answer.

## Set up on another computer

The public repository includes the application, fitted regional classifier, evaluation report, and data collection/training tools. Photo datasets, private uploads, downloaded foundation model weights, and local caches are excluded. Initial setup downloads the required model assets; subsequent inference works offline. Dataset manifests and attribution records are generated locally when collecting photos. The included evaluation report records the original pilot run, not a guarantee of future performance.

1. Install 64-bit Python 3.11 or newer with Tcl/Tk support.
2. Keep this project folder together. Double-click **Setup.cmd** while online. The pretrained model download is approximately 1.7 GB; allow additional disk space for Python libraries, the location index, and photos.
3. After setup succeeds, predictions work offline. You can also transfer the completed `assets` folder, but create a fresh Python environment on the destination computer. A virtual environment is not portable.

The environment prepared here reuses already installed Python packages to avoid reinstalling PyTorch. `requirements.txt` describes a fresh install; `environment-tested.txt` records the actual versions used here.

## Command-line use

Run these commands from this project folder:

```powershell
.venv\Scripts\python.exe cli.py doctor --verify
.venv\Scripts\python.exe cli.py predict "C:\path\photo.jpg" --output result.json
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tests\offline_integration.py "C:\path\photo.jpg"
```

`doctor --verify` validates asset checksums and tests known inside/outside coordinates. The offline integration test blocks socket connections, runs the real model twice, and checks repeatability. The GUI smoke test initializes a hidden window and exercises its result/error rendering; it is not a full manual usability test.

## Train on your photos

Use a CSV with these columns:

```csv
path,label,group,split
photos/kochi_01.jpg,kerala,kochi_neighbourhood_a,train
photos/mysuru_01.jpg,non_kerala,mysuru_neighbourhood_b,train
photos/kottayam_01.jpg,kerala,kottayam_neighbourhood_c,val
photos/udupi_01.jpg,non_kerala,udupi_neighbourhood_d,val
photos/kasaragod_01.jpg,kerala,kasaragod_neighbourhood_e,test
photos/nagercoil_01.jpg,non_kerala,nagercoil_neighbourhood_f,test
```

These are format examples, not supplied files. Paths are relative to the CSV. Labels must be `kerala` or `non_kerala`. Optional `latitude,longitude` columns support geographic overlap checks and coordinate-error measurement.

Put all photos of a location or capture session in one group and one split. Use different locations for training, validation, and testing. The program requires at least 20 training, 10 validation, and 10 test images **per class**, from at least two groups per class in each split. These are only software minimums. A dependable product needs a much larger and more representative dataset.

```powershell
.venv\Scripts\python.exe cli.py train dataset.csv
.venv\Scripts\python.exe cli.py evaluate dataset.csv --output evaluation.json
```

Training automatically selects between CLIP features, GeoCLIP features, and their combination using the validation split. It saves `trained/classifier.json`, `trained/report.json`, and per-image test results. Restart the desktop app to load a newly trained classifier. Previous classifier/report files are retained as `.previous` when overwritten.

Evaluation uses only `test` rows. The app uses the trained classifier when one exists; otherwise it uses the explicitly unvalidated GeoCLIP baseline. The trained classifier affects the Kerala decision, not the coordinate model.

## Public pilot dataset

`collect_pilot.py` is an optional **online** collector of openly licensed, geotagged Wikimedia Commons images. It assigns whole cities to separate splits in advance, limits repeated photographers and tightly clustered coordinates, and records attribution. It is not invoked by the offline app.

Community coordinates need review. Public tourist and landmark photos are not representative of arbitrary roads, homes, interiors, or screenshots. Do not present pilot accuracy as production accuracy. Read `RESEARCH.md` and the generated measurement report for details.

The reviewed manifest for this build is `data/scenes.csv`. It combines the original collection and extra validation photos, removes duplicate page IDs, and applies the pre-evaluation visual inclusion decisions in `data/curation.json`. Run `build_scene_manifest.py` to reproduce it. Collection source records are retained in both pilot directories.

## Your supplied ZIP archives

All 355 images were imported into `data/user-unlabelled/`. Original ZIPs remain untouched. Because capture locations are unknown, these photos are not assigned Kerala/non-Kerala labels and were not used to train or measure geographic accuracy. Clothing styles and objects do not establish where an image was taken.

The folder includes an inventory, a CSV for verified labels, and example predictions for nine unlabelled images. The training command rejects `unknown` labels. Before using this collection for training, verify capture locations, choose location/session groups and splits, remove duplicates, and add corresponding verified negatives. Incomplete GPS metadata is not a verified location label.

## Files and privacy

- `app.py`, `Launch.cmd`: local desktop application.
- `cli.py`, `kerala_geo/`: inference, training, validation, and evaluation.
- `assets/`: downloaded models, coordinate gallery, boundary, and provenance.
- `cache/`: derived local feature and coordinate indexes; can be rebuilt.
- `data/pilot/`: public pilot photos, manifest, source records, and licences.
- `trained/`: regional classifier and its measured results, once training finishes.
- `tests/`: software checks and actual offline inference test.
- `RESEARCH.md`: papers, design decisions, and limitations.

Prediction uses local files only. Initial setup and the optional public-data collector require internet. The original input photo is never modified. Results are only saved when requested. All bundled images retain their individual licences; see their attribution records.
