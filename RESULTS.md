# Kerala Lens: measured results and delivery status

8 October 2026

**A local model and app have been built and trained, but the requested high-accuracy goal is not yet established.** The software works offline; the regional model improves on the pretrained coordinate baseline in this small test. It still makes substantial mistakes when required to classify every photo.

## Held-out performance

All 53 test images came from towns excluded from fitting and model selection. The test towns were Kasaragod, Pathanamthitta, Thiruvananthapuram, Nagercoil, Tirunelveli, and Shivamogga.

| Method | Correct / total | Accuracy | Coverage |
| --- | --- | --- | --- |
| GeoCLIP top coordinate checked against Kerala boundary | 32 / 53 | 60.4% | 100% |
| Trained regional classifier, forced binary guess | 39 / 53 | 73.6% | 100% |
| Conservative regional decisions | 4 correct, 0 wrong, 49 uncertain | Only 4 answered cases; insufficient evidence of reliability | 7.5% |

Do not describe the conservative result as “100% accurate.” It answered very few images, all in one class. The validation data did not support the requested 90% empirical precision threshold for outside-Kerala decisions, so the conservative classifier abstains on that side. This is a material limitation, not a successful production model.

The forced binary classifier correctly identified 19/25 Kerala images and 20/28 non-Kerala images. These are small, clustered samples, so they do not justify a narrow confidence interval or a claim about arbitrary future photos.

GeoCLIP coordinate estimates had a median error of approximately **477.9 km**, with **0/53 within 25 km** on this pilot. The coordinate component is therefore unsuitable for precise location verification here. Classification and coordinate estimation are separate outputs.

## Training data and method

- 284 reviewed scene photos from 26 towns: 168 training, 63 validation, 53 test.
- Training: 84 Kerala and 84 non-Kerala photos. Validation: 33 and 30. Test: 25 and 28.
- Photos were collected from Wikimedia Commons with individual author/licence/source records. The original collection contained 416 photos, with 59 additional unique validation photos subsequently collected.
- Before model evaluation, an assistant reviewed contact sheets and excluded portraits, isolated objects, food, plant/animal close-ups, documents, and unsuitable airplane-window imagery. Review did not independently verify geographic ground truth.
- Twelve configurations compared frozen CLIP features, GeoCLIP features, and their combination across four regularization strengths. Validation data selected CLIP features with a logistic-regression head, C=100.
- The original GeoCLIP coordinate weights were not fine-tuned. There is no claim that a new worldwide foundation model was pretrained.

Exact metrics, candidate-selection results, thresholds, per-town results, and dataset hashes are in `trained/report.json`. Per-image test outcomes are in `trained/test-predictions.csv`. The fitted head is `trained/classifier.json`.

## Your images

The three supplied ZIP files contained **355 readable images**: 57 in the women archive, 57 in the men archive, and 241 in the objects/images archive. They have no verified capture-location labels. Four images contain GPS-related metadata fields, but none contains the complete latitude/longitude tag set needed to identify a location.

All were imported and inventoried locally. They were not assigned geographic labels from clothing or object styles. Nine representative files were used for an unlabelled inference check; those predictions are not accuracy evidence.

## Software verification

- 13 automated tests cover image handling, EXIF orientation, boundary coordinate order, holes, invalid coordinates, abstention, metrics, duplicate pixels, group leakage, and cross-split geographic proximity.
- Asset SHA-256 verification passed, as did known-coordinate boundary checks.
- Real inference completed with Python socket connections and DNS resolution blocked, with zero observed network attempts. Repeated predictions were identical.
- The cached coordinate calculation matched GeoCLIP's official forward method, and a saved held-out classifier prediction matched the live app calculation.
- A hidden desktop-interface smoke test exercised result and error rendering. A full manual visual usability audit was not performed.
- The final local-model test with its prepared index took approximately 16.0 seconds to load and predict, and 0.9 seconds for a warm repeated prediction. These are individual CPU measurements, not guaranteed timings.

The machine has approximately 8 GB RAM and a CPU-only PyTorch installation. The image-only runtime releases the unused CLIP text branch to reduce memory use. Internet was used for initial public downloads and data collection; prediction uses local assets.

## What is needed to establish good accuracy

The main missing resource is a substantially larger, representative dataset with verified capture locations: ordinary roads, houses, rural scenes, urban scenes, interiors, different seasons, and hard negatives from neighbouring states and other similar regions. Keep entirely new locations and photographers for final evaluation. Public landmarks, cultural objects, and scraped clothing photos cannot substitute for those labels.

The current project supplies the model, app, data preparation, reproducible training, and evaluation workflow for that next iteration. Neither research literature nor passing software tests makes an image-geolocation model error-free.
