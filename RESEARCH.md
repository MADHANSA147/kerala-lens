# Research and implementation choices

Research reviewed on 8 October 2026. This project is a local prototype, not a reproduction of every paper's training procedure.

## What the evidence supports

| Source | Relevant finding | Decision for this project |
| --- | --- | --- |
| [GeoCLIP, NeurIPS 2023](https://arxiv.org/html/2309.16020v2) and [official code](https://github.com/VicenteVivan/geo-clip) | Aligns images and GPS embeddings. Its paper reports 14.11% within 1 km and 34.47% within 25 km on Im2GPS3k; these are distance thresholds, not Kerala classification accuracy. It keeps the CLIP backbone frozen and trains projection layers. | Use its published weights for approximate coordinates. Compare frozen CLIP, GeoCLIP, and combined visual features for a new binary classifier. |
| [PIGEON, CVPR 2024](https://arxiv.org/html/2307.05845v3) and [official repository](https://github.com/LukasHaas/PIGEON) | Combines geographic cells, geographic pretraining, and retrieval refinement. The repository does not supply the trained weights, geocells, or datasets needed for a turnkey reproduction. | Use administrative boundaries for the requested output; do not claim to have installed or reproduced PIGEON. |
| [StreetCLIP, official model card](https://huggingface.co/geolocal/StreetCLIP) | Provides street-scene geolocation features, but its domain-specific training dataset excludes India and China. Weights have a noncommercial licence. | Do not assume it is superior for Kerala. It is a candidate for a later controlled comparison, not included in this build. |
| [PLONK, official project](https://github.com/nicolas-dufour/plonk) | Provides generative location distributions and public model variants. | A useful future coordinate-model comparison. It has not been installed or benchmarked here; no superiority claim is made for GeoCLIP. |

## Why there are two outputs

An approximate coordinate can be wrong enough to cross a state boundary. A supervised state classifier can therefore be useful even when the coordinate model is unchanged. The interface treats the classification and coordinate candidates as separate estimates and permits uncertainty. It never translates raw cosine similarity into a confidence percentage.

The pretrained baseline uses the original worldwide coordinate gallery. It does not restrict candidates to Kerala or manufacture a Kerala answer for every image. Its abstention policy requires agreement among five candidates, no candidate within approximately 20 km of Kerala's boundary (including coast), and a candidate spread no greater than 250 km. These are engineering heuristics, not validated statistical confidence intervals.

## What is actually trained

The local training command fits a logistic-regression head on frozen image features. It compares three feature sets and four regularization settings using validation data. The binary decision thresholds are selected from validation data with an empirical 90% precision target and a minimum of five accepted validation examples per class. That target is neither a guarantee nor a calibrated probability. Using a small validation set for both model and threshold selection can be optimistic; held-out performance and coverage must also be reported.

This is transfer learning. It is not training a worldwide geolocation network from scratch, and it does not fine-tune GeoCLIP's coordinate encoder.

## Evaluation safeguards and limits

- Assign entire cities to train, validation, or test before downloading photos. Keep groups separate when adding your own data.
- Reject exact duplicates after decoding pixels, repeated paths, and group overlap. If coordinates are supplied, reject cross-split pairs within 1 km. These checks do not detect every crop, near-duplicate, or shared photographer.
- Select models on validation data only. Never tune on the held-out test results.
- Report the accuracy when forced to choose a class, accuracy among accepted predictions, coverage, class precision, and recall including abstentions. An app that answers only one easy image must not appear to have solved the whole task.
- Public Wikimedia Commons geotags are weak labels. A GPS tag may describe the photographed subject rather than the camera location, or may be wrong. City-level separation reduces overlap for the new classifier but cannot exclude overlap with the upstream pretrained model's data.
- This pilot emphasizes selected South Asian locations. It does not demonstrate worldwide non-Kerala rejection, general indoor accuracy, night-time accuracy, or performance on arbitrary user photos.

## Provenance

Model assets are local, with SHA-256 hashes in `assets/manifest.json`. The CLIP download is pinned to an upstream revision and checked against the upstream file hash. GeoCLIP is pinned to version 1.2.3. The installed environment is recorded separately.

The [geoBoundaries API](https://www.geoboundaries.org/api.html) supplies the boundary; its original source and licence are retained in `assets/boundary-source.json`. The source metadata represents 2011 boundaries and must not be presented as a newly surveyed boundary. Commons photo authors, source pages, licences, and resizing transformations are recorded in `data/pilot/attribution.json`.
