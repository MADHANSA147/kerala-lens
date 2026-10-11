# Third-party attribution

## Models and code

- [GeoCLIP](https://github.com/VicenteVivan/geo-clip), Vicente Vivanco Cepeda, Gaurav Kumar Nayak, Mubarak Shah. NeurIPS 2023. The installed distribution includes its upstream licence. The pretrained image-projection and location-encoder weights are used unchanged.
- [CLIP](https://github.com/openai/CLIP), OpenAI. The ViT-L/14 checkpoint is downloaded from [the official model repository](https://huggingface.co/openai/clip-vit-large-patch14), revision `32bd64288804d66eefd0ccbe215aa642df71cc41`. CLIP's upstream code is MIT licensed.
- Python packages retain their own licences in the installed distributions. See `environment-tested.txt` for the tested versions.

## Kerala boundary

Boundary downloaded through [geoBoundaries](https://www.geoboundaries.org/), sourced from the DataMeet India community and Election Commission of India. The upstream metadata identifies CC BY 2.5 India and a represented year of 2011. The Kerala feature was extracted from the full India ADM1 collection without simplifying its coordinates. Full provenance is saved in `assets/boundary-source.json`.

## Public photos

Every collected photo has its own author, source page, and licence in `data/pilot/attribution.json`. Downloaded thumbnail images were converted to JPEG. The licence of each image continues to apply; this project does not relicense the collection under a single code licence. Keep attribution and applicable share-alike notices with redistributed images.
