# SatLens

**Computer vision · semantic segmentation · geospatial ML**

Fine-tuned SegFormer-b0 on satellite imagery for pixel-level land cover classification. Pan and zoom a satellite map, hit Segment view, and get a georeferenced color overlay in real time.

👉 **[Live demo](https://huggingface.co/spaces/boehnen/satlens)** · [Model weights](https://huggingface.co/boehnen/satlens-segformer)

---

## Model

- Architecture: SegFormer-b0 (3.7M params, Mix Transformer encoder + MLP decoder)
- Dataset: OpenEarthMap, 2,687 aerial tiles, 8-class OSM-derived labels, 0.25–0.5m GSD, 44 countries
- Training: 20 epochs, cosine LR, random flip augmentation, free Colab T4
- Val mIoU: **0.596**

## Inference

At runtime the app computes which tile coordinates cover the current map viewport, fetches and stitches them into a canvas, runs the model, and returns a georeferenced PNG overlay positioned on the map via Leaflet's `imageOverlay`.

The live demo uses ESRI satellite tiles, which differ in sensor and resolution from the training data. Performance on the demo reflects real domain shift rather than validation accuracy.

## Stack

- SegFormer-b0 via HuggingFace Transformers
- PyTorch, OpenEarthMap, Gradio, Leaflet.js, ESRI World Imagery

## Run locally

```bash
git clone https://github.com/boehnen/satlens
cd satlens
pip install -r requirements.txt
python app/app.py
```

## Train

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boehnen/satlens/blob/main/notebooks/train.ipynb)

---

Built with [OpenEarthMap](https://open-earth-map.org/).
