# 🌍 satlens

Semantic segmentation of map tiles using a fine-tuned SegFormer model trained on OpenEarthMap data.

Given a latitude/longitude, **satlens** fetches the aerial tile and overlays a pixel-level land cover classification — roads, buildings, forests, water, farmland, and more.

![demo placeholder](assets/demo.png)

---

## What it does

- Fetches a 512×512 aerial tile for any coordinates
- Runs a fine-tuned [SegFormer-b0](https://huggingface.co/nvidia/mit-b0) model on the tile
- Returns a colorized segmentation overlay showing land cover classes

## Land cover classes

| Color | Class |
|-------|-------|
| 🟥 | Bareland |
| 🟫 | Rangeland |
| 🟩 | Developed space |
| 🟦 | Road |
| ⬜ | Tree |
| 🟨 | Water |
| 🟪 | Agriculture |
| ⬛ | Building |

## Try it

👉 **[Live demo on Hugging Face Spaces](https://huggingface.co/spaces/boehnen/satlens)**

Or run locally:

```bash
git clone https://github.com/YOUR_USERNAME/satlens
cd satlens
pip install -r requirements.txt
python app/app.py
```

## Train it yourself

Open `notebooks/train.ipynb` in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boehnen/satlens/blob/main/notebooks/train.ipynb)

Training takes ~45 minutes on a free Colab T4 GPU.

## Stack

- [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) — transformer-based semantic segmentation
- [OpenEarthMap](https://zenodo.org/records/7223446) — aerial imagery + OSM-derived labels
- [Gradio](https://gradio.app/) — demo UI
- [Hugging Face Spaces](https://huggingface.co/spaces) — free hosting

## Project structure

```
satlens/
├── notebooks/
│   └── train.ipynb         # Full training pipeline (run in Colab)
├── app/
│   ├── app.py              # Gradio demo
│   └── inference.py        # Tile fetching + model inference
├── requirements.txt
└── README.md
```

---

Built with [OpenEarthMap](https://open-earth-map.org/) dataset.
