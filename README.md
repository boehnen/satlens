# 🌍 satlens

Pan and zoom a satellite map, hit Segment view, and get a pixel-level land cover overlay — roads, buildings, trees, water, and more.

👉 [Live demo](https://huggingface.co/spaces/boehnen/satlens)

Or run locally:

```bash
git clone https://github.com/boehnen/satlens
cd satlens
pip install -r requirements.txt
python app/app.py
```

## Results

<img width="1156" height="1613" alt="download (1)" src="https://github.com/user-attachments/assets/1715a374-0572-473a-94a3-f26ff8a38000" />
Left: aerial imagery. Center: ground truth labels. Right: model predictions.

Best validation mIoU: 0.5958 after 20 epochs on a free Colab T4.
<img width="1189" height="390" alt="download" src="https://github.com/user-attachments/assets/c3b63f6c-e59f-4db4-b5d5-be0af1231956" />

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
