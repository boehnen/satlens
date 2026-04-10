import math
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# OpenEarthMap uses 8 classes + background, index 0 is unlabeled
CLASSES = [
    "Background",
    "Bareland",
    "Rangeland",
    "Developed space",
    "Road",
    "Tree",
    "Water",
    "Agriculture land",
    "Building",
]

PALETTE = np.array([
    [0,   0,   0],
    [128, 0,   0],
    [0,   128, 0],
    [128, 128, 0],
    [255, 255, 0],
    [0,   64,  0],
    [0,   0,   255],
    [0,   255, 128],
    [255, 0,   0],
], dtype=np.uint8)


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    # Slippy map tile math: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def fetch_tile(lat: float, lon: float, zoom: int = 17) -> Image.Image:
    x, y = _lat_lon_to_tile(lat, lon, zoom)
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    headers = {"User-Agent": "satlens/1.0 (github.com/YOUR_USERNAME/satlens)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    # tiles come in at 256x256, upsample so the model has more to work with
    img = img.resize((512, 512), Image.BILINEAR)
    return img


def load_model(model_path: str, device: str = "cpu"):
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.eval()
    model.to(device)
    return model, processor


def predict(
    image: Image.Image,
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    device: str = "cpu",
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # logits come out at H/4 x W/4, interpolate back up
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    label_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
    return label_map


def colorize(label_map: np.ndarray) -> Image.Image:
    return Image.fromarray(PALETTE[label_map])


def overlay(
    original: Image.Image,
    label_map: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    color_mask = colorize(label_map).resize(original.size, Image.NEAREST)
    return Image.blend(original, color_mask, alpha=alpha)


def segment_coordinates(
    lat: float,
    lon: float,
    model,
    processor,
    zoom: int = 17,
    alpha: float = 0.5,
    device: str = "cpu",
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    tile = fetch_tile(lat, lon, zoom)
    label_map = predict(tile, model, processor, device)
    blended = overlay(tile, label_map, alpha)
    return tile, blended, label_map
