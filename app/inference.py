import math
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

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


def _lat_lon_to_tile(lat, lon, zoom):
    # Slippy map tile math: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_lat_lon(x, y, zoom):
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def fetch_tile(x, y, zoom):
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    headers = {"User-Agent": "satlens/1.0 (github.com/boehnen/satlens)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def fetch_viewport(bounds, zoom):
    """
    Fetch and stitch all tiles covering the given lat/lon bounds.
    bounds: (south, west, north, east)
    Returns (stitched_image, tile_x_min, tile_y_min, tile_x_max, tile_y_max)
    """
    south, west, north, east = bounds

    x_min, y_max = _lat_lon_to_tile(south, west, zoom)
    x_max, y_min = _lat_lon_to_tile(north, east, zoom)

    x_count = x_max - x_min + 1
    y_count = y_max - y_min + 1

    tile_size = 256
    canvas = Image.new("RGB", (x_count * tile_size, y_count * tile_size))

    for row, ty in enumerate(range(y_min, y_max + 1)):
        for col, tx in enumerate(range(x_min, x_max + 1)):
            tile = fetch_tile(tx, ty, zoom)
            canvas.paste(tile, (col * tile_size, row * tile_size))

    return canvas, x_min, y_min, x_max, y_max


def load_model(model_path, device="cpu"):
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.eval()
    model.to(device)
    return model, processor


def predict(image, model, processor, device="cpu"):
    # resize to 512 for inference, remember original size for upscaling back
    orig_size = image.size
    resized = image.resize((512, 512), Image.BILINEAR)
    inputs = processor(images=resized, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    label_map_512 = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

    # scale label map back to original canvas size
    label_img = Image.fromarray(label_map_512.astype(np.uint8))
    label_img = label_img.resize(orig_size, Image.NEAREST)
    return np.array(label_img)


def colorize(label_map):
    return Image.fromarray(PALETTE[label_map])


def overlay(original, label_map, alpha=0.5):
    color_mask = colorize(label_map).resize(original.size, Image.NEAREST)
    return Image.blend(original, color_mask, alpha=alpha)


def segment_viewport(bounds, zoom, model, processor, alpha=0.5, device="cpu"):
    """
    Full pipeline for viewport segmentation.
    Returns (overlay_image, label_map, tile_bounds_dict)
    tile_bounds_dict has the lat/lon corners so the frontend can position the overlay.
    """
    canvas, x_min, y_min, x_max, y_max = fetch_viewport(bounds, zoom)
    label_map = predict(canvas, model, processor, device)
    blended = overlay(canvas, label_map, alpha)

    # compute lat/lon bounds of the stitched tile grid
    north, west = _tile_to_lat_lon(x_min, y_min, zoom)
    south, east = _tile_to_lat_lon(x_max + 1, y_max + 1, zoom)

    tile_bounds = {"north": north, "south": south, "west": west, "east": east}
    return blended, label_map, tile_bounds
