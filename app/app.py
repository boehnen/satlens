import os
import io
import json
import base64
import numpy as np
import gradio as gr
from PIL import Image
from inference import segment_viewport, load_model, CLASSES, PALETTE

DEVICE = "cpu"
MODEL_ID = os.environ.get("MODEL_ID", "boehnen/satlens-segformer")

model, processor = load_model(MODEL_ID, device=DEVICE)


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def segment(bounds_json: str, zoom: str, alpha: str) -> str:
    try:
        bounds = json.loads(bounds_json)
        blended, label_map, tile_bounds = segment_viewport(
            bounds=(bounds["south"], bounds["west"], bounds["north"], bounds["east"]),
            zoom=int(float(zoom)),
            model=model,
            processor=processor,
            alpha=float(alpha),
            device=DEVICE,
        )
        img_b64 = pil_to_base64(blended)
        present = np.unique(label_map)
        legend_lines = []
        for idx in present:
            if idx < len(CLASSES) and idx != 0:
                r, g, b = PALETTE[idx]
                swatch = f"<span style='display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});border-radius:2px;margin-right:6px;vertical-align:middle'></span>"
                legend_lines.append(f"{swatch}{CLASSES[idx]}")
        legend_html = "<b>Detected:</b><br>" + "<br>".join(legend_lines)
        return json.dumps({"img": img_b64, "bounds": tile_bounds, "legend": legend_html})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Something went wrong: {e}"})


MAP_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script type="module">
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";
window._gradioClient = Client;
</script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; font-family: sans-serif; background: #111; color: #eee; }
#map { height: calc(100% - 44px); width: 100%; position: relative; }
#controls {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 12px; background: #1a1a1a; flex-wrap: wrap; height: 44px;
}
#segment-btn {
  padding: 6px 16px; border-radius: 6px; border: none;
  background: #2563eb; color: white; font-size: 13px;
  cursor: pointer; font-weight: 500;
}
#segment-btn:hover { background: #1d4ed8; }
#segment-btn:disabled { background: #4b6cb7; cursor: not-allowed; }
#status { font-size: 12px; color: #9ca3af; }
#legend { position: absolute; bottom: 30px; right: 8px; z-index: 1000; background: rgba(20,20,20,0.85); padding: 8px 10px; font-size: 12px; line-height: 1.9; border-radius: 6px; pointer-events: none; display: none; }
label { font-size: 12px; color: #d1d5db; white-space: nowrap; }
</style>
</head>
<body>
<div id="map"><div id="legend"></div></div>
<div id="controls">
  <button id="segment-btn" onclick="runSegment()">Segment view</button>
  <label>Opacity
    <input type="range" id="alpha" min="0.1" max="0.9" step="0.05" value="0.5"
      oninput="document.getElementById('alpha-val').textContent=parseFloat(this.value).toFixed(2);updateOpacity()"/>
  </label>
  <span id="alpha-val" style="font-size:12px;font-weight:500;min-width:28px">0.50</span>
  <span id="status">Pan and zoom, then hit Segment view</span>
</div>
<script>
const map = L.map('map', { minZoom: 15, maxZoom: 18 }).setView([37.7749, -122.4194], 16);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Tiles &copy; Esri', maxZoom: 18
}).addTo(map);
setTimeout(() => map.invalidateSize(), 300);
let overlay = null;
function updateOpacity() {
  if (overlay) overlay.setOpacity(parseFloat(document.getElementById('alpha').value));
}
function lock()   { map.dragging.disable(); map.scrollWheelZoom.disable(); map.doubleClickZoom.disable(); map.touchZoom.disable(); }
function unlock() { map.dragging.enable();  map.scrollWheelZoom.enable();  map.doubleClickZoom.enable();  map.touchZoom.enable(); }
async function runSegment() {
  const btn    = document.getElementById('segment-btn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = 'Segmenting...';
  lock();
  const b = map.getBounds();
  const boundsJSON = JSON.stringify({
    south: b.getSouth(), west: b.getWest(),
    north: b.getNorth(), east: b.getEast()
  });
  const zoom  = map.getZoom();
  const alpha = parseFloat(document.getElementById('alpha').value);
  try {
    // wait for gradio client to load
    let attempts = 0;
    while (!window._gradioClient && attempts++ < 20) {
      await new Promise(r => setTimeout(r, 200));
    }
    if (!window._gradioClient) throw new Error('Gradio client not loaded');
    const client = await window._gradioClient.connect(window.location.origin);
    const result_raw = await client.predict('/predict', {
      bounds_json: boundsJSON,
      zoom: String(zoom),
      alpha: String(alpha)
    });
    const result = JSON.parse(result_raw.data[0]);
    if (result.error) {
      status.textContent = 'Error: ' + result.error;
    } else {
      if (overlay) map.removeLayer(overlay);
      const bds = result.bounds;
      overlay = L.imageOverlay(
        'data:image/png;base64,' + result.img,
        [[bds.south, bds.west], [bds.north, bds.east]],
        { opacity: alpha, interactive: false }
      ).addTo(map);
      const leg = document.getElementById('legend'); leg.innerHTML = result.legend || ''; leg.style.display = result.legend ? 'block' : 'none';
      status.textContent = 'Done';
    }
  } catch(e) {
    status.textContent = 'Failed: ' + e.message;
  }
  btn.disabled = false;
  unlock();
}
</script>
</body>
</html>"""

with open("/app/map.html", "w") as f:
    f.write(MAP_HTML)

OUTER_HTML = """
<iframe
  src="/gradio_api/file=/app/map.html"
  style="width:100%;height:700px;border:none;border-radius:8px">
</iframe>
"""

with gr.Blocks(title="SatLens") as demo:
    gr.Markdown("# SatLens\nPan and zoom the map, then hit **Segment view** to classify land cover.")
    gr.HTML(OUTER_HTML)

    with gr.Row(visible=False):
        b_in  = gr.Textbox()
        z_in  = gr.Textbox()
        a_in  = gr.Textbox()
        r_out = gr.Textbox()
        btn   = gr.Button()

    btn.click(fn=segment, inputs=[b_in, z_in, a_in], outputs=[r_out], api_name="predict")

demo.launch(allowed_paths=["/app/map.html"], ssr_mode=False)
