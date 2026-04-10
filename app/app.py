import os
import numpy as np
import gradio as gr
from PIL import Image
from inference import segment_coordinates, load_model, CLASSES, PALETTE

# set MODEL_ID in HF Spaces env vars to point at your trained model
DEVICE = "cpu"
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/mit-b0")

model, processor = load_model(MODEL_ID, device=DEVICE)


def run(lat: float, lon: float, zoom: int, alpha: float):
    try:
        original, blended, label_map = segment_coordinates(
            lat=lat,
            lon=lon,
            model=model,
            processor=processor,
            zoom=zoom,
            alpha=alpha,
            device=DEVICE,
        )

        # only show classes that actually appear in this tile
        present = np.unique(label_map)
        legend_lines = []
        for idx in present:
            if idx < len(CLASSES):
                r, g, b = PALETTE[idx]
                swatch = f"<span style='display:inline-block;width:14px;height:14px;background:rgb({r},{g},{b});border-radius:2px;margin-right:6px;vertical-align:middle'></span>"
                legend_lines.append(f"{swatch}{CLASSES[idx]}")
        legend_html = "<b>Classes detected:</b><br>" + "<br>".join(legend_lines)

        return original, blended, legend_html

    except Exception as e:
        raise gr.Error(str(e))


with gr.Blocks(title="satlens", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
        # 🌍 satlens
        Semantic segmentation of aerial map tiles. Enter coordinates to classify land cover pixel-by-pixel.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            lat_input = gr.Number(label="Latitude", value=37.7749, precision=6)
            lon_input = gr.Number(label="Longitude", value=-122.4194, precision=6)
            zoom_input = gr.Slider(
                label="Zoom level", minimum=14, maximum=19, value=17, step=1,
                info="Higher zoom = more detail, smaller area"
            )
            alpha_input = gr.Slider(
                label="Overlay opacity", minimum=0.1, maximum=0.9, value=0.5, step=0.05
            )
            run_btn = gr.Button("Segment", variant="primary")

            gr.Markdown("**Try these:**")
            gr.Examples(
                examples=[
                    [37.7749,  -122.4194, 17, 0.5],
                    [48.8566,    2.3522,  17, 0.5],
                    [35.6762,  139.6503,  17, 0.5],
                    [-33.8688, 151.2093,  17, 0.5],
                    [51.5074,   -0.1278,  17, 0.5],
                ],
                inputs=[lat_input, lon_input, zoom_input, alpha_input],
            )

        with gr.Column(scale=2):
            with gr.Row():
                original_out = gr.Image(label="Original tile", type="pil")
                overlay_out  = gr.Image(label="Segmentation overlay", type="pil")
            legend_out = gr.HTML(label="Legend")

    run_btn.click(
        fn=run,
        inputs=[lat_input, lon_input, zoom_input, alpha_input],
        outputs=[original_out, overlay_out, legend_out],
    )

    gr.Markdown(
        """
        ---
        SegFormer-b0 fine-tuned on [OpenEarthMap](https://open-earth-map.org/) · [GitHub](https://github.com/YOUR_USERNAME/satlens)
        """
    )

demo.launch()
