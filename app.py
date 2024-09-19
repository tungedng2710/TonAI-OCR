import cv2
import numpy as np
import gradio as gr
import warnings
from ton_ocr import TonOCRPipeline

# Suppress all warnings
warnings.filterwarnings("ignore")

def clear_image():
        return None, None

def ocr_image(image):
    ocr = TonOCRPipeline()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = ocr.predict(image)

    for result in results:
        bbox = result.box           # text bounding box
        text = result.text          # text string
        score = result.score        # OCR's confidence
        img = result.img            # cropped text image

        # Draw the bounding polygon
        points = np.array(bbox, np.int32)
        points = points.reshape((-1, 1, 2))
        color = (0, 255, 255)
        is_closed = True
        thickness = 2
        cv2.polylines(image, [points], is_closed, color, thickness)

        # Add OCR text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_color = (0, 0, 255)
        text_thickness = 2
        first_point = tuple(points[0][0])
        cv2.putText(image, text, first_point, font, font_scale, text_color, text_thickness)

    # Convert image from BGR to RGB for Gradio
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Create the Gradio Blocks interface
with gr.Blocks(theme=gr.Theme.from_hub("ParityError/Interstellar")) as demo:
    gr.Markdown("# TonAI OCR - Nhận diện chữ trong ảnh")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(sources = ['upload', 'clipboard'], 
                                   label="Input Image")
            with gr.Row():
                with gr.Column():
                    submit_button = gr.Button("Submit")
                with gr.Column():
                    clear_button = gr.Button("Clear")
        with gr.Column():
            image_output = gr.Image(type="numpy", label="Output Image")
            submit_button.click(fn=ocr_image, inputs=image_input, outputs=image_output)
            clear_button.click(fn=clear_image, outputs=[image_input, image_output])

# Launch the interface
demo.launch(server_name="0.0.0.0", 
            server_port=7862,
            favicon_path="stuffs/favicon.png",
            max_threads=99)