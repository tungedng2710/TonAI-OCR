import cv2
import numpy as np
from ton_ocr import TonOCRPipeline

image_path = "stuffs/example.jpg"
image = cv2.imread(image_path)
ocr = TonOCRPipeline()
results = ocr.predict(image)

for result in results:
    bbox = result.box           # text bounding box
    text = result.text          # text string
    score = result.score        # OCR's confidence
    img = result.img            # cropped text image