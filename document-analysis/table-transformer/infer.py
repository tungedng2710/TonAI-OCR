import torch
import cv2
import json
from ton_ocr import TonOCRPipeline #pip install ton-ocr
# from huggingface_hub import hf_hub_download
from transformers import pipeline
from PIL import Image


def read_text(cropped_image):
    """
    Applies OCRPipeline to a cropped image (row),
    then returns a single string containing all recognized text.
    """
    results = ocr_pipeline.predict(cropped_image)
    if not results:
        return ""
    combined_text = " ".join(result.text for result in results)
    
    return combined_text


def intersect_boxes(box1, box2):
    """
    Given two boxes in the format {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2},
    return the intersection box in the same format, or None if there's no intersection.
    """
    xmin = max(box1['xmin'], box2['xmin'])
    ymin = max(box1['ymin'], box2['ymin'])
    xmax = min(box1['xmax'], box2['xmax'])
    ymax = min(box1['ymax'], box2['ymax'])

    if xmax > xmin and ymax > ymin:
        return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
    return None

def crop_image(image, box):
    """
    Crop the image based on the bounding box.
    Assumes 'image' is a NumPy array from OpenCV.
    """
    return image[box['ymin']:box['ymax'], box['xmin']:box['xmax']]

def parse_table_to_json(detections, image):
    """
    Takes:
      - detections: a list of dicts (your detection result).
      - image:      a NumPy array (your original image).
    Returns:
      - A dictionary representing the table in row-based JSON structure.
    """

    # 1. Filter out rows and columns
    table_rows = [d for d in detections if d['label'] == 'table row']
    table_cols = [d for d in detections if d['label'] == 'table column']

    # If you need the main table bounding box:
    # tables = [d for d in detections if d['label'] == 'table']
    # (Optional) you might want to crop the main table first.

    # 2. Sort rows by ymin and columns by xmin
    table_rows.sort(key=lambda r: r['box']['ymin'])
    table_cols.sort(key=lambda c: c['box']['xmin'])

    if not table_rows or not table_cols:
        return {}

    # 3. Use the first row as the header to get column names
    header_row = table_rows[0]['box']
    column_names = []
    for col_index, col_detection in enumerate(table_cols):
        col_box = col_detection['box']
        cell_box = intersect_boxes(header_row, col_box)
        if cell_box:
            cropped = crop_image(image, cell_box)
            header_text = read_text(cropped)
            # Fallback column name if OCR is empty
            if not header_text.strip():
                header_text = f"col_{col_index}"
        else:
            header_text = f"col_{col_index}"
        column_names.append(header_text)

    # 4. Now process all remaining rows to build row-based data
    data = {}
    # Start from the second row if the first row is a header
    for row_index, row_detection in enumerate(table_rows[1:], start=0):
        row_box = row_detection['box']
        row_data = {}
        for col_index, col_detection in enumerate(table_cols):
            col_box = col_detection['box']
            cell_box = intersect_boxes(row_box, col_box)
            if cell_box:
                cropped = crop_image(image, cell_box)
                cell_text = read_text(cropped)
            else:
                cell_text = ""

            # Use the header text as the key
            row_data[column_names[col_index]] = cell_text

        data[str(row_index)] = row_data

    return data


if __name__ == "__main__":
    FILE_PATH = "test1.png"
    image = Image.open(FILE_PATH).convert("RGB")

    # Check if CUDA is available, otherwise fall back to CPU
    device = 0 if torch.cuda.is_available() else -1

    # Create the pipeline, specifying the GPU device if available
    pipe = pipeline(
        "object-detection", 
        model="microsoft/table-transformer-structure-recognition", 
        device=device
    )

    # Run inference
    detections = pipe(image)
    ocr_pipeline = TonOCRPipeline()

    detections = detections

    # Load image
    image = cv2.imread(FILE_PATH)

    # Process the table to get row-based JSON data
    table_data = parse_table_to_json(detections, image)

    # Print or save to JSON
    # print(json.dumps(table_data, indent=2))
    with open("ocr_results.json", "w") as f:
        json.dump(table_data, f, indent=2)
