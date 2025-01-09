import json
import cv2
import numpy as np
from collections import defaultdict
from ton_ocr import TonOCRPipeline #pip install ton-ocr

ocr_pipeline = TonOCRPipeline()
# Suppose you already have your OCR pipeline and read_text() function defined as:
def read_text(cropped_image):
    results = ocr_pipeline.predict(cropped_image)
    if not results:
        return ""
    combined_text = " ".join(result.text for result in results)
    return combined_text
    # return "dummy text"

def convert_ocr_to_row_based_json(
    data,
    original_image_path,
    read_text_fn
):
    """
    data: the loaded JSON data with fields "polygons" and "boxes".
    original_image_path: path to the scanned document image.
    read_text_fn: function that takes a cropped (numpy) image and returns recognized text.

    Returns a dictionary of the form:
    {
      "0": { "Header1": "value", "Header2": "value", ... },
      "1": { "Header1": "value", "Header2": "value", ... },
      ...
    }
    where row 0 is used as a header.
    """

    polygons = data["polygons"]
    boxes = data["boxes"]

    # Load the full image with OpenCV
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)

    # We'll store recognized text for each cell in table[row][col]
    table = defaultdict(lambda: defaultdict(str))

    for i, box in enumerate(boxes):
        start_row, end_row, start_col, end_col = box
        polygon = polygons[i]

        # polygon is [x1, y1, x2, y2, x3, y3, x4, y4] for the cell corners
        # 1) Convert to int
        coords = list(map(int, polygon))

        # 2) Extract bounding rectangle
        xs = coords[0::2]  # even indices: x-coords
        ys = coords[1::2]  # odd indices:  y-coords
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # 3) Crop the region from the original image
        #    Ensure we clamp indices to the image boundaries, 
        #    in case there's any slight out-of-bound indexing.
        h, w = original_image.shape[:2]
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)

        cropped_image = original_image[y_min:y_max, x_min:x_max]

        # 4) Run OCR on the cropped image
        recognized_text = read_text_fn(cropped_image)

        # Fill the recognized text into table[r][c]
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                table[r][c] = recognized_text

    # The first row is the header
    header_cols = sorted(table[0].keys())
    header_names = [table[0][col] for col in header_cols]

    # Build the final row-based JSON
    row_based_data = {}
    max_row = max(table.keys()) if table else 0

    output_row_index = 0
    for r in range(1, max_row + 1):
        row_dict = {}
        for idx, col in enumerate(header_cols):
            col_name = header_names[idx]
            # If the cell is missing, store empty string
            row_dict[col_name] = table[r].get(col, "")
        row_based_data[str(output_row_index)] = row_dict
        output_row_index += 1

    return row_based_data

if __name__ == "__main__":
    # 1. Load the OCR results (polygons & boxes) from JSON
    with open("output.json", "r", encoding="utf-8") as f:
        data_example = json.load(f)

    # 2. Define the path to the scanned document image
    image_path = "input.jpg"  # or .jpg, etc.

    # 3. Convert the OCR data to row-based JSON using your read_text() function
    #    (assuming you have defined read_text(cropped_image) somewhere)
    result = convert_ocr_to_row_based_json(
        data_example,
        original_image_path=image_path,
        read_text_fn=read_text  # Pass your function here
    )

    # 4. Print the result
    print(json.dumps(result, indent=2, ensure_ascii=False))
    with open("ocr_results.json", "w") as f:
        json.dump(result, f, indent=2)