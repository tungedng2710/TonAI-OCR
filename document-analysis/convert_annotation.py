import json
import os
import tqdm

ROOT_DIR = "target"
OUTPUT_DIR = "target_refined"
for filename in os.listdir(ROOT_DIR):
    with open(f"{ROOT_DIR}/{filename}", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get image file name
    image_path = data["task"]["data"]["ocr"]
    image_name = os.path.basename(image_path)

    # We will store mapping of polygon_id -> label type ("Text" or "Cell")
    id_to_label = {}
    for item in data["result"]:
        if item["type"] == "labels":
            polygon_id = item["id"]
            label_name = item["value"]["labels"][0] if item["value"]["labels"] else None
            id_to_label[polygon_id] = label_name

    # Now extract text/cell from textarea type
    text_items = []
    cell_items = []

    for item in data["result"]:
        if item["type"] == "textarea":
            polygon_id = item["id"]
            label_type = id_to_label.get(polygon_id, "")

            entry = {
                "value": item["value"]["text"][0] if item["value"]["text"] else "",
                "point": item["value"]["points"]
            }

            if label_type == "Text":
                text_items.append(entry)
            elif label_type == "Cell":
                cell_items.append(entry)

    # Build output JSON
    output = {
        "image_name": image_name,
        "text": text_items,
        "cell": cell_items
    }

    # Save or print
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = f"{OUTPUT_DIR}/{image_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Simplified JSON saved to {output_path}")