import json
import os

# Load your JSON (replace with open(file) if needed)
with open("input.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract file name from image path
image_name = os.path.basename(data["task"]["data"]["ocr"])

simplified = {"image_name": image_name}
temp_store = {}

for item in data["result"]:
    obj_id = item["id"]
    if obj_id not in temp_store:
        temp_store[obj_id] = {"point": None, "label": None, "value": None}
    
    if item["type"] == "labels":
        temp_store[obj_id]["label"] = item["value"]["labels"][0]
        temp_store[obj_id]["point"] = item["value"]["points"]
    elif item["type"] == "textarea":
        temp_store[obj_id]["value"] = item["value"]["text"][0]
        temp_store[obj_id]["point"] = item["value"]["points"]

simplified.update(temp_store)

# Save to file
output_path = f"{simplified['image_name']}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(simplified, f, indent=2, ensure_ascii=False)

print(f"Simplified JSON saved to {output_path}")