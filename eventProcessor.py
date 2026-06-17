import json
import os

def transform_blocks(blocks):
    transformed = []
    for block in blocks:
        new_block = dict(block)

        if isinstance(new_block.get("time"), list):
            new_block["time"] = new_block["time"]

        entities = new_block.get("entities")
        if isinstance(entities, list) and entities:
            new_block["entities"] = entities[0]
            new_block["post_entities"] = entities[1:]
        elif isinstance(entities, str):
            new_block["post_entities"] = new_block.get("post_entities", [])
        else:
            new_block["entities"] = ""
            new_block["post_entities"] = []

        transformed.append(new_block)
    return transformed

def process_event_data(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = transform_blocks(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f"Formatted data saved to: {output_file}")
