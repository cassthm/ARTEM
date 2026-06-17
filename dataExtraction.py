import os
import re
import ast
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# === Setup paths ===
base_path = "bookname"
books_dir = os.path.join(base_path, "book2")
output_dir = os.path.join(base_path, "book2")
os.makedirs(output_dir, exist_ok=True)

# === Initialize LLM pipeline ===
model_path = "LLM_model_path"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.1
)

# === Prompt builder ===
def event_extraction_prompt(chapter_text, chapter_num):
    return f"""
You are an information extraction agent.

Extract structured information from **only chapter {chapter_num}** below. Return a **single dictionary** with the following keys:

{{
  "chapter": {chapter_num},
  "date": ["..."],
  "location": "...",
  "entity": ["..."],
  "content": "..."
}}

The content should be a SINGLE SHORT sentence that summarises the book chapter. For example, content could be:
"plasma conduit rupture at the lunar station"
"containment failure in the reactor core"

Return ONLY the dictionary. Do not wrap in a code block. Do not output multiple dictionaries.

Chapter {chapter_num}:
\"\"\"{chapter_text}\"\"\"
"""

# === Extraction logic ===
def extract_info(chapter_text, chapter_number):
    prompt = event_extraction_prompt(chapter_text, chapter_number)
    response = llm_pipe(prompt, return_full_text=False)[0]['generated_text']

    dict_candidates = re.findall(r"\{.*?\}", response, re.DOTALL)
    for raw_dict in dict_candidates:
        try:
            cleaned = raw_dict.replace("'", '"')
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(raw_dict)

            return {
                "chapter": chapter_number,
                "time": parsed.get("date", []),
                "spaces": parsed.get("location", []),
                "entities": parsed.get("entity", []),
                "content": parsed.get("content", [])
            }
        except Exception:
            continue

    return {
        "chapter": chapter_number,
        "time": [],
        "spaces": [],
        "entities": [],
        "content": []
    }

# === Process the single book directory ===
book_tag = "book1"
output_book_dir = os.path.join(output_dir, book_tag)
os.makedirs(output_book_dir, exist_ok=True)

print(f"Processing book1 directory")

# === Check for book.json file ===
book_file = os.path.join(books_dir, "book.json")

if not os.path.exists(book_file):
    print("book.json not found. Looking for other book files...")
    json_files = [f for f in os.listdir(books_dir) if f.endswith('.json') and 'book' in f.lower()]
    if json_files:
        book_file = os.path.join(books_dir, json_files[0])
        print(f"Using {json_files[0]} as book file")
    else:
        raise FileNotFoundError(f"No book.json or similar file found in {books_dir}")

# === Load and extract chapters ===
with open(book_file, "r", encoding="utf-8") as f:
    raw_content = f.read()

try:
    book_data = json.loads(raw_content)
    if isinstance(book_data, dict):
        raw_text = book_data.get('content', '') or book_data.get('text', '') or str(book_data)
    elif isinstance(book_data, list):
        raw_text = '\n'.join([str(item) for item in book_data])
    else:
        raw_text = str(book_data)
except json.JSONDecodeError:
    raw_text = raw_content

# === Extract chapters ===
split_chapters = re.findall(r"(Chapter\s+(\d+))(.*?)(?=Chapter\s+\d+|$)", raw_text, re.DOTALL)
chapters = [{"chapter": int(ch_num), "text": ch_text.strip()} for _, ch_num, ch_text in split_chapters]

if not chapters:
    print("No chapters found using 'Chapter X' pattern. Checking for alternative patterns...")
    alt_patterns = [
        r"(CHAPTER\s+(\d+))(.*?)(?=CHAPTER\s+\d+|$)",
        r"(\d+\.\s*)(.*?)(?=\d+\.\s*|$)",
        r"(Chapter\s+([IVX]+))(.*?)(?=Chapter\s+[IVX]+|$)"
    ]

    for pattern in alt_patterns:
        split_chapters = re.findall(pattern, raw_text, re.DOTALL)
        if split_chapters:
            chapters = [{"chapter": i+1, "text": ch_text.strip()} for i, (_, _, ch_text) in enumerate(split_chapters)]
            print(f"Found {len(chapters)} chapters using alternative pattern")
            break

if not chapters:
    print("No chapters found. Processing entire content as single chapter...")
    chapters = [{"chapter": 1, "text": raw_text.strip()}]

print(f"Found {len(chapters)} chapters to process")

# === Extract information from chapters ===
results = []
for ch in chapters:
    print(f"Processing Chapter {ch['chapter']}")
    results.append(extract_info(ch["text"], ch["chapter"]))

# === Save extracted features ===
df = pd.DataFrame(results)
df.to_json(os.path.join(output_book_dir, f"extracted_features_{book_tag}.json"), orient="records", indent=2)

# === Save processing summary ===
summary_path = os.path.join(output_dir, "processing_summary.txt")
with open(summary_path, "w") as f:
    f.write("Book Processing Summary:\n\n")
    f.write(f"Source directory: {books_dir}\n")
    f.write(f"Output directory: {output_book_dir}\n")
    f.write(f"Chapters processed: {len(chapters)}\n")
    f.write(f"Features extracted: {len(results)}\n")

print(f"Processing complete!")
print(f"Results saved to: {output_book_dir}")
print(f"Processed {len(chapters)} chapters")
print(f"Summary saved to: {summary_path}")
