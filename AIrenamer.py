import time
import os
import shutil
import argparse
import json
from pathlib import Path
import requests

# Send request to the specified LLM endpoint
def get_llm_response(code: str, endpoint_url: str) -> dict:
    prompt = f"""
You are a code classifier and file renamer.

Given the following Python script, suggest:
1. A short and descriptive new filename (without extension) based on what the script does and how complex.
2. Whether it is 'interesting' or 'uninteresting'.

Respond in JSON format:
{{
  "name": "new_filename",
  "interest": "interesting" or "uninteresting"
}}

Python code:
\"\"\"
{code}
\"\"\"
"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    }

    try:
        t0 = time.time()
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=3600)
        response.raise_for_status()
        rtime = time.time() - t0
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        tokens = data["usage"]["completion_tokens"]
        print(f"Stats: Tokens: {tokens}, time: {round(rtime,2)} sec, tokens/s: {round(tokens/rtime,2)}")
        return json.loads(reply)
    except Exception as e:
        print("Failed to get or parse LLM response:", e)
        return {"name": "unnamed_script", "interest": "uninteresting"}

# Process all .py files
def process_directory(base_dir: Path, llm_url: str):
    high_interest_dir = base_dir / "high_interest"
    low_interest_dir = base_dir / "low_interest"
    high_interest_dir.mkdir(exist_ok=True)
    low_interest_dir.mkdir(exist_ok=True)

    for root, dirs, files in os.walk(base_dir):
        if high_interest_dir in Path(root).parents or low_interest_dir in Path(root).parents:
            continue

        for filename in files:
            if filename.endswith(".py") or filename.endswith(".sage"):
                file_path = Path(root) / filename
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

                suffix = file_path.suffix
                if suffix == ".py":
                    ftype = "python"
                else:
                    ftype = "sage"
                print(f"Processing {ftype} script: {file_path}")
                
                result = get_llm_response(code, llm_url)

                new_name = result.get("name", "unnamed_script").strip().replace(" ", "_") + suffix
                interest = result.get("interest", "uninteresting").lower()

                target_dir = high_interest_dir if interest == "interesting" else low_interest_dir
                target_path = target_dir / new_name

                # Avoid collisions
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{new_name[:-3]}_{counter}.py"
                    counter += 1

                try:
                    shutil.move(str(file_path), target_path)
                    print(f"Moved to: {target_path}")
                except Exception as e:
                    print(f"Failed to move {file_path} to {target_path}: {e}")

# CLI
def main():
    parser = argparse.ArgumentParser(description="Rename and classify Python scripts using an LLM.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing Python files.")
    parser.add_argument("--llm-url", type=str, required=True, help="URL of the LLM endpoint (e.g. http://localhost:11434/v1/chat)")
    args = parser.parse_args()

    base_dir = Path(args.directory).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: '{base_dir}' is not a valid directory.")
        return

    process_directory(base_dir, args.llm_url)

if __name__ == "__main__":
    main()

