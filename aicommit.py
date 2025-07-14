#!/usr/bin/python3



import os
import re
import openai
import argparse
import subprocess


BASE_URL = "https://api.groq.com/openai/v1"
#MODEL="meta-llama/llama-4-maverick-17b-128e-instruct"
#MODEL="meta-llama/llama-4-scout-17b-16e-instruct"
#MODEL="meta-llama/llama-guard-4-12b"
#MODEL="llama3-70b-8192"
MODEL="llama3-8b-8192"
#MODEL="llama-3.3-70b-versatile"
#MODEL="gemma2-9b-it"
#MODEL="qwen/qwen3-32b"
#MODEL="qwen-qwq-32b"
#MODEL="llama-3.1-8b-instant"
MODEL="deepseek-r1-distill-llama-70b"

def get_git_diff(filename=None):
    """Gets the Git diff of unstaged and staged changes."""
    try:
        if filename:
            diff = subprocess.check_output(["git", "diff", filename], text=True)
        else:
            diff = subprocess.check_output(["git", "diff"], text=True)
        if not diff.strip():
            return None  # No changes detected
        return diff
    except subprocess.CalledProcessError as e:
        print(f"Error getting Git diff: {e}")
        return None

def generate_commit_message(diff_text):
    """Generates a commit message using OpenAI's GPT."""
    prompt = f"Summarize the following Git diff as a commit message. Output only the commit message, nothing else.\n\n{diff_text}"    
    try:
        client = openai.OpenAI(base_url=BASE_URL, api_key=os.environ['API_KEY'])  # Local LLM
        response = client.chat.completions.create(
            model=MODEL,  # Not using gpt-4o for local
            messages=[{"role": "system", "content": "You are a helpful assistant skilled in writing Git commit messages."},
                      {"role": "user", "content": prompt}],
            max_tokens=2000
        )
        msg = response.choices[0].message.content.strip()
        msg = cleaned = re.sub(r'<think>.*?</think>', '', msg, flags=re.DOTALL)
        return msg
    except Exception as e:
        print(f"Error generating commit message: {e}")
        return None

def apply_commit(commit_message, filename=None):
    """Stages and commits the changes with the generated message."""
    try:
        if filename:
            subprocess.run(["git", "add", filename], check=True)
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
        else:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print(f"Committed successfully: {commit_message}")
    except subprocess.CalledProcessError as e:
        print(f"Error committing changes: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Git commit messages using AI.")
    parser.add_argument("filename", nargs="?", help="Specific file to commit")
    args = parser.parse_args()
    
    if not (diff_text := get_git_diff(args.filename)):
        print("No changes detected.")
        return
    
    if not (commit_message := generate_commit_message(diff_text)):
        print("Failed to generate a commit message.")
        return
    
    apply_commit(commit_message, args.filename)

if __name__ == "__main__":
    main()
