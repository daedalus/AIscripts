#!/usr/bin/python3

import subprocess
import openai
import os
import argparse

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
    prompt = f"Summarize the following Git diff and generate a brief, concise and meaningful commit message in less than fifty words:\n\n{diff_text}"
    
    try:
        client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")  # Local LLM
        response = client.chat.completions.create(
            model="local-model",  # Not using gpt-4o for local
            messages=[{"role": "system", "content": "You are a helpful assistant skilled in writing Git commit messages."},
                      {"role": "user", "content": prompt}],
            max_tokens=50,
        )
        return response.choices[0].message.content.strip().replace('"','').replace("Commit Message:","")
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
