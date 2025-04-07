#!/usr/bin/python3

import subprocess
import openai
import os

# Set your OpenAI API key (or use an environment variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_git_diff():
    """Gets the Git diff of unstaged and staged changes."""
    try:
        diff = subprocess.check_output(["git", "diff"], text=True)
        if not diff.strip():
            return None  # No changes detected
        return diff
    except subprocess.CalledProcessError as e:
        print(f"Error getting Git diff: {e}")
        return None

def generate_commit_message(diff_text):
    """Generates a commit message using OpenAI's GPT."""
    prompt = f"Summarize the following Git diff and generate a concise and meaningful commit message:\n\n{diff_text}"
    
    try:
        #response = openai.ChatCompletion.create(
        client = openai.OpenAI(api_key = OPENAI_API_KEY)  # Create a client object
        reponse = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant skilled in writing Git commit messages."},
                      {"role": "user", "content": prompt}],
            max_tokens=50,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating commit message: {e}")
        return None

def apply_commit(commit_message):
    """Stages and commits the changes with the generated message."""
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print(f"Committed successfully: {commit_message}")
    except subprocess.CalledProcessError as e:
        print(f"Error committing changes: {e}")

def main():
    if not (diff_text := get_git_diff()):
        print("No changes detected.")
        return
    
    if not (commit_message := generate_commit_message(diff_text)):
        print("Failed to generate a commit message.")
        return
    
    apply_commit(commit_message)

if __name__ == "__main__":
    main()

