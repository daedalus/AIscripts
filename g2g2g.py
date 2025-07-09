import os
import shutil
import tempfile
import argparse
import requests
import json
from git import Repo, GitCommandError

CACHE_FILE = "commit_msg_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def get_commit_message_from_llm(diff_text, endpoint, model, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Generate a concise and meaningful Git commit message for the following diff:\n\n{diff_text}"
            }
        ],
        "temperature": 0.5,
    }

    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def copy_working_tree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.git'))

def replay_commits_with_new_messages(old_repo_path, new_repo_path, branch, llm_callback, dry_run=False):
    old_repo = Repo(old_repo_path)
    assert not old_repo.bare

    new_repo = Repo.init(new_repo_path)
    commits = list(old_repo.iter_commits(branch, reverse=True))
    temp_checkout_dir = tempfile.mkdtemp()

    prev_commit = None

    cache = load_cache()

    for commit in commits:
        commit_id = commit.hexsha
        print(f"\n📋 Processing {commit_id[:7]}")

        old_repo.git.checkout(commit_id)
        copy_working_tree(old_repo_path, temp_checkout_dir)

        for root, _, files in os.walk(temp_checkout_dir):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, temp_checkout_dir)
                dest = os.path.join(new_repo_path, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(abs_path, dest)
                new_repo.git.add(rel_path)

        if commit_id in cache:
            new_message = cache[commit_id]
            print(f"📝 Cached message: {new_message}")
        else:
            diff = old_repo.git.diff(prev_commit.hexsha, commit_id) if prev_commit else old_repo.git.show(commit_id)
            new_message = llm_callback(diff)
            print(f"📝 Suggested message: {new_message}")
            cache[commit_id] = new_message
            save_cache(cache)

        if not dry_run:
            new_repo.index.commit(new_message, author=commit.author, committer=commit.committer)

        prev_commit = commit

    old_repo.git.checkout(branch)
    shutil.rmtree(temp_checkout_dir)

    if not dry_run:
        print(f"\n✅ New repo created at: {new_repo_path}")

        for tag in old_repo.tags:
            try:
                new_repo.create_tag(tag.name, ref=new_repo.commit())
                print(f"🏷️ Recreated tag: {tag.name}")
            except GitCommandError as e:
                print(f"⚠️ Failed to create tag {tag.name}: {e}")

        new_repo.create_head(branch, ref=new_repo.head.commit)
        print(f"🌿 Branch '{branch}' created in new repo")
    else:
        print("\n⚠️ Dry run mode: no changes committed to new repo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("old_repo", help="Path to the original repository")
    parser.add_argument("new_repo", help="Path to the new repository")
    parser.add_argument("--branch", default="master", help="Branch to replay (default: master)")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible inference endpoint URL")
    parser.add_argument("--model", required=True, help="Model name to use (e.g., gpt-4)")
    parser.add_argument("--api-key", help="API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without committing anything")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    def wrapped_llm(diff_text):
        return get_commit_message_from_llm(diff_text, args.endpoint, args.model, api_key)

    replay_commits_with_new_messages(
        args.old_repo, args.new_repo, args.branch, wrapped_llm, dry_run=args.dry_run
    )