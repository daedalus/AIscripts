import os
import subprocess
import argparse
from tqdm import tqdm
import shutil

def run_git_command(repo_path, args, check=True, capture_output=False):
    """Helper to run git commands inside a repository."""
    result = subprocess.run(["git", "-C", repo_path] + args, check=check,
                             capture_output=capture_output, text=True)
    return result.stdout.strip() if capture_output else None

def is_git_repo(path):
    """Check if a directory is a git repository."""
    return os.path.isdir(os.path.join(path, ".git"))

def find_git_repos(base_dir):
    """Find all git repositories under base_dir."""
    git_repos = []
    for root, dirs, files in os.walk(base_dir):
        if is_git_repo(root):
            git_repos.append(root)
            dirs.clear()  # Don't recurse into subdirectories of a git repo
    return git_repos

def get_default_branch(repo_path):
    """Get the default branch name ('main' or 'master' or other)."""
    try:
        output = run_git_command(repo_path, ["symbolic-ref", "refs/remotes/origin/HEAD"], capture_output=True)
        return output.split("/")[-1]
    except subprocess.CalledProcessError:
        # Try common defaults
        for branch in ("main", "master"):
            try:
                run_git_command(repo_path, ["rev-parse", "--verify", branch], capture_output=True)
                return branch
            except subprocess.CalledProcessError:
                continue
        return "main"  # Fallback

def has_uncommitted_changes(repo_path):
    """Check if there are uncommitted changes."""
    output = run_git_command(repo_path, ["status", "--porcelain"], capture_output=True)
    return bool(output)

def handle_conflict(repo_path, default_branch, colorize):
    """Create a 'conflicts' branch and save the conflicts."""
    print(colored(f"[{repo_path}] Conflict detected. Creating 'conflicts' branch.", "red", colorize))

    try:
        run_git_command(repo_path, ["merge", "--abort"], check=False)
        run_git_command(repo_path, ["rebase", "--abort"], check=False)
    except subprocess.CalledProcessError:
        pass

    run_git_command(repo_path, ["checkout", "-b", "conflicts"])
    run_git_command(repo_path, ["add", "-A"])
    run_git_command(repo_path, ["commit", "-m", "Save conflict state"])
    run_git_command(repo_path, ["checkout", default_branch])

    try:
        run_git_command(repo_path, ["pull", "--rebase"])
    except subprocess.CalledProcessError:
        print(colored(f"[{repo_path}] Rebase after conflict branch creation failed.", "yellow", colorize))

def colored(text, color, colorize=True):
    """Return colored text if colorize is True."""
    if not colorize:
        return text
    colors = {"red": 91, "green": 92, "yellow": 93, "blue": 94, "magenta": 95, "cyan": 96}
    return f"\033[{colors.get(color, 0)}m{text}\033[0m"

def process_repo(repo_path, output_dir, skip_existing, overwrite_existing, colorize, delete_repo):
    """Process a single repository."""
    print(colored(f"Processing {repo_path}...", "blue", colorize))
    
    if has_uncommitted_changes(repo_path):
        print(colored(f"[{repo_path}] Warning: Uncommitted changes detected. Skipping pull.", "yellow", colorize))
    
    default_branch = get_default_branch(repo_path)
    try:
        run_git_command(repo_path, ["fetch"])
        run_git_command(repo_path, ["pull", "--rebase"])
    except subprocess.CalledProcessError:
        handle_conflict(repo_path, default_branch, colorize)

    repo_name = os.path.basename(os.path.abspath(repo_path))
    bundle_filename = f"{repo_name}.bundle"
    bundle_path = os.path.join(output_dir, bundle_filename)

    if os.path.exists(bundle_path):
        if skip_existing:
            print(colored(f"[{repo_path}] Bundle already exists, skipping.", "yellow", colorize))
            return "skipped"
        elif not overwrite_existing:
            print(colored(f"[{repo_path}] Bundle exists and overwrite not allowed. Skipping.", "red", colorize))
            return "skipped"

    print(colored(f"[{repo_path}] Creating bundle at {bundle_path}...", "green", colorize))
    run_git_command(repo_path, ["bundle", "create", bundle_path, "--all"])

    if delete_repo:
        print(colored(f"[{repo_path}] Deleting repository directory.", "red", colorize))
        try:
            shutil.rmtree(repo_path)  # Delete the entire repository directory
        except Exception as e:
            print(colored(f"[{repo_path}] Error deleting repository: {e}", "red", colorize))

    return "created"

def main(base_dir, output_dir, skip_existing, overwrite_existing, colorize, delete_repo):
    """Main function to process all repositories."""
    git_repos = find_git_repos(base_dir)
    print(colored(f"Found {len(git_repos)} repositories.", "magenta", colorize))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    created = skipped = errors = 0

    for repo in tqdm(git_repos, desc="Processing repositories", ncols=80, colour="cyan"):
        try:
            result = process_repo(repo, output_dir, skip_existing, overwrite_existing, colorize, delete_repo)
            if result == "created":
                created += 1
            else:
                skipped += 1
        except Exception as e:
            print(colored(f"Error processing {repo}: {e}", "red", colorize))
            errors += 1

    print(colored(f"\nSummary:", "magenta", colorize))
    print(colored(f"Bundles created: {created}", "green", colorize))
    print(colored(f"Bundles skipped: {skipped}", "yellow", colorize))
    print(colored(f"Errors: {errors}", "red", colorize))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Git repositories and create bundles.")
    parser.add_argument("base_dir", help="Directory to scan for Git repositories")
    parser.add_argument("--output-dir", default=None, help="Directory to save bundles (default: inside each repo)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip repositories that already have a bundle")
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite bundles if they already exist")
    parser.add_argument("--delete-repo", action="store_true", help="Delete the repository directory after processing")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()
    base_directory = args.base_dir.strip()
    output_directory = args.output_dir.strip() if args.output_dir else None

    if not os.path.isdir(base_directory):
        print("\033[91mInvalid base directory.\033[0m")
    else:
        if output_directory is None:
            output_directory = os.path.join(os.path.abspath(base_directory), "bundles")
        main(base_directory, output_directory, args.skip_existing, args.overwrite_existing, not args.no_color, args.delete_repo)