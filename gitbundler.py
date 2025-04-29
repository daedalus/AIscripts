import sys
import os
import subprocess
import argparse
import shutil
import re
import urllib.request
import urllib.error
from tqdm.auto import tqdm


def safe_checkout_or_create_branch(repo_path, branch):
    # Determine current branch
    current_branch = run_git_command(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"], capture_output=True)
    
    # Try to create the branch from current if it doesn't exist
    if not branch_exists(repo_path, branch):
        run_git_command(repo_path, ["checkout", "-b", branch])
    else:
        # If it exists, force checkout only if no conflicts
        run_git_command(repo_path, ["stash", "--include-untracked"], check=False)
        run_git_command(repo_path, ["checkout", branch])
        run_git_command(repo_path, ["stash", "pop"], check=False)

def branch_exists(repo_path, branch_name):
    result = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "--verify", branch_name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

def remote_repo_exists(remote_url, verbose=False):
    """Check if the Git remote URL is reachable on GitHub or similar."""
    if remote_url.startswith("git@"):
        # Convert SSH to HTTPS
        match = re.match(r"git@([^:]+):(.+?)(\.git)?$", remote_url)
        if not match:
            return False
        domain, path = match.groups()[:2]
        test_url = f"https://{domain}/{path}"
    elif remote_url.startswith("https://") or remote_url.startswith("http://"):
        test_url = re.sub(r"\.git$", "", remote_url)
    else:
        return False  # Unsupported remote

    try:
        req = urllib.request.Request(test_url, method='HEAD')
        with urllib.request.urlopen(req, timeout=5) as resp:
            if verbose:
                tqdm.write(f"Checked {test_url}: {resp.status}")
            return 200 <= resp.status < 400
    except urllib.error.HTTPError as e:
        if verbose:
            tqdm.write(f"HEAD {test_url} failed: HTTP {e.code}")
        return False
    except Exception as e:
        if verbose:
            tqdm.write(f"HEAD {test_url} failed: {e}")
        return False

def run_git_command(repo_path, args, check=True, capture_output=False, verbose=False):
    """Helper to run git commands inside a repository."""
    result = subprocess.run(
        ["git", "-C", repo_path] + args,
        check=check,
        capture_output=(capture_output or not verbose),
        text=True
    )
    if capture_output:
        return result.stdout.strip()
    if verbose and result.stdout:
        tqdm.write(result.stdout)
    return None

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

def get_default_branch(repo_path, verbose=False):
    """Get the default branch name ('main' or 'master' or other)."""
    try:
        output = run_git_command(repo_path, ["symbolic-ref", "refs/remotes/origin/HEAD"], capture_output=True, verbose=verbose)
        return output.split("/")[-1]
    except subprocess.CalledProcessError:
        for branch in ("main", "master"):
            try:
                run_git_command(repo_path, ["rev-parse", "--verify", branch], capture_output=True, verbose=verbose)
                return branch
            except subprocess.CalledProcessError:
                continue
        return "main"

def has_uncommitted_changes(repo_path, verbose=False):
    """Check if there are uncommitted changes."""
    output = run_git_command(repo_path, ["status", "--porcelain"], capture_output=True, verbose=verbose)
    return bool(output)

def save_uncommitted_branch(repo_path, default_branch, colorize, verbose):
    """Save uncommitted changes in a new branch called 'uncommitted'."""
    tqdm.write(colored(f"[{repo_path}] Uncommitted changes found. Saving to 'uncommitted' branch.", "yellow", colorize))

    if not branch_exists(repo_path, "uncommitted"):
        run_git_command(repo_path, ["checkout", "-b", "uncommitted"])
    else:
        run_git_command(repo_path, ["checkout", "uncommitted"])

    run_git_command(repo_path, ["add", "-A"], verbose=verbose)
    run_git_command(repo_path, ["commit", "-m", "Save uncommitted changes"], check=False, verbose=verbose)
    run_git_command(repo_path, ["checkout", default_branch], verbose=verbose)

def handle_conflict(repo_path, default_branch, colorize, verbose):
    """Create a 'conflicts' branch and save the conflicts."""
    tqdm.write(colored(f"[{repo_path}] Conflict detected. Creating 'conflicts' branch.", "red", colorize))

    run_git_command(repo_path, ["merge", "--abort"], check=False, verbose=verbose)
    run_git_command(repo_path, ["rebase", "--abort"], check=False, verbose=verbose)

    run_git_command(repo_path, ["checkout", "-b", "conflicts"], verbose=verbose)
    run_git_command(repo_path, ["add", "-A"], verbose=verbose)
    run_git_command(repo_path, ["commit", "-m", "Save conflict state"], check=False, verbose=verbose)
    run_git_command(repo_path, ["checkout", default_branch], verbose=verbose)

    try:
        run_git_command(repo_path, ["pull", "--rebase"], verbose=verbose)
    except subprocess.CalledProcessError:
        tqdm.write(colored(f"[{repo_path}] Rebase after conflict branch creation failed.", "yellow", colorize))

def colored(text, color, colorize=True):
    """Return colored text if colorize is True."""
    if not colorize:
        return text
    colors = {"red": 91, "green": 92, "yellow": 93, "blue": 94, "magenta": 95, "cyan": 96}
    return f"\033[{colors.get(color, 0)}m{text}\033[0m"

def process_repo(repo_path, output_dir, skip_existing, overwrite_existing, colorize, delete_repo, verbose):
    """Process a single repository."""
    tqdm.write(colored(f"Processing {repo_path}...", "blue", colorize))

    default_branch = get_default_branch(repo_path, verbose=verbose)

    if has_uncommitted_changes(repo_path):
        tqdm.write(colored(f"[{repo_path}] Uncommitted changes detected. Saving to 'uncommitted' branch.", "yellow", colorize))
        safe_checkout_or_create_branch(repo_path, "uncommitted")
        run_git_command(repo_path, ["add", "-A"])
        run_git_command(repo_path, ["commit", "-m", "Save uncommitted changes"])
        run_git_command(repo_path, ["checkout", default_branch])
    else:
        try:
            # Get remote URL
            remote_url = run_git_command(repo_path, ["config", "--get", "remote.origin.url"], capture_output=True, verbose=verbose)

            # Check if remote URL is reachable
            tqdm.write("Pulling changes from remote.")
            if remote_repo_exists(remote_url, verbose=verbose):
                run_git_command(repo_path, ["fetch"], verbose=verbose)
                run_git_command(repo_path, ["pull", "--rebase"], verbose=verbose)
            else:
                tqdm.write(colored(f"[{repo_path}] Remote {remote_url} unreachable. Skipping fetch/rebase.", "yellow", colorize))

        except subprocess.CalledProcessError:
            tqdm.write(colored(f"[{repo_path}] Remote unreachable or pull failed. Skipping sync.", "yellow", colorize))

    repo_name = os.path.basename(os.path.abspath(repo_path))
    bundle_filename = f"{repo_name}.bundle"
    bundle_path = os.path.join(output_dir, bundle_filename)

    skipped = False
    if os.path.exists(bundle_path):
        if skip_existing:
            tqdm.write(colored(f"[{repo_path}] Bundle already exists, skipping.", "yellow", colorize))
            #return "skipped"
            skipped = True
        elif not overwrite_existing:
            tqdm.write(colored(f"[{repo_path}] Bundle exists and overwrite not allowed. Skipping.", "red", colorize))
            skipped = True

    if not skipped:
        tqdm.write(colored(f"[{repo_path}] Creating bundle at {bundle_path}...", "green", colorize))
        run_git_command(repo_path, ["bundle", "create", bundle_path, "--all"], verbose=verbose)

    if delete_repo:
        tqdm.write(colored(f"[{repo_path}] Deleting repository directory.", "red", colorize))
        try:
            shutil.rmtree(repo_path)
        except Exception as e:
            tqdm.write(colored(f"[{repo_path}] Error deleting repository: {e}", "red", colorize))

    return "created"

def main(base_dir, output_dir, skip_existing, overwrite_existing, colorize, delete_repo, verbose):
    """Main function to process all repositories."""
    git_repos = find_git_repos(base_dir)
    tqdm.write(colored(f"Found {len(git_repos)} repositories.", "magenta", colorize))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    created = skipped = errors = 0

    for repo in tqdm(git_repos, ncols=80, colour="cyan", file=sys.stderr, position=0, leave=True):
        #tqdm.write(f"Processing directories")
        try:
            result = process_repo(repo, output_dir, skip_existing, overwrite_existing, colorize, delete_repo, verbose)
            if result == "created":
                created += 1
            else:
                skipped += 1
        except Exception as e:
            tqdm.write(colored(f"Error processing {repo}: {e}", "red", colorize))
            errors += 1

    tqdm.write(colored(f"\nSummary:", "magenta", colorize))
    tqdm.write(colored(f"Bundles created: {created}", "green", colorize))
    tqdm.write(colored(f"Bundles skipped: {skipped}", "yellow", colorize))
    tqdm.write(colored(f"Errors: {errors}", "red", colorize))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Git repositories and create bundles.")
    parser.add_argument("base_dir", help="Directory to scan for Git repositories")
    parser.add_argument("--output-dir", default=None, help="Directory to save bundles (default: inside each repo)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip repositories that already have a bundle")
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite bundles if they already exist")
    parser.add_argument("--delete-repo", action="store_true", help="Delete the repository directory after processing")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Git command output")

    args = parser.parse_args()
    base_directory = args.base_dir.strip()
    output_directory = args.output_dir.strip() if args.output_dir else None

    if not os.path.isdir(base_directory):
        tqdm.write("\033[91mInvalid base directory.\033[0m")
    else:
        if output_directory is None:
            output_directory = os.path.join(os.path.abspath(base_directory), "bundles")
        main(
            base_directory,
            output_directory,
            args.skip_existing,
            args.overwrite_existing,
            not args.no_color,
            args.delete_repo,
            args.verbose
        )

