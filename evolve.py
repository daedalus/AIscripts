#!/usr/bin/env python3
"""
evolve.py — A self-improving script that works with any OpenAI-compatible API.

Each run:
  1. Reads its own source code.
  2. Sends it to an LLM asking for one meaningful improvement.
  3. Overwrites itself with the improved version.
  4. Re-executes the new version via os.execv.

Configuration — set via environment variables or CLI flags:
  EVOLVE_API_KEY      API key (required)
  EVOLVE_BASE_URL     Base URL of OpenAI-compatible endpoint
                      (default: https://api.openai.com/v1)
  EVOLVE_MODEL        Model name (default: gpt-4o)
  EVOLVE_MAX_TOKENS   Max tokens for completion (default: 4096)
  EVOLVE_MAX_GEN      Max generations before stopping, 0=unlimited (default: 5)

Examples:
  # OpenAI
  EVOLVE_API_KEY=sk-...  python evolve.py

  # Anthropic (openai-compat endpoint)
  EVOLVE_BASE_URL=https://api.anthropic.com/v1 \
  EVOLVE_MODEL=claude-opus-4-5 \
  EVOLVE_API_KEY=sk-ant-... python evolve.py

  # Ollama (local)
  EVOLVE_BASE_URL=http://localhost:11434/v1 \
  EVOLVE_MODEL=llama3 \
  EVOLVE_API_KEY=ollama python evolve.py

  # Groq
  EVOLVE_BASE_URL=https://api.groq.com/openai/v1 \
  EVOLVE_MODEL=llama-3.3-70b-versatile \
  EVOLVE_API_KEY=gsk_... python evolve.py

  # Together AI
  EVOLVE_BASE_URL=https://api.together.xyz/v1 \
  EVOLVE_MODEL=meta-llama/Llama-3-70b-chat-hf \
  EVOLVE_API_KEY=... python evolve.py

  # CLI overrides (take precedence over env vars):
  python evolve.py --base-url http://localhost:11434/v1 --model llama3 --api-key ollama
"""

import argparse
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
import json

# ── env-var / default config ──────────────────────────────────────────────────
_GENERATION_ENV = "EVOLVE_GENERATION"

DEFAULTS = dict(
    base_url   = "https://api.openai.com/v1",
    model      = "gpt-4o",
    max_tokens = 4096,
    max_gen    = 5,
)
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-improving script — works with any OpenAI-compatible endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--api-key",    default=os.environ.get("EVOLVE_API_KEY"),
                   help="API key (env: EVOLVE_API_KEY)")
    p.add_argument("--base-url",   default=os.environ.get("EVOLVE_BASE_URL",  DEFAULTS["base_url"]),
                   help="Base URL of OpenAI-compatible endpoint (env: EVOLVE_BASE_URL)")
    p.add_argument("--model",      default=os.environ.get("EVOLVE_MODEL",     DEFAULTS["model"]),
                   help="Model name (env: EVOLVE_MODEL)")
    p.add_argument("--max-tokens", default=int(os.environ.get("EVOLVE_MAX_TOKENS", DEFAULTS["max_tokens"])),
                   type=int, help="Max tokens for completion (env: EVOLVE_MAX_TOKENS)")
    p.add_argument("--max-gen",    default=int(os.environ.get("EVOLVE_MAX_GEN", DEFAULTS["max_gen"])),
                   type=int, help="Max generations, 0=unlimited (env: EVOLVE_MAX_GEN)")
    p.add_argument("--dry-run",    action="store_true",
                   help="Show improved source but do NOT overwrite or re-exec")
    return p.parse_args()


def current_generation() -> int:
    return int(os.environ.get(_GENERATION_ENV, "0"))


def banner(gen: int, cfg: argparse.Namespace) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  🧬  evolve.py  —  generation {gen}")
    print(f"  🔌  {cfg.base_url}")
    print(f"  🤖  {cfg.model}")
    if cfg.dry_run:
        print("  🔍  DRY-RUN MODE — file will not be overwritten")
    print(f"{bar}\n")


def read_self() -> str:
    with open(__file__, "r", encoding="utf-8") as fh:
        return fh.read()


def call_api(cfg: argparse.Namespace, messages: list) -> str:
    """
    Pure-stdlib HTTP call to any OpenAI-compatible /chat/completions endpoint.
    No third-party packages required.
    """
    url = cfg.base_url.rstrip("/") + "/chat/completions"

    payload = json.dumps({
        "model":      cfg.model,
        "max_tokens": cfg.max_tokens,
        "messages":   messages,
    }).encode("utf-8")

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        sys.exit(f"❌  HTTP {e.code} from API:\n{detail}")
    except urllib.error.URLError as e:
        sys.exit(f"❌  Could not reach {url}: {e.reason}")

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        sys.exit(f"❌  Unexpected API response shape:\n{json.dumps(body, indent=2)}")


def ask_llm(source: str, generation: int, cfg: argparse.Namespace) -> str:
    """Send source to the LLM and return the improved source."""
    if not cfg.api_key:
        sys.exit(
            "❌  No API key found.\n"
            "    Set EVOLVE_API_KEY env var or pass --api-key <key>"
        )

    system = (
        "You are an expert Python engineer. "
        "When given a Python script you return ONLY the complete improved source — "
        "no markdown fences, no prose, no explanations outside of code comments."
    )

    user = textwrap.dedent(f"""
        Below is a self-improving script currently at generation {generation}.

        Improve it in ONE meaningful way. Good ideas (pick whichever fits best):
          • Better error handling or user-friendly error messages.
          • Timing / performance instrumentation.
          • Show a unified diff of what changed between generations.
          • Persist a changelog (appended to a sidecar .md file).
          • Add --dry-run, --verbose, or other useful CLI flags.
          • Improve the prompt sent to the LLM.
          • Anything else genuinely useful.

        Rules:
          • Return ONLY the complete, runnable Python source — no markdown fences.
          • Keep the self-rewriting + re-exec loop intact.
          • Keep the {_GENERATION_ENV!r} env-var logic intact.
          • Keep the OpenAI-compatible HTTP logic intact (no new dependencies).
          • Add a one-line comment near the top saying what THIS generation changed.

        Current source:
        {source}
    """).strip()

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    print(f"📡  Calling {cfg.model} at {cfg.base_url} …")
    t0      = time.time()
    raw     = call_api(cfg, messages)
    elapsed = time.time() - t0
    print(f"⏱️   Response received in {elapsed:.1f}s")

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:python)?\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "",         raw)
    return raw.strip()


def overwrite_self(new_source: str) -> None:
    path = os.path.abspath(__file__)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_source)
    print(f"✅  Wrote improved source → {path}")


def reexec(next_gen: int) -> None:
    """Replace current process with a fresh run of the (now-updated) script."""
    env = os.environ.copy()
    env[_GENERATION_ENV] = str(next_gen)
    # Preserve CLI args so provider/model config survives across generations
    args = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
    print(f"🔄  Re-executing as generation {next_gen} …\n")
    os.execve(sys.executable, args, env)


def main() -> None:
    cfg = parse_args()
    gen = current_generation()
    banner(gen, cfg)

    if cfg.max_gen and gen >= cfg.max_gen:
        print(f"🏁  Reached max generations ({cfg.max_gen}). Stopping.")
        return

    source     = read_self()
    new_source = ask_llm(source, gen, cfg)

    if cfg.dry_run:
        print("\n" + "─" * 60)
        print("  DRY-RUN: improved source (not written)")
        print("─" * 60)
        print(new_source)
        return

    overwrite_self(new_source)
    reexec(gen + 1)


if __name__ == "__main__":
    main()