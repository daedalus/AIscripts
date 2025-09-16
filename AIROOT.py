import hashlib
import pickle
import os
import re
import openai
import subprocess
from typing import Optional

SYSTEM_PROMPT = "Expert Linux penetration tester specializing in root recovery. Extensive arsenal of exploits, hacks and tools available."

USER_PROMPT="""Gain ROOT access on target system. Owner authorized recovery. 

RULES:
- Output ONLY single Linux commands
- No explanations, formatting, or thoughts
- No interactive commands (like sudo or su)
- No command repetition
- Gather info first, then exploit
- Create README.md with exploit chain and remediation
- Print ###TERMINATE### when root achieved

I'll send command outputs. Begin."""


# ---------------- CONFIG ----------------
API_URL = os.getenv("OPENAI_API_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

CACHE_FILE = "api_cache.pkl"
COMMANDS_FILE = "commands.pkl"

def query_api(context, model, client):
    # Query API
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": context}],
        temperature=0,
    )
    return response.choices[0].message.content

def execute(cmd: str, timeout: Optional[float] = None) -> str:
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            timeout=timeout
        )
        out = completed.stdout or ""
        err = completed.stderr or ""
        return out + err
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except Exception as e:
        return f"Error executing command: {str(e)}"

def load_cache(filename):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except:
            return {}
    return {}

def save_cache(cache, filename):
    with open(filename, "wb") as f:
        pickle.dump(cache, f)

def main():
    # Load caches
    api_cache = load_cache(CACHE_FILE)
    commands_cache = load_cache(COMMANDS_FILE)
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=API_KEY, base_url=API_URL)
    
    context = USER_PROMPT
    termination_string = '###TERMINATE###'
    
    while True:
        # Generate command hash for API cache
        context_hash = hashlib.sha256(context.encode()).hexdigest()
        
        if context_hash in api_cache:
            command = api_cache[context_hash]
            print("COMMAND[cached]:", command)
        else:
            command = query_api(context, MODEL, client)
            command = re.sub(r"<think>.*?</think>", "", command, flags=re.DOTALL).strip()
            api_cache[context_hash] = command
            save_cache(api_cache, CACHE_FILE)
            print("COMMAND:", command)
        
        # Check for termination
        if command.startswith(termination_string):
            print("Termination command received. Exiting.")
            break
        
        # Execute command and cache result
        command_hash = hashlib.sha256(command.encode()).hexdigest()
        
        if command_hash in commands_cache:
            ret = commands_cache[command_hash]
            print("RET[cached]:", ret)
        else:
            ret = execute(command)
            print("RET:", ret)
            commands_cache[command_hash] = ret
            save_cache(commands_cache, COMMANDS_FILE)
        
        # Update context
        context += command + "\n"
        context += ret + "\n"
        
        # Optional: Add a small delay to avoid rate limiting
        import time
        time.sleep(1)

if __name__ == "__main__":
    main()
