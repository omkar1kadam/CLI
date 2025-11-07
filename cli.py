#!/usr/bin/env python3
"""
Full Gemini-powered CLI
Merged & improved file: combines your previous code + robust interpreter + chained actions.

Requirements:
  pip install google-genai rich

Set API key:
  setx GEMINI_API_KEY "YOUR_KEY"  (Windows) or export GEMINI_API_KEY="YOUR_KEY" (Linux/Mac)
"""

import os
import re
import json
import shutil
import subprocess
import time
from typing import Optional, Dict, Any, List

# Gemini SDK import (google-genai)
try:
    from google import genai
except Exception as e:
    raise ImportError("google-genai SDK not found. Install with: pip install -U google-genai") from e

# Rich for colored output & prompts
from rich.console import Console
from rich.prompt import Prompt

console = Console()

# --- Config ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    console.print("[red]Error: GEMINI_API_KEY not set![/red]")
    raise SystemExit(1)

MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
client = genai.Client(api_key=API_KEY)

# --- Helpers for calling Gemini safely ---
def call_gemini(prompt: str, max_retries: int = 2, temperature: float = 0.2) -> Optional[str]:
    """Call Gemini and return text response (or None on failure). Retries on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config={"temperature": temperature},
            )
            # Newer responses expose .text
            text = getattr(resp, "text", None)
            if text is None:
                text = str(resp)
            return text.strip()
        except Exception as e:
            console.print(f"[yellow]Gemini call failed (attempt {attempt+1}): {e}[/yellow]")
            if attempt < max_retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            return None

def extract_json(text: str) -> Any:
    """
    Extract JSON object/array from a text blob.
    Returns parsed JSON or raises json.JSONDecodeError / ValueError.
    """
    if not text:
        raise ValueError("Empty response text")

    # Try direct load first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find the first JSON object or array in text
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if not match:
        raise ValueError("No JSON found in response")

    candidate = match.group(1)
    # Try to balance braces if necessary by expanding to last brace
    # (already matched to last '}' or ']' via regex)
    return json.loads(candidate)

# --- Interpreter: convert user input to steps (chained actions) ---
def interpret_command_to_steps(user_input: str) -> Optional[List[Dict[str, Any]]]:
    """
    Ask Gemini to produce a JSON with "steps": [ {action, params}, ... ]
    This function validates and returns the steps list or None.
    """
    system_instructions = f"""
You are an AI assistant that converts CLI-style natural language into a strict JSON "steps" plan.
Return ONLY valid JSON (no extra commentary). Format:

{{
  "steps": [
    {{
      "action": "<action_name>",
      "params": {{ ... }}
    }}
  ]
}}

Allowed actions (use these names exactly):
createfile, writefile, readfile, renamefile, deletefile, copyfile, movefile,
createfolder, deletefolder, runfile, editfile, searchfile,
pwd, ls, cd, clear,
summarizefile, explaincode, generatecode, convertcode

Rules:
- For generatecode steps: params MUST include "filename", "language", and "description".
- For runfile/readfile/deletefile/renamefile: include "filename" (rename uses "old" and "new" keys).
- If user asks multiple things, break into sequential steps.
- If any required param cannot be inferred, include the step but keep params empty; the client will prompt/fallback.
- Return minimal JSON necessary — do not include extraneous fields.

User request: {user_input}
"""
    text = call_gemini(system_instructions, max_retries=2, temperature=0.15)
    if not text:
        console.print("[red]No response from Gemini.[/red]")
        return None

    try:
        parsed = extract_json(text)
        if isinstance(parsed, dict) and "steps" in parsed and isinstance(parsed["steps"], list):
            return parsed["steps"]
        # If top-level is a steps array directly
        if isinstance(parsed, list):
            return parsed
        raise ValueError("Parsed JSON not in expected {steps: [...] } format")
    except Exception as e:
        console.print(f"[red]Error parsing Gemini JSON:[/red] {e}")
        console.print("[yellow]Raw Gemini output:[/yellow]")
        console.print(text)
        return None

# --- File & System Operations (safe, modular) ---
def create_file(filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("")  # create empty file
    console.print(f"[green]Created file {filename}[/green]")

def write_file(filename: str, content: str, append: bool = False):
    mode = "a" if append else "w"
    with open(filename, mode, encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
    console.print(f"[green]Wrote to {filename} (append={append})[/green]")

def read_file(filename: str):
    if not os.path.exists(filename):
        console.print(f"[red]File not found: {filename}[/red]")
        return
    with open(filename, "r", encoding="utf-8") as f:
        console.print(f.read())

def rename_file(old: str, new: str):
    if not os.path.exists(old):
        console.print(f"[red]Source file not found: {old}[/red]")
        return
    os.rename(old, new)
    console.print(f"[green]Renamed {old} → {new}[/green]")

def delete_file(filename: str):
    if not os.path.exists(filename):
        console.print(f"[red]File not found: {filename}[/red]")
        return
    os.remove(filename)
    console.print(f"[green]Deleted {filename}[/green]")

def copy_file(src: str, dest: str):
    if not os.path.exists(src):
        console.print(f"[red]Source file not found: {src}[/red]")
        return
    shutil.copy(src, dest)
    console.print(f"[green]Copied {src} → {dest}[/green]")

def move_file(src: str, dest: str):
    if not os.path.exists(src):
        console.print(f"[red]Source file not found: {src}[/red]")
        return
    shutil.move(src, dest)
    console.print(f"[green]Moved {src} → {dest}[/green]")

def create_folder(name: str):
    os.makedirs(name, exist_ok=True)
    console.print(f"[green]Created folder {name}[/green]")

def delete_folder(name: str):
    if not os.path.exists(name):
        console.print(f"[red]Folder not found: {name}[/red]")
        return
    shutil.rmtree(name)
    console.print(f"[green]Deleted folder {name}[/green]")

def search_in_file(filename: str, query: str):
    if not os.path.exists(filename):
        console.print(f"[red]File not found: {filename}[/red]")
        return
    with open(filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if query in line:
                console.print(f"[blue]{idx}: {line.rstrip()}[/blue]")

# --- Execute / Run files ---
def execute_file(filename: str):
    if not os.path.exists(filename):
        console.print(f"[red]File not found: {filename}[/red]")
        return
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".cpp":
            # compile to an exe/binary in same folder
            exe_name = os.path.splitext(filename)[0] + (".exe" if os.name == "nt" else "")
            compile_cmd = ["g++", filename, "-o", exe_name]
            console.print(f"[cyan]Compiling {filename}...[/cyan]")
            subprocess.run(compile_cmd, check=True)
            console.print(f"[green]Compiled → {exe_name}[/green]")
            # Execute
            if os.name == "nt":
                run_cmd = [exe_name]
            else:
                run_cmd = ["./" + exe_name]
            proc = subprocess.run(run_cmd, capture_output=True, text=True)
            console.print(proc.stdout)
            if proc.stderr:
                console.print(f"[red]{proc.stderr}[/red]")
        elif ext == ".py":
            proc = subprocess.run(["python", filename], capture_output=True, text=True)
            console.print(proc.stdout)
            if proc.stderr:
                console.print(f"[red]{proc.stderr}[/red]")
        else:
            console.print(f"[yellow]Execution not supported for {ext} files[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Execution error:[/red] {e}")

# --- AI content actions (generate/explain/convert/summarize) ---
def generate_code(filename: str, language: str, description: str):
    if not filename or not language or not description:
        console.print("[red]generate_code missing filename/language/description[/red]")
        return
    prompt = f"Write only the code (no extra text). Language: {language}. Task: {description}."
    text = call_gemini(prompt, max_retries=2, temperature=0.2)
    if not text:
        console.print("[red]Failed to generate code[/red]")
        return
    # If model adds commentary, try to extract code block or entire text
    code = text
    # strip common wrappers (triple backticks)
    code = re.sub(r"^```(?:\w+)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    write_file(filename, code, append=False)
    console.print(f"[green]Generated {language} code → {filename}[/green]")

def explain_code(filename: str):
    if not os.path.exists(filename):
        console.print(f"[red]File not found: {filename}[/red]")
        return
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    prompt = f"Explain this code briefly and clearly:\n\n{content}"
    text = call_gemini(prompt, max_retries=2, temperature=0.2)
    if text:
        console.print(text)
    else:
        console.print("[red]Failed to get explanation[/red]")

def summarize_file(filename: str):
    if not os.path.exists(filename):
        console.print(f"[red]File not found: {filename}[/red]")
        return
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    prompt = f"Summarize the following file content in a short paragraph:\n\n{content}"
    text = call_gemini(prompt, max_retries=2, temperature=0.2)
    if text:
        console.print(text)
    else:
        console.print("[red]Failed to summarize file[/red]")

def convert_code(filename: str, language: str):
    if not os.path.exists(filename):
        console.print(f"[red]File not found: {filename}[/red]")
        return
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    prompt = f"Convert this code to {language}. Provide only the code."
    text = call_gemini(prompt + "\n\n" + content, max_retries=2, temperature=0.2)
    if text:
        # clean code block markers
        code = re.sub(r"^```(?:\w+)?\s*", "", text)
        code = re.sub(r"\s*```$", "", code)
        console.print(code)
    else:
        console.print("[red]Failed to convert code[/red]")

# --- Step processor (handles chaining and missing params) ---
def process_steps(steps: List[Dict[str, Any]], original_user: str):
    """
    steps: list of {"action": str, "params": {...}}
    If params are missing for critical steps, attempt to infer from original_user text.
    """
    for idx, step in enumerate(steps):
        action = step.get("action")
        params = step.get("params") or {}
        console.print(f"[bold]Step {idx+1}: {action}[/bold]")

        # param fallbacks / common normalization
        # Some models use different param names; try to map common ones
        def p(key, alt=None):
            return params.get(key) or (params.get(alt) if alt else None)

        # Try naive inference from the original_user string if necessary
        if action in ("createfile", "writefile", "readfile", "deletefile", "runfile", "searchfile", "generatecode", "explaincode", "summarizefile", "convertcode"):
            filename = p("filename") or p("file") or p("name")
            if not filename:
                # attempt to extract a filename-like token from user input (e.g. something ending with .cpp/.py/.txt)
                m = re.search(r'([\w\-. ]+\.\w{1,5})', original_user)
                if m:
                    filename = m.group(1).strip()
                    console.print(f"[yellow]Inferred filename: {filename}[/yellow]")
            params["filename"] = filename

        # Action dispatch with validations & informative messages
        try:
            if action == "createfile":
                if not params.get("filename"):
                    console.print("[red]Missing filename for createfile[/red]")
                else:
                    create_file(params["filename"])

            elif action == "writefile":
                filename = params.get("filename")
                content = params.get("content") or params.get("text") or params.get("description")
                append = bool(params.get("append", False))
                if not filename or content is None:
                    console.print("[red]Missing filename or content for writefile[/red]")
                else:
                    write_file(filename, content, append=append)

            elif action == "readfile":
                if not params.get("filename"):
                    console.print("[red]Missing filename for readfile[/red]")
                else:
                    read_file(params["filename"])

            elif action == "renamefile":
                # support "old" & "new" or "filename" & "new_name"
                old = params.get("old") or params.get("filename")
                new = params.get("new") or params.get("new_name")
                if not old or not new:
                    console.print("[red]Missing old/new for renamefile[/red]")
                else:
                    rename_file(old, new)

            elif action == "deletefile":
                if not params.get("filename"):
                    console.print("[red]Missing filename for deletefile[/red]")
                else:
                    delete_file(params["filename"])

            elif action == "copyfile":
                src = params.get("source") or params.get("from")
                dest = params.get("destination") or params.get("to")
                if not src or not dest:
                    console.print("[red]Missing source/destination for copyfile[/red]")
                else:
                    copy_file(src, dest)

            elif action == "movefile":
                src = params.get("source") or params.get("from")
                dest = params.get("destination") or params.get("to")
                if not src or not dest:
                    console.print("[red]Missing source/destination for movefile[/red]")
                else:
                    move_file(src, dest)

            elif action == "createfolder":
                folder = params.get("foldername") or params.get("name")
                if not folder:
                    console.print("[red]Missing foldername for createfolder[/red]")
                else:
                    create_folder(folder)

            elif action == "deletefolder":
                folder = params.get("foldername") or params.get("name")
                if not folder:
                    console.print("[red]Missing foldername for deletefolder[/red]")
                else:
                    delete_folder(folder)

            elif action == "searchfile":
                filename = params.get("filename")
                query = params.get("text") or params.get("query")
                if not filename or not query:
                    console.print("[red]Missing filename or query for searchfile[/red]")
                else:
                    search_in_file(filename, query)

            elif action == "runfile":
                if not params.get("filename"):
                    console.print("[red]Missing filename for runfile[/red]")
                else:
                    execute_file(params["filename"])

            elif action == "generatecode":
                filename = params.get("filename")
                language = params.get("language") or params.get("lang")
                description = params.get("description") or params.get("task") or params.get("content")
                if not filename or not language or not description:
                    console.print("[red]Missing filename/language/description for generatecode[/red]")
                else:
                    generate_code(filename, language, description)

            elif action == "explaincode":
                if not params.get("filename"):
                    console.print("[red]Missing filename for explaincode[/red]")
                else:
                    explain_code(params["filename"])

            elif action == "summarizefile":
                if not params.get("filename"):
                    console.print("[red]Missing filename for summarizefile[/red]")
                else:
                    summarize_file(params["filename"])

            elif action == "convertcode":
                filename = params.get("filename")
                language = params.get("language") or params.get("lang")
                if not filename or not language:
                    console.print("[red]Missing filename/language for convertcode[/red]")
                else:
                    convert_code(filename, language)

            elif action == "pwd":
                console.print(os.getcwd())

            elif action == "ls":
                for entry in os.listdir():
                    console.print(entry)

            elif action == "cd":
                path = params.get("path") or params.get("directory")
                if not path:
                    console.print("[red]Missing path for cd[/red]")
                else:
                    try:
                        os.chdir(path)
                        console.print(f"[green]Changed dir → {path}[/green]")
                    except Exception as e:
                        console.print(f"[red]cd failed: {e}[/red]")

            elif action == "clear":
                os.system("cls" if os.name == "nt" else "clear")

            else:
                console.print(f"[yellow]Unknown or unsupported action: {action}[/yellow]")

        except Exception as e:
            console.print(f"[red]Error processing step {idx+1} ({action}): {e}[/red]")

# --- CLI main loop ---
def main_loop():
    console.print("[bold cyan]Gemini CLI — type 'exit' or 'quit' to leave.[/bold cyan]")
    while True:
        user_input = Prompt.ask(">>").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            console.print("Goodbye.")
            break

        # Interpret to steps
        steps = interpret_command_to_steps(user_input)
        if not steps:
            console.print("[yellow]Could not interpret command. Try being more explicit (e.g. 'generate code in C++ for selection sort and save as algo.cpp').[/yellow]")
            continue

        process_steps(steps, user_input)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        console.print("\n[cyan]Interrupted. Bye.[/cyan]")
