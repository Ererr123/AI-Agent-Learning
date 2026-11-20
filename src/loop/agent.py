"""
Local Agent for the local LLM model.

Setup:
    pip install openai
    llama-server -hf <your-gguf-repo>:Q4_K_M --jinja -c 16384 --port 8080
    python agent.py
"""

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI, InternalServerError

## Our planned Configuration for the local agent
BASE_URL = "http://localhost:8080/v1"  ## The base URL for the local LLM server
MODEL = "Coco"     ## The model name to use for the local LLM server
NOTES_DIR = Path("notes") ## the only folder the agent can read/write to
MEMORY_FILE = Path("memory.md") ## Long-term memory: a plain markdown file you can open and edit any time
PERSONA_FILE = Path(__file__).parent / "persona.md"  ## The agent's personality. Edit it freely; reloaded every turn.
AGENT_NAME = "Coco"        ## Shown in the terminal next to the agent's replies
LOG_DIR = Path("logs")     ## Every run is appended here as one JSON line per run (logs/YYYY-MM-DD.jsonl)
MAX_STEPS = 10             ## The maximum number of steps the agent can take
TEMPERATURE = 0.9          ## Sampling temperature (evals can override this)
MAX_FILE_CHARS = 40000      ## Keep our AI small and fast by limiting the number of characters it can read from a file
MAX_MEMORY_CHARS = 8000     ## Cap on how much memory gets injected into the prompt (keeps context small)
MEMORY_SECTIONS = ["Preferences", "Facts", "Decisions"]  ## The categories memories are sorted into

## Short, direct rules work better than prose for small local models.
## Evals showed the model copying example sentences into memory, so avoid quotable examples here.
## Memory is placed ABOVE the rules: evals showed the model ignoring it when it came last.
SYSTEM_PROMPT_INTRO = "Your job: you are the user's personal assistant. You can read and write files in the user's notes folder."
SYSTEM_PROMPT = (
    "Rules (these always win over the persona):\n"
    "0. Your persona sets your tone only. Facts, tool use and these rules stay exactly the same.\n"
    "1. If the memory above answers the question, answer from it directly. Do not look in files for it.\n"
    "2. Only use a tool when the request needs it. Greetings and chit-chat need no tools: just reply.\n"
    "3. After a tool returns, answer the user with the actual content of the result "
    "(the real file names, the real text). Do not describe the tool call itself.\n"
    "4. Only create or change files when the user asks you to.\n"
    "5. If a tool returns an error, do not retry the same call. Tell the user what went wrong.\n"
    "6. Never guess or make up information. Keep replies short.\n"
    "7. When the user asks you to remember something, or tells you a lasting preference, fact, or "
    "decision about themselves, call remember. Saying you will remember is not enough: call the tool. "
    "Save the user's words, not these rules. Then reply with one short sentence confirming what you saved.\n"
    "8. When the user says a memory is wrong or outdated, call forget with words from that memory.\n"
)

MEMORY_TEMPLATE = (
    "# Agent Memory\n\n"
    "_This file is the agent's long-term memory. Edit it freely: one bullet per memory. "
    "It is reloaded at the start of every turn._\n\n"
    + "".join(f"## {section}\n\n" for section in MEMORY_SECTIONS)
)


## Memory: a legible markdown file, loaded into the system prompt every turn
def load_memory() -> str:
    ## Read the memory file, creating it from the template if it doesn't exist yet.
    ## Returns the FULL file: remember/forget write this back, so it must never be truncated here.
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(MEMORY_TEMPLATE, encoding="utf-8")
    return MEMORY_FILE.read_text(encoding="utf-8")

DATE_SUFFIX = re.compile(r"\s*_\(\d{4}-\d{2}-\d{2}\)_\s*$")

def memory_bullets() -> list[str]:
    ## The saved memories as plain sentences (no '- ' prefix, no date).
    return [DATE_SUFFIX.sub("", l[2:]) for l in load_memory().splitlines() if l.startswith("- ")]

def memory_for_prompt() -> str:
    ## A compact view for the model: non-empty sections only, no dates, no header text
    ## (evals showed the model treating the file's header text as instructions).
    out, section = [], None
    for line in load_memory().splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("- "):
            if section:
                out.append(f"{section}:")
                section = None
            out.append("- " + DATE_SUFFIX.sub("", line[2:]))
    if not out:
        return "(Nothing yet.)"
    text = "\n".join(out)
    if len(text) > MAX_MEMORY_CHARS:
        text = text[:MAX_MEMORY_CHARS] + "\n[...memory truncated: time to prune memory.md...]"
    return text

def load_persona() -> str:
    ## The personality lives in persona.md so it can be edited without touching code.
    try:
        return PERSONA_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""  ## No persona file: plain assistant

def build_system_prompt(memory: str | None = None) -> str:
    ## Rebuilt every turn, so edits to memory.md or persona.md take effect immediately.
    ## The persona is included in prompt_version(), so evals show when the personality changed.
    if memory is None:
        memory = memory_for_prompt()
    persona = load_persona()
    return ((f"{persona}\n\n" if persona else "") +
            f"{SYSTEM_PROMPT_INTRO}\n\n"
            f"Your memory of the user (use this to answer questions about them):\n{memory}\n\n"
            f"{SYSTEM_PROMPT}")

def remember(fact: str, category: str = "Facts") -> str:
    ## Add a bullet under the matching '## Category' heading in the memory file.
    fact = " ".join(fact.split())  ## Collapse newlines/extra spaces so each memory stays one bullet
    if not fact:
        return "Error: Nothing to remember."
    category = next((s for s in MEMORY_SECTIONS if s.lower() == category.strip().lower()), "Facts")

    lines = load_memory().splitlines()
    if any(fact.lower() in line.lower() for line in lines if line.startswith("- ")):
        return "Already remembered."

    bullet = f"- {fact} _({datetime.now():%Y-%m-%d})_"
    heading = f"## {category}"
    if heading not in lines:
        lines += ["", heading, ""]
    ## Insert after the last bullet in this section (or right after the heading if it's empty)
    insert_at = lines.index(heading) + 1
    for i in range(insert_at, len(lines)):
        if lines[i].startswith("## "):
            break
        if lines[i].startswith("- "):
            insert_at = i + 1
    if insert_at == lines.index(heading) + 1:
        lines.insert(insert_at, "")  ## Blank line between heading and first bullet
        insert_at += 1
    lines.insert(insert_at, bullet)

    MEMORY_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return f"Remembered under {category}: {fact}"

STOPWORDS = {"a", "an", "the", "is", "am", "are", "was", "my", "i", "me", "user", "users", "that",
             "to", "of", "and", "s", "it", "this", "anymore", "not", "no", "about"}

def _keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS

def forget(text: str) -> str:
    ## Remove the memory that best matches the given text.
    ## Small models reword things ("my favorite color is green" vs "Favorite color is green"),
    ## so match on shared keywords instead of an exact substring.
    needle = text.strip().lower()
    if not needle:
        return "Error: Say which memory to forget."
    lines = load_memory().splitlines()
    bullets = [l for l in lines if l.startswith("- ")]

    matches = [l for l in bullets if needle in l.lower()]  ## Exact substring wins if there is one
    if not matches:
        wanted = _keywords(text)
        scored = [(len(wanted & _keywords(DATE_SUFFIX.sub("", l))) / len(wanted), l)
                  for l in bullets] if wanted else []
        best = max((s for s, _ in scored), default=0)
        if best >= 0.6:  ## At least 60% of the keywords must appear in the memory
            matches = [l for s, l in scored if s == best]

    if not matches:
        current = "\n".join(f"- {b}" for b in memory_bullets()) or "(none)"
        return f"Error: No memory matches '{text}'. Current memories:\n{current}\nCall forget again with words copied from one of these."
    if len(matches) > 1:
        return "Error: That matches several memories, be more specific:\n" + "\n".join(matches)
    lines.remove(matches[0])
    MEMORY_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return f"Forgot: {DATE_SUFFIX.sub('', matches[0][2:])}"

client = OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=120, max_retries=0)  ## Create an OpenAI client with the local LLM server


## Tools the agent can use to read and write files in the notes directory
def list_notes() -> str:
    ## List all the files in the notes directory.
    if not NOTES_DIR.exists():
        return "Error: Notes directory does not exist."
    files = [f.name for f in NOTES_DIR.iterdir() if f.is_file()]
    return json.dumps(files, indent=2)

def safe_path(file_path: str) -> Path | None:
    ## Resolve a path and make sure it stays inside the notes directory (blocks '../' escapes).
    full_path = (NOTES_DIR / file_path).resolve()
    if not full_path.is_relative_to(NOTES_DIR.resolve()):
        return None
    return full_path

def read_file(file_path: str) -> str:
    ## Read the contents of a file in the notes directory.
    full_path = safe_path(file_path)
    if full_path is None:
        return f"Error: {file_path} is outside the notes directory. Access denied."
    if not full_path.exists():
        existing = [f.name for f in NOTES_DIR.iterdir() if f.is_file()] if NOTES_DIR.exists() else []
        return (f"Error: File {file_path} does not exist. Files that do exist: {existing or 'none'}. "
                "Tell the user the file was not found.")
    if not full_path.is_file():
        return f"Error: {file_path} is not a file."
    content = full_path.read_text(encoding="utf-8")
    if len(content) > MAX_FILE_CHARS:
        return f"Error: File {file_path} is too large to read (max {MAX_FILE_CHARS} characters)."
    return content

def write_file(file_path: str, content: str) -> str:
    ## Write the contents to a file in the notes directory.
    full_path = safe_path(file_path)
    if full_path is None:
        return f"Error: {file_path} is outside the notes directory. Access denied."
    if not full_path.parent.exists():
        return f"Error: Directory for file {file_path} does not exist."
    if len(content) > MAX_FILE_CHARS:
        return f"Error: Content is {len(content)} characters, over the {MAX_FILE_CHARS} limit. Write less."
    full_path.write_text(content, encoding="utf-8")
    return f"File {file_path} written successfully."

def current_time() -> str:
    ## Return the current time as a string.
    return datetime.now().isoformat()


## A dictionary of the tools the agent can use, mapping tool names to functions.
TOOLS_FUNCTIONS = {
    "list_notes": list_notes,
    "read_file": read_file,
    "write_file": write_file,
    "current_time": current_time,
    "remember": remember,
    "forget": forget,
}

## JSON schema for the tools the agent can use, to be used in the OpenAI API call.
TOOLS_SCHEMA = [
    
    {
        ## The schema for the list_notes function, which lists all the files in the notes directory.
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "List all the files in the notes directory.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        ## The schema for the read_file function, which reads the contents of a file in the notes directory.
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the notes directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        ## The schema for the write_file function, which writes the contents to a file in the notes directory.
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write the contents to a file in the notes directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Just the file name inside the notes folder, e.g. 'todo.txt'. Do not include 'notes/'."
                    },
                    "content": {
                        "type": "string",
                        "description": "The exact literal text to save. Plain text only: no expressions, no string concatenation with +, no placeholders. If you need the current time, call current_time first and paste the result as plain text."
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        ## The schema for the current_time function, which returns the current time as a string.
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Return the current time as a string.",
            "parameters": {"type": "object", "properties": {}}  ## Added to fix bug crashing: even tools with no arguments need this
        }
    },
    {
        ## The schema for the remember function, which saves a lasting memory to memory.md.
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a lasting preference, fact, or decision about the user to long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "One short sentence restating what the user told you about themselves."
                    },
                    "category": {
                        "type": "string",
                        "enum": MEMORY_SECTIONS,
                        "description": "Preferences = how the user likes things done. Facts = things about the user. Decisions = choices the user made."
                    }
                },
                "required": ["fact", "category"]
            }
        }
    },
    {
        ## The schema for the forget function, which removes a memory from memory.md.
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Remove a wrong or outdated memory from long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "A distinctive part of the memory's text, copied from the memory shown in your prompt."
                    }
                },
                "required": ["text"]
            }
        }
    }
]

## Loop to run the agent, taking user input and generating responses from the local LLM model.
def run_tool(name: str, raw_args:str ) -> str:
    ## Execute a tool by name with the given raw arguments.
    func = TOOLS_FUNCTIONS.get(name)
    if not func:
        return f"Error: Tool {name} not found."
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return f"Error: Invalid JSON arguments: {raw_args}"
    try:
        result = func(**args)
        return result
    except TypeError as e:  ## Must come before Exception, or it never gets caught
        return f"Error: Invalid arguments for tool {name}: {str(e)}"
    except Exception as e:
        return f"Error: Exception occurred while executing tool {name}: {str(e)}"

## Observability: every run produces a trace record, appended to logs/YYYY-MM-DD.jsonl
def prompt_version() -> str:
    ## Short fingerprint of the prompt + tools, so logs/evals show exactly which version produced a result.
    ## Built from an empty-memory system prompt, so layout changes in build_system_prompt count too.
    blob = build_system_prompt(memory="<memory>") + json.dumps(TOOLS_SCHEMA, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]

def check_tools_reach_model() -> str | None:
    ## Some GGUF chat templates silently drop the tools, so the model can't call them and starts
    ## inventing answers. Detect it: a prompt with tools must be much longer than one without.
    ## Returns None if fine, otherwise a description of the problem.
    msgs = [{"role": "system", "content": build_system_prompt(memory="(Nothing yet.)")},
            {"role": "user", "content": "What time is it? Use the current_time tool."}]
    try:
        bare = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=1, temperature=0)
        with_tools = client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS_SCHEMA,
                                                    tool_choice="auto", max_tokens=200, temperature=0)
    except Exception as e:
        return f"Tool request failed: {e}"
    if bare.usage and with_tools.usage:
        added = with_tools.usage.prompt_tokens - bare.usage.prompt_tokens
        if added < 100:
            return (f"The tool definitions are not reaching the model (they added only {added} prompt tokens). "
                    "The model's chat template is probably dropping them. Start llama-server with --jinja, "
                    "and/or use a GGUF whose chat template supports tool calling.")
    if not with_tools.choices[0].message.tool_calls:
        return "The tools are in the prompt, but the model didn't call current_time when asked to directly."
    return None

_server_model = None
def server_model() -> str:
    ## llama-server ignores the MODEL string and serves whatever it was started with,
    ## so ask the server what is actually loaded. That's the name that belongs in logs and evals.
    global _server_model
    if _server_model is None:
        try:
            _server_model = client.models.list().data[0].id
        except Exception:
            _server_model = "unknown"
    return _server_model

def write_log(record: dict) -> None:
    ## Append one run as a single JSON line. Logging must never crash the agent.
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR / f"{datetime.now():%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[warning: could not write log: {e}]")

def run_agent(user_input: str, history: list[dict], log: bool = True) -> str:
    ## Run one user turn and return the agent's reply. The full trace is logged.
    record = run_agent_traced(user_input, history)
    if log:
        write_log(record)
    return record["output"]

### Main function to run the local agent, taking user input and generating responses from the local LLM model.
def run_agent_traced(user_input: str, history: list[dict]) -> dict:
    ## history holds only the conversation; the system prompt (with fresh memory) is added per call.
    ## Returns a trace record: what happened at every step, for logs and evals.
    history.append({"role": "user", "content": user_input})
    started = time.perf_counter()
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "server_model": server_model(),
        "prompt_version": prompt_version(),
        "temperature": TEMPERATURE,
        "input": user_input,
        "tool_calls": [],       ## [{step, name, args, result, ms}]
        "invalid_tool_json": 0, ## How often the server rejected the model's tool call
        "repeated_calls": 0,    ## Identical tool calls blocked by the loop guard
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prompt_ms": 0,         ## Server time spent reading the prompt (from llama-server timings)
        "gen_ms": 0,            ## Server time spent generating tokens. Usually the bottleneck.
        "steps": 0,
        "stop_reason": None,    ## "answer" | "max_steps"
        "output": None,
        "latency_ms": None,
    }

    def finish(output: str, reason: str) -> dict:
        record.update(output=output, stop_reason=reason,
                      latency_ms=round((time.perf_counter() - started) * 1000))
        return record

    seen_calls = set()  ## (tool name, raw args) already executed this turn
    for step in range(MAX_STEPS):
        record["steps"] = step + 1
        ## Call the local LLM model with the system prompt, user input, and history.
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": build_system_prompt()}] + history,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                max_tokens=1024, ## Hard cap on how long any single model response can be
                temperature=TEMPERATURE,
            )
        except InternalServerError:
            ## The server could not parse the model's tool call, so tell the model and let it retry.
            record["invalid_tool_json"] += 1
            history.append({"role": "user", "content": "Your last tool call was invalid JSON. Retry using only literal strings (no + or expressions)."})
            continue

        if response.usage:
            record["prompt_tokens"] += response.usage.prompt_tokens or 0
            record["completion_tokens"] += response.usage.completion_tokens or 0
        timings = (response.model_extra or {}).get("timings") or {}  ## llama-server extra: where time went
        record["prompt_ms"] += round(timings.get("prompt_ms", 0))
        record["gen_ms"] += round(timings.get("predicted_ms", 0))

        ## Get the model's response message
        message = response.choices[0].message

        if not message.tool_calls:
            ## If the model did not call a tool, return its response as the final output.
            history.append({"role": "assistant", "content": message.content})
            return finish(message.content, "answer")

        ## If the model called a tool, add the assistant's tool call message to the history.
        history.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in message.tool_calls
            ],
        })

        ## Execute each tool call in the message and add the results to the history.
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            t0 = time.perf_counter()
            ## Loop guard: evals showed the model retrying an identical failing call until MAX_STEPS.
            call_key = (tool_name, tool_args)
            repeated = call_key in seen_calls
            if repeated:
                record["repeated_calls"] += 1
                tool_result = ("Error: You already made this exact call and got the result above. "
                               "Do not call it again. Answer the user now.")
            else:
                seen_calls.add(call_key)
                tool_result = run_tool(tool_name, tool_args)
            record["tool_calls"].append({
                "step": step + 1,
                "name": tool_name,
                "repeated": repeated,
                "args": tool_args,
                "result": tool_result[:500],  ## Truncated so logs stay readable
                "ms": round((time.perf_counter() - t0) * 1000),
            })
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": tool_result
            })

    ## If the maximum number of steps is reached, return a message indicating that the agent has stopped.
    return finish("Stopped after maximum steps reached. Please try again with a more specific request.", "max_steps")


## Main function to run the local agent in a loop, taking user input and generating responses from the local LLM model.
def main():
    NOTES_DIR.mkdir(exist_ok=True)  ## Create the notes directory if it doesn't exist
    load_memory()                   ## Create memory.md if it doesn't exist
    history = []
    print(f"{AGENT_NAME} is ready (server: {BASE_URL}). Notes folder: {NOTES_DIR}")
    print(f"Persona:     {PERSONA_FILE.resolve() if PERSONA_FILE.exists() else '(none: plain assistant)'}")
    print(f"Memory file: {MEMORY_FILE.resolve()}")
    print(f"Run logs:    {LOG_DIR.resolve()} (prompt version {prompt_version()})")
    print(f"Model:       {server_model()}")
    problem = check_tools_reach_model()
    if problem:
        print(f"\nWARNING: {problem}\n")
    print("Type 'quit' to exit, 'clear' to reset the conversation, 'memory' to show what I remember.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if user_input.lower() in ["quit", "exit"]:
            print("Exiting...")
            break
        if user_input.lower() == "clear":
            history = []
            print("History cleared (long-term memory kept).\n")
            continue
        if user_input.lower() == "memory":
            print(load_memory() + "\n")
            continue
        if not user_input:
            print("Please enter a valid input.\n")
            continue

        try:
            response = run_agent(user_input, history)
            print(f"{AGENT_NAME}: {response}\n")
        except KeyboardInterrupt:
            print("\n[Interrupted. Type 'quit' to exit.]\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    main()