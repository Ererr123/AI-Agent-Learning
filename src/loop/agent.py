"""
Local Agent for the local LLM model.

Setup:
    pip install openai
    llama-server -hf <your-gguf-repo>:Q4_K_M --jinja -c 16384 --port 8080
    python agent.py
"""

import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from openai import OpenAI, InternalServerError

## Our planned Configuration for the local agent
BASE_URL = "http://localhost:8080/v1"  ## The base URL for the local LLM server
MODEL = "ChocoBanana"     ## The model name to use for the local LLM server
NOTES_DIR = Path("notes") ## the only folder the agent can read/write to
MAX_STEPS = 10             ## The maximum number of steps the agent can take
MAX_FILE_CHARS = 40000      ## Keep our AI small and fast by limiting the number of characters it can read from a file

SYSTEM_PROMPT = (
    "You are a personal assistant that can read through files in a local directory and "
    "write notes to them. You can use whatever tools you need to accomplish your goals, "
    "but you must always be careful to stay within the notes directory. "
    "Do not gueess or make up information. Keep your responses concise and to the point. "
)

client = OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=120, max_retries=0)  ## Create an OpenAI client with the local LLM server


## Tools the agent can use to read and write files in the notes directory
def list_notes() -> str:
    ## List all the files in the notes directory.
    if not NOTES_DIR.exists():
        return "Error: Notes directory does not exist."
    files = [f.name for f in NOTES_DIR.iterdir() if f.is_file()]
    return json.dumps(files, indent=2)

def read_file(file_path: str) -> str:
    ## Read the contents of a file in the notes directory.
    full_path = NOTES_DIR / file_path
    if not full_path.exists():
        return f"Error: File {file_path} does not exist."
    if not full_path.is_file():
        return f"Error: {file_path} is not a file."
    content = full_path.read_text(encoding="utf-8")
    if len(content) > MAX_FILE_CHARS:
        return f"Error: File {file_path} is too large to read (max {MAX_FILE_CHARS} characters)."
    return content

def write_file(file_path: str, content: str) -> str:
    ## Write the contents to a file in the notes directory.
    full_path = NOTES_DIR / file_path
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
    except Exception as e:
        return f"Error: Exception occurred while executing tool {name}: {str(e)}"
    except TypeError as e:
        return f"Error: Invalid arguments for tool {name}: {str(e)}"

### Main function to run the local agent, taking user input and generating responses from the local LLM model.
def run_agent(user_input: str, history: list[dict] = []) -> str:
    history.append({"role": "user", "content": user_input})

    for step in range(MAX_STEPS):
        ## Call the local LLM model with the system prompt, user input, and history.
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                 max_tokens=1024, ## Hard cap on how long any single model response can be
                temperature=0.9,
            )
        except InternalServerError:
            ## The server could not parse the model's tool call, so tell the model and let it retry.
            history.append({"role": "user", "content": "Your last tool call was invalid JSON. Retry using only literal strings (no + or expressions)."})
            continue

        ## Get the model's response message
        message = response.choices[0].message

        if not message.tool_calls:
            ## If the model did not call a tool, return its response as the final output.
            history.append({"role": "assistant", "content": message.content})
            return message.content

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
            tool_result = run_tool(tool_name, tool_args)
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": tool_result
            })

    ## If the maximum number of steps is reached, return a message indicating that the agent has stopped.
    return "Stopped after maximum steps reached. Please try again with a more specific request."


## Main function to run the local agent in a loop, taking user input and generating responses from the local LLM model.
def main():
    NOTES_DIR.mkdir(exist_ok=True)  ## Create the notes directory if it doesn't exist
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Local agent ready (server: {BASE_URL}). Notes folder: {NOTES_DIR}")
    print("Type 'quit' to exit, 'clear' to reset the conversation.\n")

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
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("History cleared.\n")
            continue
        if not user_input:
            print("Please enter a valid input.\n")
            continue

        try:
            response = run_agent(user_input, history)
            print(f"Agent: {response}\n")
        except KeyboardInterrupt:
            print("\n[Interrupted. Type 'quit' to exit.]\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    main()