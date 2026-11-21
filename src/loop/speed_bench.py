"""
Speed benchmark: one realistic conversation against the live model server, timed per turn.

Like a real chat session it runs the startup check first, then a fixed list of turns in ONE
conversation, so you can see whether llama-server reuses its prompt cache between steps and turns.
Notes and memory live in a throwaway sandbox (your real persona.md is used, so prompt size is real).

    python speed_bench.py               # sandbox turns only (notes, memory, chit-chat)
    python speed_bench.py --personal    # also a Gmail search and a GitHub repo list (needs
                                        # mcp_servers.local.json). Their replies are never printed or saved.

Results go to speed_results/<timestamp>.json and are compared with the previous run of the same model.
Answer quality is not checked here: that's what evals.py is for. Run both after a speed change.
"""
import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import agent
import memory_store

RESULTS_DIR = Path(__file__).parent / "speed_results"

NOTES = {
    "shopping.txt": "Shopping list:\n- oat milk\n- coffee beans\n- bananas\n- dish soap\n",
    "ideas.md": "# Ideas\n- Build a weather tool for Coco\n- Learn about vector databases\n",
}
MEMORIES = [("The user's name is Palmtree", "About the user"),
            ("Palmtree prefers short answers", "Preferences")]

## (label, user turn, private) in one conversation. Private turns use personal services.
TURNS = [
    ("greeting", "Hi Coco!", False),
    ("memory answer", "What's my name?", False),
    ("notes list+read", "What notes do I have? Read me the shopping one.", False),
    ("remember", "Remember that my favorite color is green.", False),
    ("follow-up", "What was the first item on that shopping list?", False),
]
PERSONAL_TURNS = [
    ("gmail search", "Search my email for Quantum Sky and tell me what the latest one is about.", True),
    ("github repos", "What GitHub repositories do I have?", True),
]


def summarize(trace: dict) -> dict:
    calls = trace["llm_calls"]
    return {
        "wall_s": round(trace["latency_ms"] / 1000, 1),
        ## When the first words appeared (streaming, like the real chat). None if nothing streamed.
        "first_text_s": round(trace["first_text_ms"] / 1000, 1) if trace.get("first_text_ms") is not None else None,
        "steps": trace["steps"],
        "read_tokens": sum(c["prompt_n"] for c in calls),     ## Actually read by the server
        "cached_tokens": sum(c["cache_n"] for c in calls),    ## Reused from the cache
        "prompt_s": round(trace["prompt_ms"] / 1000, 1),
        "gen_tokens": trace["completion_tokens"],
        "gen_s": round(trace["gen_ms"] / 1000, 1),
        "tools": [t["name"] for t in trace["tool_calls"]],
        ## A call that read more than it reused (and read a lot) re-read the conversation: a cache miss.
        ## New content is legitimately read (a long email result), but it's never most of the prompt.
        "misses": sum(1 for c in calls if c["prompt_n"] > 1000 and c["cache_n"] < c["prompt_n"]),
    }


def previous(model: str) -> dict | None:
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("server_model") == model:
            return data
    return None


def main():
    parser = argparse.ArgumentParser(description="Time one realistic Coco conversation.")
    parser.add_argument("--personal", action="store_true", help="Add Gmail and GitHub turns (uses mcp_servers.local.json)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Default 0, so reply lengths (and so timings) are comparable between runs")
    parser.add_argument("--label", default="", help="Note saved with the results, e.g. 'np1 + frozen memory'")
    args = parser.parse_args()

    agent.TEMPERATURE = args.temperature
    if not args.personal:
        agent.MCP_LOCAL_CONFIG = None
    turns = TURNS + (PERSONAL_TURNS if args.personal else [])

    try:
        model = agent.server_model()
        agent.client.models.list()
    except Exception as e:
        sys.exit(f"Can't reach the model server at {agent.BASE_URL}: {e}")

    rows, startup_s, prompt_version = [], None, None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        agent.NOTES_DIR, agent.MEMORY_FILE = root / "notes", root / "memory.md"
        agent.NOTES_DIR.mkdir()
        for name, content in NOTES.items():
            (agent.NOTES_DIR / name).write_text(content, encoding="utf-8")
        agent.load_memory()
        for fact, category in MEMORIES:
            memory_store.remember(agent.MEMORY_FILE, fact, category)
        try:
            problems = agent.start_tools()
            if problems:
                sys.exit("MCP servers did not start cleanly:\n  " + "\n  ".join(problems))
            t0 = time.perf_counter()
            problem = agent.check_tools_reach_model()  ## Same startup check a real chat runs
            startup_s = round(time.perf_counter() - t0, 1)
            if problem:
                print(f"WARNING (startup check): {problem}")
            history = []
            prompt_version = agent.prompt_version()  ## Needs the running tools, so take it now
            print(f"Model: {model} | prompt {prompt_version} | temperature {agent.TEMPERATURE}")
            print(f"Startup check: {startup_s}s\n")
            print(f"{'turn':16} {'wall':>6} {'first':>6} {'steps':>5} {'read':>6} {'cached':>7} {'prompt':>7} {'gen':>12}  tools")
            for label, turn, private in turns:
                trace = agent.run_agent_traced(turn, history, on_text=lambda piece: None)  ## Stream, like chat
                row = {"label": label, **summarize(trace)}
                if not private:
                    row["reply"] = (trace["output"] or "")[:200]
                rows.append(row)
                gen_rate = f"{row['gen_tokens'] / row['gen_s']:.0f}t/s" if row["gen_s"] else "-"
                miss = "  CACHE MISS" if row["misses"] else ""
                first = f"{row['first_text_s']:5.1f}s" if row["first_text_s"] is not None else "     -"
                print(f"{label:16} {row['wall_s']:5.1f}s {first} {row['steps']:5} {row['read_tokens']:6} "
                      f"{row['cached_tokens']:7} {row['prompt_s']:6.1f}s {row['gen_s']:5.1f}s {gen_rate:>6}  "
                      f"{', '.join(row['tools']) or '-'}{miss}")
        finally:
            agent.stop_tools()

    total = round(sum(r["wall_s"] for r in rows), 1)
    print(f"\nTotal: {total}s over {len(rows)} turns (+ {startup_s}s startup check)")

    prev = previous(model)
    if prev:
        by_label = {r["label"]: r for r in prev["turns"]}
        print(f"\nVs. {prev['timestamp']} {prev.get('label') or ''}".rstrip() + ":")
        for r in rows:
            old = by_label.get(r["label"])
            if old:
                print(f"  {r['label']:16} {old['wall_s']:6.1f}s -> {r['wall_s']:6.1f}s")
        same = [r for r in rows if r["label"] in by_label]
        if same:
            old_total = sum(by_label[r["label"]]["wall_s"] for r in same)
            new_total = sum(r["wall_s"] for r in same)
            print(f"  {'total':16} {old_total:6.1f}s -> {new_total:6.1f}s")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now()
    out = {"timestamp": stamp.isoformat(timespec="seconds"), "label": args.label, "server_model": model,
           "prompt_version": prompt_version, "temperature": agent.TEMPERATURE,
           "personal": args.personal, "startup_s": startup_s, "total_s": total, "turns": rows}
    out_file = RESULTS_DIR / f"{stamp:%Y-%m-%d_%H%M%S}.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
