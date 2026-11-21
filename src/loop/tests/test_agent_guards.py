"""
asks_to_remember(): when it's true, the first model call must be a tool call (see run_agent_traced).

    python -m unittest tests.test_agent_guards -v     (from src/loop)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402


class AsksToRemember(unittest.TestCase):
    def test_plain_requests(self):
        for text in ["Remember that I'm allergic to peanuts.",
                     "remember: my sister is called Palmtree",
                     "Please remember that I prefer tea.",
                     "Just so you know, I live in Portland. Remember that.",
                     "From now on I want bullet points, and remember it.",
                     "Is it raining? Remember I hate umbrellas."]:
            self.assertTrue(agent.asks_to_remember(text), text)

    def test_not_requests(self):
        for text in ["Do you remember my name?",
                     "I can't remember where my keys are.",
                     "Remembering things is hard.",
                     "What do you remember about me?",
                     "Hi Coco!",
                     "Tell me about the movie Remember Me"]:
            self.assertFalse(agent.asks_to_remember(text), text)


class FakeHost:
    def __init__(self):
        self.calls = []

    def tool_schemas(self):
        return [{"type": "function", "function": {"name": n, "description": "", "parameters": {"type": "object"}}}
                for n in ("remember", "list_notes")]

    def call(self, name, args):
        from mcp_host import CallResult
        self.calls.append((name, args))
        return CallResult("Saved.", True, "personal", True)


class RememberGuard(unittest.TestCase):
    ## The model claims "got it" without calling remember: that reply must never reach the user.
    def setUp(self):
        self.saved = (agent.host, agent.call_model, agent.server_model)
        agent.host = FakeHost()
        agent.server_model = lambda: "fake"

    def tearDown(self):
        agent.host, agent.call_model, agent.server_model = self.saved

    def script(self, *replies):
        sent = []
        def fake_call_model(messages, tools, on_text=None, tool_choice="auto"):
            sent.append((list(messages), on_text))
            reply = replies[len(sent) - 1]
            if on_text and reply.content:
                on_text(reply.content)
            return reply
        agent.call_model = fake_call_model
        return sent

    def say(self, text):
        return agent.ModelReply(content=text, tool_calls=[])

    def remember(self):
        return agent.ModelReply(content=None, tool_calls=[
            {"id": "c1", "name": "remember", "arguments": '{"fact": "Lives in Portland"}'}])

    def test_claim_without_saving_is_retried_and_hidden(self):
        sent = self.script(self.say("Got it, Portland! (remembered) ✅"), self.remember(), self.say("Saved: Portland."))
        history, shown = [{"role": "system", "content": "sys"}], []
        record = agent.run_agent_traced("I live in Portland. Remember that.", history, on_text=shown.append)
        self.assertEqual(record["output"], "Saved: Portland.")
        self.assertEqual(record["guard_retries"], 1)
        self.assertEqual([c[0] for c in agent.host.calls], ["remember"])
        self.assertEqual(shown, ["Saved: Portland."])            ## The false claim was never shown
        self.assertIsNone(sent[0][1])                             ## Held back while unsaved...
        self.assertIn("nothing has been saved", sent[1][0][-1]["content"])  ## ...retry carries the note
        self.assertFalse(any("nothing has been saved" in str(m.get("content")) for m in history))
        self.assertFalse(any("(remembered)" in str(m.get("content")) for m in history))

    def test_gives_up_after_two_retries(self):
        self.script(self.say("Got it."), self.say("Got it!"), self.say("Noted."))
        record = agent.run_agent_traced("Remember that I like tea.", [{"role": "system", "content": "sys"}])
        self.assertEqual((record["output"], record["guard_retries"]), ("Noted.", 2))

    def test_normal_messages_are_not_held(self):
        sent = self.script(self.say("Hi Palmtree!"))
        shown = []
        record = agent.run_agent_traced("Hi Coco", [{"role": "system", "content": "sys"}], on_text=shown.append)
        self.assertFalse(record["remember_guard"])
        self.assertEqual(shown, ["Hi Palmtree!"])
        self.assertIsNotNone(sent[0][1])


if __name__ == "__main__":
    unittest.main()
