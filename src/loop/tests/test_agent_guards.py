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


if __name__ == "__main__":
    unittest.main()
