"""
Tests for gmail_server.py against a fake mailbox: no Google account or network needed.
From src/loop:  python -m unittest discover -s tests -v
"""

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
LOOP = HERE.parent
sys.path.insert(0, str(LOOP))

import gmail_server  # noqa: E402
from gmail_server import extract_body, read_email, search_emails  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp_host import MCPHost  # noqa: E402


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def message(mid, subject, body_parts, sender="Palmtree <palmtree@example.com>"):
    return {"id": mid, "snippet": f"snippet {mid}", "payload": {
        "mimeType": "multipart/alternative",
        "headers": [{"name": "From", "value": sender}, {"name": "To", "value": "me@example.com"},
                    {"name": "Subject", "value": subject}, {"name": "Date", "value": "Fri, 20 Nov 2026"}],
        "parts": body_parts}}


class FakeGmail:
    def __init__(self, messages):
        self.messages = {m["id"]: m for m in messages}
        self.calls = []

    def get(self, path, **params):
        self.calls.append((path, params))
        if path == "/messages":
            ids = list(self.messages)[: params["maxResults"]]
            return {"messages": [{"id": i} for i in ids]} if ids else {}
        mid = path.rsplit("/", 1)[1]
        if mid not in self.messages:
            raise ToolError("No email with that id.")
        return self.messages[mid]


PLAIN = [{"mimeType": "text/plain", "body": {"data": b64("Lunch at noon?")}},
         {"mimeType": "text/html", "body": {"data": b64("<p>Lunch at <b>noon</b>?</p>")}}]
HTML_ONLY = [{"mimeType": "text/html", "body": {"data": b64(
    "<style>p{}</style><p>Your invoice</p><script>evil()</script><div>Total: $5</div>")}}]


class Tools(unittest.TestCase):
    def setUp(self):
        self.gmail = FakeGmail([message("abc123", "Lunch", PLAIN), message("def456", "Invoice", HTML_ONLY)])

    def test_search_returns_summaries(self):
        out = search_emails(self.gmail, "from:palmtree", 10)
        self.assertIn("id abc123", out)
        self.assertIn("subject: Invoice", out)
        self.assertEqual(self.gmail.calls[0], ("/messages", {"q": "from:palmtree", "maxResults": 10}))

    def test_search_caps_result_count(self):
        search_emails(self.gmail, "x", 500)
        self.assertEqual(self.gmail.calls[0][1]["maxResults"], gmail_server.MAX_RESULTS)
        before = len(self.gmail.calls)
        search_emails(self.gmail, "x", 0)
        self.assertEqual(self.gmail.calls[before][1]["maxResults"], 1)

    def test_search_with_no_matches(self):
        self.assertIn("No emails match", search_emails(FakeGmail([]), "nothing"))

    def test_read_prefers_plain_text(self):
        out = read_email(self.gmail, "abc123")
        self.assertIn("Subject: Lunch", out)
        self.assertIn("Lunch at noon?", out)
        self.assertNotIn("<b>", out)

    def test_read_falls_back_to_html_without_scripts(self):
        out = read_email(self.gmail, "def456")
        self.assertIn("Your invoice", out)
        self.assertIn("Total: $5", out)
        self.assertNotIn("evil", out)

    def test_read_rejects_non_ids(self):
        for bad in ["../../settings/filters", "abc 123", "", "a" * 100]:
            with self.assertRaises(ToolError, msg=bad):
                read_email(self.gmail, bad)
        self.assertEqual(self.gmail.calls, [])  ## Nothing was sent to Gmail

    def test_long_bodies_are_cut(self):
        long = FakeGmail([message("big1", "Big", [{"mimeType": "text/plain", "body": {"data": b64("y" * 20000)}}])])
        out = read_email(long, "big1")
        self.assertIn("cut off", out)
        self.assertLess(len(out), gmail_server.MAX_BODY_CHARS + 500)

    def test_nested_parts(self):
        nested = {"mimeType": "multipart/mixed", "parts": [{"mimeType": "multipart/alternative", "parts": PLAIN}]}
        self.assertEqual(extract_body(nested), "Lunch at noon?")
        self.assertIn("No readable text", extract_body({"mimeType": "image/png"}))


class Credentials(unittest.TestCase):
    def test_only_readonly_scope_is_requested(self):
        self.assertEqual(gmail_server.SCOPES, ["https://www.googleapis.com/auth/gmail.readonly"])

    def test_token_with_extra_scopes_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = Path(tmp) / "gmail_token.json"
            token.write_text(json.dumps({"scopes": gmail_server.SCOPES + [
                "https://www.googleapis.com/auth/gmail.send"]}))
            with self.assertRaisesRegex(RuntimeError, "not only gmail.readonly"):
                gmail_server.load_credentials(Path(tmp))

    def test_unauthorized_server_fails_at_startup_with_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "servers.log"
            spec = {"command": sys.executable, "tools": {"search_emails": {}},
                    "args": [str(LOOP / "gmail_server.py"), "--secrets-dir", str(Path(tmp) / "secrets")]}
            host = MCPHost({"gmail": spec}, errlog_path=log, startup_timeout=30)
            try:
                problems = host.start()
            finally:
                host.close()
            self.assertEqual(len(problems), 1)
            self.assertIn("gmail", problems[0])
            self.assertIn("--authorize", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
