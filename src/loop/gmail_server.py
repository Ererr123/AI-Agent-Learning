"""
Coco's read-only Gmail MCP server.

It asks Google only for the gmail.readonly scope, so even a bug here cannot send, draft,
label, archive or delete mail: Google refuses. (mcp_host.py labels its results as
untrusted content, via "untrusted_content" in mcp_servers.json.) It offers two tools:

    search_emails  search the mailbox with Gmail search syntax; returns short summaries
    read_email     read one message (headers + plain-text body, size-capped)

Credentials live OUTSIDE the repository, in the secrets folder (default %APPDATA%\\coco):
    google_client_secret.json   OAuth client you download from Google Cloud (Desktop app)
    gmail_token.json            created by --authorize; delete it to disconnect this computer

One-time setup (opens your browser for Google's consent screen):
    python gmail_server.py --authorize

Normal use: mcp_host.py starts it over stdio (see mcp_servers.json). It never opens a
browser itself; if the token is missing or revoked it exits with instructions instead.
"""

import argparse
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]  ## The ONLY scope requested
API = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_RESULTS = 20          ## Most summaries one search can return
MAX_BODY_CHARS = 6000     ## Longer bodies are cut off (keeps the model's prompt small)
MESSAGE_ID = re.compile(r"^[A-Za-z0-9]{1,64}$")  ## Gmail ids are hex-like; blocks URL tricks


def default_secrets_dir() -> Path:
    return Path(os.environ.get("APPDATA") or Path.home()) / "coco"


## ---------- Talking to Gmail ----------

class Gmail:
    """Thin read-only REST client. Tests swap in a fake with the same get() method."""

    def __init__(self, session):
        self.session = session  ## A google.auth AuthorizedSession: refreshes the token itself

    def get(self, path: str, **params) -> dict:
        import requests
        try:
            response = self.session.get(API + path, params=params, timeout=20)
        except requests.RequestException as e:
            raise ToolError(f"Could not reach Gmail: {type(e).__name__}") from None
        if response.status_code in (401, 403):
            raise ToolError("Gmail refused the request (access revoked or expired). "
                            "Run: python gmail_server.py --authorize")
        if response.status_code == 404:
            raise ToolError("No email with that id.")
        if not response.ok:
            raise ToolError(f"Gmail returned HTTP {response.status_code}.")
        return response.json()


## ---------- Turning messages into text ----------

def header(message: dict, name: str) -> str:
    for h in message.get("payload", {}).get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("br", "p", "div", "tr", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextOnly()
    parser.feed(html)
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)).strip()


def _decode(data: str) -> str:
    import base64
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def extract_body(payload: dict) -> str:
    ## Prefer the text/plain part anywhere in the MIME tree; fall back to text/html.
    def find(part, mime):
        if part.get("mimeType") == mime and part.get("body", {}).get("data"):
            return _decode(part["body"]["data"])
        for child in part.get("parts", []) or []:
            found = find(child, mime)
            if found is not None:
                return found
        return None

    plain = find(payload, "text/plain")
    if plain is not None:
        return plain.strip()
    html = find(payload, "text/html")
    return html_to_text(html) if html is not None else "(No readable text in this email.)"


## ---------- The tools (plain functions, so they can be tested without a server) ----------

def search_emails(gmail, query: str, max_results: int = 10) -> str:
    max_results = max(1, min(int(max_results), MAX_RESULTS))
    listing = gmail.get("/messages", q=query, maxResults=max_results)
    ids = [m["id"] for m in listing.get("messages", [])][:max_results]
    if not ids:
        hint = (" Operators like from:/subject: narrow the search a lot: try again with plain keywords."
                if ":" in query else " Try fewer or different keywords.")
        return f"No emails match '{query}'.{hint}"
    lines = []
    for message_id in ids:
        m = gmail.get(f"/messages/{message_id}", format="metadata",
                      metadataHeaders=["From", "Subject", "Date"])
        lines.append(f"- id {message_id} | {header(m, 'Date')} | from {header(m, 'From')} | "
                     f"subject: {header(m, 'Subject') or '(none)'} | {m.get('snippet', '')}")
    return "\n".join(lines)


def read_email(gmail, message_id: str) -> str:
    message_id = message_id.strip()
    if not MESSAGE_ID.match(message_id):
        raise ToolError("That is not a Gmail message id. Use an id from search_emails.")
    m = gmail.get(f"/messages/{message_id}", format="full")
    body = extract_body(m.get("payload", {}))
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + f"\n[...email cut off at {MAX_BODY_CHARS} characters...]"
    return "\n".join([
        f"From: {header(m, 'From')}",
        f"To: {header(m, 'To')}",
        f"Date: {header(m, 'Date')}",
        f"Subject: {header(m, 'Subject')}",
        "",
        body,
    ])


def build_server(gmail) -> MCPServer:
    ## log_level WARNING: the SDK logs failed calls with their text at INFO
    server = MCPServer("coco-gmail", version="1.0.0", log_level="WARNING")

    @server.tool(name="search_emails", description=(
        "Search the user's Gmail. Use plain keywords, e.g. 'Quantum Sky': they match the sender, "
        "subject AND body. Only add operators (from:, subject:, newer_than:7d, is:unread) when the "
        "user asks for them. Returns ids, senders, dates, subjects and snippets, newest first."))
    def _search(query: str, max_results: int = 10) -> str:
        return search_emails(gmail, query, max_results)

    @server.tool(name="read_email", description="Read one email by the id that search_emails returned.")
    def _read(message_id: str) -> str:
        return read_email(gmail, message_id)

    return server


## ---------- Credentials ----------

def authorize(secrets_dir: Path) -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow
    client_file = secrets_dir / "google_client_secret.json"
    if not client_file.exists():
        sys.exit(f"Missing {client_file}\nDownload your OAuth client (type: Desktop app) from Google Cloud "
                 "and save it there. See docs/mcp-setup.md (Gmail).")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    granted = set(creds.granted_scopes or creds.scopes or [])
    if granted != set(SCOPES):
        sys.exit(f"Refusing to save: Google granted {sorted(granted)}, expected only {SCOPES}.")
    token_file = secrets_dir / "gmail_token.json"
    token_file.write_text(creds.to_json(), encoding="utf-8")
    print(f"Saved read-only Gmail access to {token_file}. Delete that file to disconnect.")


def load_credentials(secrets_dir: Path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    token_file = secrets_dir / "gmail_token.json"
    if not token_file.exists():
        raise RuntimeError(f"Gmail is not authorized on this computer ({token_file} not found). "
                           "Run: python gmail_server.py --authorize")
    stored = set(json.loads(token_file.read_text(encoding="utf-8")).get("scopes", []))
    if stored != set(SCOPES):
        raise RuntimeError(f"{token_file} grants {sorted(stored)}, not only gmail.readonly. "
                           "Delete it and run: python gmail_server.py --authorize")
    creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds.valid:
        if not creds.refresh_token:
            raise RuntimeError("Gmail token has expired. Run: python gmail_server.py --authorize")
        try:
            creds.refresh(Request())
        except Exception as e:
            raise RuntimeError(f"Could not refresh Gmail access ({type(e).__name__}). "
                               "Run: python gmail_server.py --authorize") from None
        token_file.write_text(creds.to_json(), encoding="utf-8")
    return creds


def main():
    parser = argparse.ArgumentParser(description="Coco's read-only Gmail MCP server.")
    parser.add_argument("--authorize", action="store_true", help="One-time browser sign-in, then exit")
    parser.add_argument("--secrets-dir", type=Path, default=default_secrets_dir(),
                        help="Folder holding the OAuth client and token (default: %%APPDATA%%\\coco)")
    args = parser.parse_args()
    args.secrets_dir.mkdir(parents=True, exist_ok=True)

    if args.authorize:
        authorize(args.secrets_dir)
        return
    try:
        creds = load_credentials(args.secrets_dir)
    except RuntimeError as e:
        sys.exit(str(e))  ## Goes to logs/mcp-servers.log; the host reports the server as not started
    from google.auth.transport.requests import AuthorizedSession
    build_server(Gmail(AuthorizedSession(creds))).run("stdio")


if __name__ == "__main__":
    main()
