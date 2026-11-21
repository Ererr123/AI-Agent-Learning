"""
Streaming: call_model must give the same result streamed as not streamed.
Uses a fake OpenAI client that returns hand-made chunks, so no model server is needed.

    python -m unittest tests.test_streaming -v     (from src/loop)
"""
import sys
import unittest
from pathlib import Path

import httpx2 as httpx  ## The HTTP library the openai package uses
from openai import APIError, APIConnectionError
from openai.types.chat import ChatCompletionChunk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402


def chunk(delta=None, usage=None, timings=None, finish=None):
    data = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        data["usage"] = usage
    if timings:
        data["timings"] = timings  ## llama-server's extra field
    return ChatCompletionChunk.model_validate(data)


class FakeClient:
    def __init__(self, items):
        self.items, self.kwargs = items, None
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        def gen():
            for item in self.items:
                if isinstance(item, Exception):
                    raise item
                yield item
        return gen()


class CallModelStreaming(unittest.TestCase):
    def setUp(self):
        self.real_client = agent.client

    def tearDown(self):
        agent.client = self.real_client

    def run_stream(self, items):
        agent.client = FakeClient(items)
        pieces = []
        reply = agent.call_model([{"role": "user", "content": "hi"}], [], on_text=pieces.append)
        self.assertTrue(agent.client.kwargs["stream"])
        return reply, pieces

    def test_text_is_passed_on_as_it_arrives(self):
        reply, pieces = self.run_stream([
            chunk({"role": "assistant", "content": "Hel"}),
            chunk({"content": "lo, Palmtree"}),
            chunk({}, finish="stop"),
            chunk(usage={"prompt_tokens": 50, "completion_tokens": 4, "total_tokens": 54},
                  timings={"prompt_n": 3, "cache_n": 47, "prompt_ms": 12.5, "predicted_n": 4, "predicted_ms": 200}),
        ])
        self.assertEqual(pieces, ["Hel", "lo, Palmtree"])
        self.assertEqual(reply.content, "Hello, Palmtree")
        self.assertEqual(reply.tool_calls, [])
        self.assertEqual((reply.prompt_tokens, reply.completion_tokens), (50, 4))
        self.assertEqual(reply.timings["cache_n"], 47)

    def test_tool_calls_are_assembled_from_pieces(self):
        reply, pieces = self.run_stream([
            chunk({"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path": '}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"todo.txt"}'}}]}),
            chunk({"tool_calls": [
                {"index": 1, "id": "call_b", "type": "function", "function": {"name": "current_time", "arguments": "{}"}}]}),
            chunk({}, finish="tool_calls"),
        ])
        self.assertEqual(pieces, [])
        self.assertIsNone(reply.content)
        self.assertEqual(reply.tool_calls, [
            {"id": "call_a", "name": "read_file", "arguments": '{"path": "todo.txt"}'},
            {"id": "call_b", "name": "current_time", "arguments": "{}"},
        ])

    def test_missing_tool_call_id_gets_one(self):
        reply, _ = self.run_stream([
            chunk({"tool_calls": [{"index": 0, "type": "function", "function": {"name": "list_notes", "arguments": "{}"}}]}),
        ])
        self.assertEqual(reply.tool_calls[0]["id"], "call_0")

    def test_error_inside_the_stream_is_an_invalid_tool_call(self):
        request = httpx.Request("POST", "http://localhost/v1/chat/completions")
        with self.assertRaises(agent.InvalidToolCall):
            self.run_stream([chunk({"content": "Let me"}),
                             APIError("Failed to parse tool call", request, body=None)])

    def test_connection_errors_are_not_swallowed(self):
        request = httpx.Request("POST", "http://localhost/v1/chat/completions")
        with self.assertRaises(APIConnectionError):
            self.run_stream([APIConnectionError(request=request)])


class ReplyPrinterOutput(unittest.TestCase):
    def test_streamed_reply_is_not_printed_twice(self):
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            p = agent.ReplyPrinter()
            p("Hi ")
            p("there")
            p.finish("Hi there")
        self.assertEqual(out.getvalue(), f"{agent.AGENT_NAME}: Hi there\n\n")

    def test_unstreamed_reply_is_printed_whole(self):
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            agent.ReplyPrinter().finish("Stopped after maximum steps reached.")
        self.assertEqual(out.getvalue(), f"{agent.AGENT_NAME}: Stopped after maximum steps reached.\n\n")


if __name__ == "__main__":
    unittest.main()
