# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Regression tests for streaming data returned by tools."""
import base64
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from agentscope.agent import Agent
from agentscope.event import (
    ToolResultDataDeltaEvent,
    ToolResultTextDeltaEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
)
from agentscope.message import (
    AssistantMsg,
    Base64Source,
    DataBlock,
    ToolResultState,
    URLSource,
)


class ToolDataStreamTest(IsolatedAsyncioTestCase):
    """Tool data events preserve source block identity and payloads."""

    async def test_data_chunks_keep_identity_and_merge_in_message(
        self,
    ) -> None:
        """Chunks with one block id remain one block after event replay."""
        agent = SimpleNamespace(
            state=SimpleNamespace(reply_id="reply-1"),
        )
        chunks = [
            DataBlock(
                id="audio-1",
                source=Base64Source(
                    data=base64.b64encode(payload).decode("ascii"),
                    media_type="audio/wav",
                ),
            )
            for payload in (b"hello", b"world")
        ]
        reply = AssistantMsg(id="reply-1", name="agent", content=[])
        reply.append_event(
            ToolResultStartEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                tool_call_name="stream_audio",
            ),
        )

        events = []
        for chunk in chunks:
            async for event in Agent._convert_tool_chunk_to_event(
                agent,
                "tool-1",
                [chunk],
            ):
                events.append(event)
                reply.append_event(event)
        reply.append_event(
            ToolResultEndEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                state=ToolResultState.SUCCESS,
            ),
        )

        results = list(reply.get_content_blocks("tool_result"))
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(
            {
                "event_block_ids": [event.block_id for event in events],
                "result_block_ids": [block.id for block in result.output],
                "result_payloads": [
                    base64.b64decode(block.source.data)
                    for block in result.output
                ],
            },
            {
                "event_block_ids": ["audio-1", "audio-1"],
                "result_block_ids": ["audio-1"],
                "result_payloads": [b"helloworld"],
            },
        )

    async def test_url_data_block_keeps_identity(self) -> None:
        """A one-shot URL result preserves the tool block identity."""
        agent = SimpleNamespace(
            state=SimpleNamespace(reply_id="reply-1"),
        )
        block = DataBlock(
            id="file-1",
            source=URLSource(
                url="https://example.com/result.bin",
                media_type="application/octet-stream",
            ),
        )

        events = [
            event
            async for event in Agent._convert_tool_chunk_to_event(
                agent,
                "tool-1",
                [block],
            )
        ]

        self.assertEqual(
            [
                {
                    "block_id": event.block_id,
                    "url": str(event.url),
                    "media_type": event.media_type,
                }
                for event in events
            ],
            [
                {
                    "block_id": "file-1",
                    "url": "https://example.com/result.bin",
                    "media_type": "application/octet-stream",
                },
            ],
        )

    async def test_nonstandard_base64_chunks_keep_compatibility(
        self,
    ) -> None:
        """Placeholder payloads retain the existing string fallback."""
        reply = AssistantMsg(id="reply-1", name="agent", content=[])
        reply.append_event(
            ToolResultStartEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                tool_call_name="stream_data",
            ),
        )
        for data in ("a", "A==", "Yg=="):
            reply.append_event(
                ToolResultDataDeltaEvent(
                    reply_id="reply-1",
                    tool_call_id="tool-1",
                    block_id="data-1",
                    data=data,
                    media_type="application/octet-stream",
                ),
            )

        results = list(reply.get_content_blocks("tool_result"))
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].output), 3)
        reply.append_event(
            ToolResultEndEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                state=ToolResultState.SUCCESS,
            ),
        )
        output = results[0].output
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].source.data, "aGI=")

    async def test_merging_data_chunks_preserves_block_order(self) -> None:
        """Removing repeated data chunks also normalizes adjacent text."""
        reply = AssistantMsg(id="reply-1", name="agent", content=[])
        reply.append_event(
            ToolResultStartEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                tool_call_name="stream_data",
            ),
        )
        first = base64.b64encode(b"hello").decode("ascii")
        second = base64.b64encode(b"world").decode("ascii")
        for event in (
            ToolResultDataDeltaEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                block_id="data-1",
                data=first,
                media_type="application/octet-stream",
            ),
            ToolResultTextDeltaEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                delta="middle",
            ),
            ToolResultDataDeltaEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                block_id="data-1",
                data=second,
                media_type="application/octet-stream",
            ),
            ToolResultTextDeltaEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                delta="-end",
            ),
            ToolResultEndEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                state=ToolResultState.SUCCESS,
            ),
        ):
            reply.append_event(event)

        result = list(reply.get_content_blocks("tool_result"))[0]
        self.assertEqual(
            [
                (
                    block.id,
                    base64.b64decode(block.source.data),
                )
                if isinstance(block, DataBlock)
                else (block.id, block.text)
                for block in result.output
            ],
            [
                ("data-1", b"helloworld"),
                (result.output[1].id, "middle-end"),
            ],
        )
