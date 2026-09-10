# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Regression tests for streaming data returned by tools."""
import base64
from types import SimpleNamespace
from typing import Any, AsyncGenerator
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
    TextBlock,
    ToolCallBlock,
    ToolResultState,
    URLSource,
)
from agentscope.permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionDecision,
)
from agentscope.state import AgentState
from agentscope.tool import ToolBase, ToolChunk, Toolkit, ToolResponse


class CrossTypeIdTool(ToolBase):
    """Yield text and data blocks that intentionally share an ID."""

    name: str = "cross_type_id"
    description: str = "Exercise cross-type block ID normalization."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"data_first": {"type": "boolean"}},
    }
    is_concurrency_safe: bool = True
    is_read_only: bool = True
    is_external_tool: bool = False
    is_mcp: bool = False

    async def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        """Allow the test tool call."""
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            decision_reason="Test tool always allows",
            message="Test tool always allows",
        )

    async def __call__(
        self,
        data_first: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[ToolChunk, None]:
        """Yield blocks with a cross-type ID conflict."""
        blocks = [
            DataBlock(
                id="same",
                source=Base64Source(
                    data=base64.b64encode(payload).decode("ascii"),
                    media_type="application/octet-stream",
                ),
            )
            for payload in (b"a", b"b")
        ]
        text = TextBlock(id="same", text="prefix")
        sequence = (
            [blocks[0], text, blocks[1]] if data_first else [text, *blocks]
        )
        for block in sequence:
            yield ToolChunk(content=[block])


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

    async def test_cross_type_id_conflicts_use_canonical_chunk_ids(
        self,
    ) -> None:
        """Streamed events and the final response share data semantics."""
        for data_first, expected_payloads in (
            (False, [b"a", b"b"]),
            (True, [b"ab"]),
        ):
            with self.subTest(data_first=data_first):
                toolkit = Toolkit(tools=[CrossTypeIdTool()])
                agent = SimpleNamespace(
                    state=SimpleNamespace(reply_id="reply-1"),
                )
                reply = AssistantMsg(
                    id="reply-1",
                    name="agent",
                    content=[],
                )
                reply.append_event(
                    ToolResultStartEvent(
                        reply_id="reply-1",
                        tool_call_id="tool-1",
                        tool_call_name="cross_type_id",
                    ),
                )
                streamed_data_ids = []
                response = None

                async for result in toolkit.call_tool(
                    ToolCallBlock(
                        id="tool-1",
                        name="cross_type_id",
                        input=('{"data_first": true}' if data_first else "{}"),
                    ),
                    AgentState(),
                ):
                    if isinstance(result, ToolResponse):
                        response = result
                    else:
                        async for event in Agent._convert_tool_chunk_to_event(
                            agent,
                            "tool-1",
                            result.content,
                        ):
                            if isinstance(event, ToolResultDataDeltaEvent):
                                streamed_data_ids.append(event.block_id)
                            reply.append_event(event)

                self.assertIsNotNone(response)
                reply.append_event(
                    ToolResultEndEvent(
                        reply_id="reply-1",
                        tool_call_id="tool-1",
                        state=response.state,
                    ),
                )
                replayed = list(reply.get_content_blocks("tool_result"))[0]
                response_data = [
                    block
                    for block in response.content
                    if isinstance(block, DataBlock)
                ]
                replayed_data = [
                    block
                    for block in replayed.output
                    if isinstance(block, DataBlock)
                ]

                self.assertEqual(
                    list(dict.fromkeys(streamed_data_ids)),
                    [block.id for block in response_data],
                )
                self.assertEqual(
                    [block.id for block in replayed_data],
                    [block.id for block in response_data],
                )
                self.assertEqual(
                    [
                        base64.b64decode(block.source.data)
                        for block in replayed_data
                    ],
                    expected_payloads,
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

    async def test_empty_media_type_keeps_previous_value(self) -> None:
        """An empty later media type follows Toolkit merge semantics."""
        reply = AssistantMsg(id="reply-1", name="agent", content=[])
        reply.append_event(
            ToolResultStartEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                tool_call_name="stream_data",
            ),
        )
        for payload, media_type in ((b"a", "audio/wav"), (b"b", "")):
            reply.append_event(
                ToolResultDataDeltaEvent(
                    reply_id="reply-1",
                    tool_call_id="tool-1",
                    block_id="data-1",
                    data=base64.b64encode(payload).decode("ascii"),
                    media_type=media_type,
                ),
            )
        reply.append_event(
            ToolResultEndEvent(
                reply_id="reply-1",
                tool_call_id="tool-1",
                state=ToolResultState.SUCCESS,
            ),
        )

        result = list(reply.get_content_blocks("tool_result"))[0]
        self.assertEqual(len(result.output), 1)
        self.assertEqual(result.output[0].source.media_type, "audio/wav")
        self.assertEqual(
            base64.b64decode(result.output[0].source.data),
            b"ab",
        )

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
