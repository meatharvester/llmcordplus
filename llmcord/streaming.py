from __future__ import annotations

import time
from typing import Any, Awaitable, cast
import inspect
import logging
import re

import discord
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .constants import (
    EMBED_COLOR_COMPLETE,
    EMBED_COLOR_INCOMPLETE,
    STREAMING_INDICATOR,
    EDIT_DELAY_SECONDS,
    THINKING_SINCE_TEMPLATE,
    FOOTER_STREAMING_SUFFIX,
)
from .messages import MsgNode
from .reasoning import ThinkBlockRedactor


async def stream_and_reply(
    *,
    new_msg: discord.Message,
    openai_client: AsyncOpenAI,
    model: str,
    display_model: str,
    messages: list[ChatCompletionMessageParam],
    embed: discord.Embed,
    use_plain_responses: bool,
    max_message_length: int,
    extra_headers: dict[str, Any] | None,
    extra_query: dict[str, Any] | None,
    extra_body: dict[str, Any] | None,
    msg_nodes: dict[int, MsgNode],
    block_response_regex: str | None = None,
    reply_length_cap: int | None = None,
) -> tuple[list[discord.Message], list[str]]:
    """Stream chat completion and update Discord messages."""

    response_msgs: list[discord.Message] = []
    response_contents: list[str] = []
    last_edit_time: float = 0.0

    # Timing state
    start_perf = time.perf_counter()
    output_start_perf: float | None = None
    reasoning_started: bool = False
    reasoning_start_unix: int | None = None
    reasoning_preview: str = ""
    reasoning_accumulated: str = ""

    # Accumulated visible output
    response_full_text: str = ""

    # Simple think block redactor for <think> tags only
    think_redactor = ThinkBlockRedactor()

    # Optional regex to block outgoing messages
    regex_pattern: re.Pattern[str] | None = None
    if block_response_regex:
        try:
            regex_pattern = re.compile(block_response_regex)
        except Exception:
            # If the regex is invalid, ignore it gracefully
            regex_pattern = None

    # Keep a handle to the underlying OpenAI stream so we can close it early on abort
    stream: Any | None = None

    def _truncate_for_log(text: str, limit: int = 200) -> str:
        if text is None:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "… (truncated)"

    def _extract_reasoning_text(source: Any | None) -> str:
        """Return reasoning/thinking text from OpenAI-compatible delta/message objects."""

        if not source:
            return ""

        for attr in ("reasoning", "reasoning_content", "thinking"):
            value = getattr(source, attr, None)
            if not value:
                continue

            if isinstance(value, str):
                return value

            if isinstance(value, (list, tuple)):
                return "".join(str(part) for part in value if part)

            if isinstance(value, dict):
                text = value.get("text") or value.get("content") or value.get("delta")
                if text:
                    return str(text)
                return str(value)

            return str(value)

        return ""

    def _format_reasoning_preview(text: str) -> str:
        if not text:
            return ""
        normalized = re.sub(r"\r\n?|\n", "\n", text)
        lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            return ""
        preview_lines: list[str] = []
        for line in lines[-3:]:
            truncated = line[-50:]
            if len(line) > 50:
                truncated = "..." + truncated
            preview_lines.append(f"-# *{truncated}*")
        return "\n".join(preview_lines)

    async def abort_and_send_error(error_text: str) -> None:
        """Delete any messages we created, release locks, and notify the user."""
        # Proactively close the OpenAI stream if it's still open
        try:
            if stream is not None:
                try:
                    # Best-effort close; different providers may implement close differently
                    close_func = getattr(stream, "close", None)
                    if callable(close_func):
                        maybe_awaitable = close_func()
                        if inspect.isawaitable(maybe_awaitable):
                            await cast(Awaitable[object], maybe_awaitable)

                    # Some variants expose an underlying HTTP response with close()/aclose()
                    response_obj = getattr(stream, "response", None)
                    if response_obj is not None:
                        aclose_func = getattr(response_obj, "aclose", None)
                        close_func = getattr(response_obj, "close", None)
                        if callable(aclose_func):
                            maybe_awaitable = aclose_func()
                            if inspect.isawaitable(maybe_awaitable):
                                await cast(Awaitable[object], maybe_awaitable)
                        elif callable(close_func):
                            close_func()
                except Exception:
                    pass
        except Exception:
            pass
        # Delete any partial response messages (including warnings embed if present)
        for msg in list(response_msgs):
            try:
                await msg.delete()
            except Exception:
                pass
            try:
                node = msg_nodes.get(msg.id)
                if node is not None and node.lock.locked():
                    node.lock.release()
                msg_nodes.pop(msg.id, None)
            except Exception:
                pass
        response_msgs.clear()
        try:
            error_embed = discord.Embed(
                description=error_text, color=discord.Color.red()
            )
            await new_msg.reply(embed=error_embed, silent=True)
        except Exception:
            pass

    try:
        # If warnings exist and we're using embeds, send them as a separate message first
        if (not use_plain_responses) and getattr(embed, "fields", None):
            try:
                if len(embed.fields) > 0:
                    warn_msg = await new_msg.reply(embed=embed, silent=True)
                    response_msgs.append(warn_msg)
                    msg_nodes[warn_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[warn_msg.id].lock.acquire()
            except Exception:
                pass

        async with new_msg.channel.typing():
            # Correct usage: await create() to get an async iterator
            stream = await openai_client.chat.completions.create(
                model=model,
                messages=messages[::-1],
                stream=True,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
            )

            async for event in stream:
                # Some providers send heartbeat/meta events without choices
                if not hasattr(event, "choices") or not event.choices:
                    continue

                choice = event.choices[0]
                # Extract raw content delta if present
                raw_delta = getattr(getattr(choice, "delta", None), "content", "") or ""
                reasoning_delta = _extract_reasoning_text(
                    getattr(choice, "delta", None)
                )
                visible_delta = ""

                finish_reason = getattr(choice, "finish_reason", None)

                extra_segments: list[str] = []
                if reasoning_delta:
                    reasoning_accumulated += reasoning_delta
                    extra_segments.append(
                        f"{ThinkBlockRedactor.OPEN_TAG}{reasoning_delta}{ThinkBlockRedactor.CLOSE_TAG}"
                    )
                    reasoning_preview = _format_reasoning_preview(reasoning_accumulated)

                # Some providers (e.g., Ollama) attach full reasoning to the completed message.
                if finish_reason is not None:
                    message_reasoning = _extract_reasoning_text(
                        getattr(choice, "message", None)
                    )
                    if message_reasoning:
                        reasoning_accumulated += message_reasoning
                        extra_segments.append(
                            f"{ThinkBlockRedactor.OPEN_TAG}{message_reasoning}{ThinkBlockRedactor.CLOSE_TAG}"
                        )
                        reasoning_preview = _format_reasoning_preview(
                            reasoning_accumulated
                        )

                composite_delta = "".join(extra_segments) + raw_delta

                visible_delta = ""
                saw_thinking = False

                if composite_delta:
                    visible_delta, saw_thinking = think_redactor.process(
                        composite_delta
                    )

                if saw_thinking and not reasoning_started:
                    reasoning_started = True
                    reasoning_start_unix = int(time.time())

                # On finish, flush any buffered think text even if no content in this chunk
                if finish_reason is not None:
                    tail = think_redactor.flush()
                    if tail:
                        visible_delta += tail

                # Record first visible output time
                if output_start_perf is None and visible_delta:
                    output_start_perf = time.perf_counter()

                # Skip if no visible content and not finishing AND no thinking detected
                if (
                    not visible_delta
                    and finish_reason is None
                    and not reasoning_started
                ):
                    continue

                # Accumulate visible output
                if visible_delta:
                    response_full_text += visible_delta

                # If a block regex is configured, abort immediately if the accumulated
                # outgoing text would match it (works for both embed and plain modes).
                if regex_pattern is not None:
                    try:
                        match_obj = regex_pattern.search(response_full_text or "")
                        if match_obj is not None:
                            try:
                                logging.info(
                                    "Blocked by regex during stream accumulation | model=%s | matched=%r | preview=%r",
                                    display_model,
                                    _truncate_for_log(match_obj.group(0), 120),
                                    _truncate_for_log(response_full_text, 200),
                                )
                            except Exception:
                                pass
                            await abort_and_send_error(
                                "Response blocked by server policy."
                            )
                            return [], []
                    except Exception:
                        pass

                # Enforce global reply length cap (across the whole reply), if configured
                if reply_length_cap is not None and reply_length_cap > 0:
                    if len(response_full_text) >= reply_length_cap:
                        await abort_and_send_error(
                            f"Reply length exceeded the configured cap ({reply_length_cap} characters)."
                        )
                        return [], []

                # Update Discord messages
                if not use_plain_responses:
                    ready_to_edit = (
                        time.monotonic() - last_edit_time
                    ) >= EDIT_DELAY_SECONDS
                    is_final_edit = getattr(choice, "finish_reason", None) is not None

                    if ready_to_edit or is_final_edit:
                        # Build header for the first message only
                        header = ""
                        if (
                            reasoning_started
                            and output_start_perf is None
                            and reasoning_start_unix is not None
                        ):
                            header = THINKING_SINCE_TEMPLATE.replace(
                                "{ts}", str(reasoning_start_unix)
                            )
                            if reasoning_preview:
                                header = header + "\n" + reasoning_preview

                        # Build full body including streaming indicator (only for the last segment)
                        body_now = response_full_text
                        if not is_final_edit:
                            body_now = body_now + STREAMING_INDICATOR

                        # Split content across multiple messages so nothing is overwritten
                        split_descriptions: list[str] = []
                        remaining = body_now.lstrip("\n")
                        # First message description
                        newline_len = 1 if header and remaining else 0
                        first_capacity = max(
                            0, max_message_length - len(header) - newline_len
                        )
                        first_chunk = remaining[:first_capacity]
                        desc0 = (
                            header
                            + ("\n" if header and first_chunk else "")
                            + first_chunk
                        )
                        split_descriptions.append(desc0)
                        remaining = remaining[len(first_chunk) :]
                        # Subsequent messages descriptions
                        while remaining:
                            chunk = remaining[:max_message_length]
                            split_descriptions.append(chunk)
                            remaining = remaining[len(chunk) :]

                        # If a block regex is configured, block immediately if any outgoing message
                        # (embed description) would match it.
                        if regex_pattern is not None:
                            try:
                                for desc in split_descriptions:
                                    match_obj = regex_pattern.search(desc or "")
                                    if match_obj is not None:
                                        try:
                                            logging.info(
                                                "Blocked by regex before sending embed chunk | model=%s | matched=%r | preview=%r",
                                                display_model,
                                                _truncate_for_log(
                                                    match_obj.group(0), 120
                                                ),
                                                _truncate_for_log(desc or "", 200),
                                            )
                                        except Exception:
                                            pass
                                        await abort_and_send_error(
                                            "Response blocked by server policy."
                                        )
                                        return [], []
                            except Exception:
                                # If anything goes wrong applying the regex, fail open (continue)
                                pass

                        # Compute live tokens/sec estimate for footer
                        try:
                            now = time.perf_counter()
                            elapsed_live = (
                                now - (output_start_perf or start_perf)
                            ) or 1e-6
                            approx_tokens_live = len(response_full_text) / 4.0
                            tps_live = (
                                approx_tokens_live / elapsed_live
                                if elapsed_live > 0
                                else 0.0
                            )
                        except Exception:
                            tps_live = 0.0
                        footer_live = f"{display_model} • {tps_live:.1f} tok/s{FOOTER_STREAMING_SUFFIX}"

                        # Create or edit messages to match descriptions
                        for i, desc in enumerate(split_descriptions):
                            embed_i = discord.Embed(
                                description=desc,
                                color=(
                                    EMBED_COLOR_COMPLETE
                                    if is_final_edit
                                    else EMBED_COLOR_INCOMPLETE
                                ),
                            )
                            embed_i.set_footer(text=footer_live)

                            if i < len(response_msgs):
                                # Edit existing message
                                await response_msgs[i].edit(embed=embed_i)
                            else:
                                # Create a new message, chained to the last one for readability
                                reply_to = (
                                    new_msg if not response_msgs else response_msgs[-1]
                                )
                                msg = await reply_to.reply(embed=embed_i, silent=True)
                                response_msgs.append(msg)
                                msg_nodes[msg.id] = MsgNode(parent_msg=new_msg)
                                await msg_nodes[msg.id].lock.acquire()

                        last_edit_time = time.monotonic()

                # Break after final finish chunk
                if getattr(choice, "finish_reason", None) is not None:
                    break

    except Exception as e:
        # Handle any streaming errors
        error_embed = discord.Embed(
            description=f"Error during streaming: {str(e)}", color=discord.Color.red()
        )
        if response_msgs:
            await response_msgs[-1].edit(embed=error_embed)
        else:
            await new_msg.reply(embed=error_embed, silent=True)
        raise

    # Handle plain text responses (split into multiple messages respecting max length)
    if use_plain_responses:
        content = response_full_text
        remaining = content
        while remaining:
            chunk = remaining[:max_message_length]
            # If a block regex is configured, block immediately if any outgoing chunk would match it.
            if regex_pattern is not None:
                try:
                    match_obj = regex_pattern.search(chunk or "")
                    if match_obj is not None:
                        try:
                            logging.info(
                                "Blocked by regex before sending plain chunk | model=%s | matched=%r | preview=%r",
                                display_model,
                                _truncate_for_log(match_obj.group(0), 120),
                                _truncate_for_log(chunk or "", 200),
                            )
                        except Exception:
                            pass
                        await abort_and_send_error("Response blocked by server policy.")
                        return [], []
                except Exception:
                    pass
            reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
            response_msg = await reply_to_msg.reply(content=chunk, suppress_embeds=True)
            response_msgs.append(response_msg)
            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
            await msg_nodes[response_msg.id].lock.acquire()
            remaining = remaining[len(chunk) :]

    # Finalize: compute tok/s and update the first message with final footer
    try:
        if response_msgs:
            end_perf = time.perf_counter()
            elapsed = (end_perf - (output_start_perf or start_perf)) or 1e-6

            approx_tokens = len(response_full_text) / 4.0
            tps = approx_tokens / elapsed if elapsed > 0 else 0.0

            footer_text = f"{display_model} • {tps:.1f} tok/s"

            # Build the final set of message descriptions (split across messages)
            header = ""
            if (
                reasoning_started
                and output_start_perf is None
                and reasoning_start_unix is not None
            ):
                header = THINKING_SINCE_TEMPLATE.replace(
                    "{ts}", str(reasoning_start_unix)
                )
                if reasoning_preview:
                    header = header + "\n" + reasoning_preview

            remaining = response_full_text.lstrip("\n")
            final_descriptions: list[str] = []
            newline_len = 1 if header and remaining else 0
            first_capacity = max(0, max_message_length - len(header) - newline_len)
            first_chunk = remaining[:first_capacity]
            desc0 = header + ("\n" if header and first_chunk else "") + first_chunk
            final_descriptions.append(desc0)
            remaining = remaining[len(first_chunk) :]
            while remaining:
                chunk = remaining[:max_message_length]
                final_descriptions.append(chunk)
                remaining = remaining[len(chunk) :]

            # Apply final embeds and footer to all messages
            for i, desc in enumerate(final_descriptions):
                color = EMBED_COLOR_COMPLETE
                embed_i = discord.Embed(description=desc, color=color)
                # Add '(cont.)' marker on continuation messages for clarity
                footer_text_i = footer_text + (" • (cont.)" if i > 0 else "")
                embed_i.set_footer(text=footer_text_i)
                if i < len(response_msgs):
                    await response_msgs[i].edit(embed=embed_i)
                else:
                    reply_to = new_msg if not response_msgs else response_msgs[-1]
                    msg = await reply_to.reply(embed=embed_i, silent=True)
                    response_msgs.append(msg)
                    msg_nodes[msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[msg.id].lock.acquire()
    except Exception:
        pass

    # Return a single consolidated content segment
    response_contents = [response_full_text]
    return response_msgs, response_contents
