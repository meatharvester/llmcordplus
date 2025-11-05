from __future__ import annotations

import discord

PROVIDERS_SUPPORTING_USERNAMES: tuple[str, ...] = ("openai",)


# Discord embed styles
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()


# Streaming and message formatting
EMBED_DESCRIPTION_MAX_LENGTH = 4096
EMBED_TOTAL_MAX_LENGTH = 6000
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1


# Internal caches
MAX_MESSAGE_NODES = 500


# Common text fragments
FOOTER_REASONING_SUFFIX = " • thinking..."
FOOTER_STREAMING_SUFFIX = " • streaming..."

THINKING_SINCE_TEMPLATE = "**💭 Thinking since <t:{ts}:R>...**"
DONE_THINKING_PREFIX = ""

# Warning texts (templates)
WARNING_MAX_TEXT_TEMPLATE = "⚠️ Max {max_text:,} characters per message"
WARNING_MAX_IMAGES_TEMPLATE = "⚠️ Max {max_images} image{s} per message"
WARNING_CANT_SEE_IMAGES = "⚠️ Can't see images"
WARNING_UNSUPPORTED_ATTACHMENTS = "⚠️ Unsupported attachments"
WARNING_ONLY_USING_LAST_TEMPLATE = "⚠️ Only using last {messages_count} message{s}"

__all__ = [
    "PROVIDERS_SUPPORTING_USERNAMES",
    "EMBED_COLOR_COMPLETE",
    "EMBED_COLOR_INCOMPLETE",
    "EMBED_DESCRIPTION_MAX_LENGTH",
    "EMBED_TOTAL_MAX_LENGTH",
    "STREAMING_INDICATOR",
    "EDIT_DELAY_SECONDS",
    "MAX_MESSAGE_NODES",
    "FOOTER_REASONING_SUFFIX",
    "FOOTER_STREAMING_SUFFIX",
    "THINKING_SINCE_TEMPLATE",
    "DONE_THINKING_PREFIX",
    "WARNING_MAX_TEXT_TEMPLATE",
    "WARNING_MAX_IMAGES_TEMPLATE",
    "WARNING_CANT_SEE_IMAGES",
    "WARNING_UNSUPPORTED_ATTACHMENTS",
    "WARNING_ONLY_USING_LAST_TEMPLATE",
]
