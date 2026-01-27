from __future__ import annotations

import re

import discord

from app.config import MAX_CHARS_PER_MESSAGE

_URL_RE = re.compile(
    r"""(?xi)
\bhttps?://[^\s<>()]+
"""
)

# Markdown link: [text](https://example.com)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(\s*https?://[^)]+\s*\)", re.IGNORECASE)

# Custom emojis: <:name:id> or <a:name:id>  -> "name"
_CUSTOM_EMOJI_RE = re.compile(r"<a?:([a-zA-Z0-9_~]+):\d+>")

# Optional: raw user/role mentions if any slip through (usually handled by clean_content)
_RAW_MENTION_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")

try:
    import emoji as _emoji  # type: ignore
except Exception:
    _emoji = None

# Fallback: broad (imperfect) emoji-ish remover (covers most, not all)
_EMOJI_FALLBACK_RE = re.compile(
    r"["
    # Emoticons, Misc Symbols, Dingbats
    r"\U0001F600-\U0001F64F"
    r"\U00002600-\U000026FF"
    r"\U00002700-\U000027BF"
    # Transport & Map, Misc Symbols & Pictographs, Supplemental, Extended
    r"\U0001F680-\U0001F6FF"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA70-\U0001FAFF"
    # Flags
    r"\U0001F1E6-\U0001F1FF"
    r"]+"
)

# Variation selectors + ZWJ (commonly part of emoji sequences)
_EMOJI_JOINERS_RE = re.compile(r"[\u200D\uFE0E\uFE0F]")

# :emoji_name: tokens after demojize
_DEMOJIZED_TOKEN_RE = re.compile(r":([a-zA-Z0-9_\-\+]+):")


def unicode_emojis_to_speech(text: str) -> str:
    """Convert unicode emoji to speakable words.

    Requires `emoji` library. If not available, removes most emoji.
    """
    if _emoji is not None:
        # "😀" -> ":grinning_face:"
        dem = _emoji.demojize(text, language="en")
        # ":grinning_face:" -> " grinning face "
        dem = _DEMOJIZED_TOKEN_RE.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", dem)
        # Clean leftover joiners if any
        dem = _EMOJI_JOINERS_RE.sub("", dem)
        return dem

    # Fallback: just remove most emoji-like codepoints and joiners
    text = _EMOJI_JOINERS_RE.sub("", text)
    text = _EMOJI_FALLBACK_RE.sub(" ", text)
    return text


def preprocess_discord_text(message: discord.Message, max_chars: int = MAX_CHARS_PER_MESSAGE) -> str:
    """Turn Discord message content into something TTS-friendly."""
    text = (message.clean_content or "").strip()
    if not text:
        return ""

    # Convert markdown links to visible text (drop the URL)
    text = _MD_LINK_RE.sub(r"\1", text)

    # Remove bare URLs
    text = _URL_RE.sub("", text)

    # Turn custom emojis into their name
    text = _CUSTOM_EMOJI_RE.sub(r"\1", text).replace("_", " ")

    # Turn unicode emoji into speech (or remove)
    text = unicode_emojis_to_speech(text)

    # If any raw mention tokens remain, drop them (or replace with something)
    text = _RAW_MENTION_RE.sub("", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Enforce length cap after cleanup
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"

    return text
