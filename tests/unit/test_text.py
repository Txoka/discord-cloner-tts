from __future__ import annotations

from dataclasses import dataclass

import app.tts.text as tts_text


@dataclass
class Msg:
    clean_content: str | None


def test_preprocess_discord_text_basic_cleanup():
    msg = Msg(
        clean_content="Hello [link](https://example.com) <@123> <:wave:42> https://x.y"
    )
    out = tts_text.preprocess_discord_text(msg, max_chars=200)
    assert "link" in out
    assert "https" not in out
    assert "wave" in out
    assert "<@" not in out


def test_preprocess_discord_text_empty_returns_blank():
    msg = Msg(clean_content="  ")
    assert tts_text.preprocess_discord_text(msg) == ""


def test_preprocess_discord_text_length_cap():
    msg = Msg(clean_content="a" * 10)
    out = tts_text.preprocess_discord_text(msg, max_chars=5)
    assert out.endswith("…")
    assert len(out) == 6


def test_unicode_emojis_to_speech_fallback(monkeypatch):
    monkeypatch.setattr(tts_text, "_emoji", None)
    out = tts_text.unicode_emojis_to_speech("hi 😀")
    assert "😀" not in out


def test_preprocess_discord_text_emoji_only(monkeypatch):
    class FakeEmoji:
        @staticmethod
        def demojize(text, language="en"):
            return ":grinning_face:"

    monkeypatch.setattr(tts_text, "_emoji", FakeEmoji)
    msg = Msg(clean_content="😀")
    out = tts_text.preprocess_discord_text(msg, max_chars=200)
    assert "grinning face" in out
