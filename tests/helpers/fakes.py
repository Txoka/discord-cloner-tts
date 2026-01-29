from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FakeUser:
    id: int
    bot: bool = False


@dataclass
class FakeChannel:
    id: int
    name: str = "chan"


class FakeGuild:
    def __init__(self, guild_id: int, member: Any | None = None) -> None:
        self.id = int(guild_id)
        self.voice_client = None
        self._member = member

    def get_member(self, member_id: int):
        return self._member


class FakeVoiceClient:
    def __init__(self) -> None:
        self.connected = True
        self.play_calls: list[bytes] = []
        self._sink = None
        self._playing = False

    def is_connected(self) -> bool:
        return self.connected

    async def disconnect(self, force: bool = False) -> None:
        self.connected = False

    async def move_to(self, channel: Any) -> None:
        return None

    def play(self, src: Any, after=None) -> None:
        data = b""
        if hasattr(src, "source") and isinstance(src.source, io.BytesIO):
            data = src.source.getvalue()
        self.play_calls.append(data)
        self._playing = True
        if after:
            after(None)

    def is_playing(self) -> bool:
        return self._playing

    def stop(self) -> None:
        self._playing = False

    def listen(self, sink: Any, after=None) -> None:
        self._sink = sink
        if after:
            after(None)

    def stop_listening(self) -> None:
        return None


class FakeVoiceChannel:
    def __init__(self, voice_client: FakeVoiceClient, channel_id: int = 1):
        self._voice_client = voice_client
        self.name = "voice"
        self.id = int(channel_id)

    async def connect(self, cls=None):
        return self._voice_client


class FakeInteractionResponse:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.deferred = False

    async def send_message(self, content: str, ephemeral: bool = True) -> None:
        self.messages.append(content)

    async def defer(self, ephemeral: bool = True) -> None:
        self.deferred = True


class FakeInteractionFollowup:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, content: str, ephemeral: bool = True) -> None:
        self.messages.append(content)


class FakeInteraction:
    def __init__(self, guild_id: int, channel_id: int, user: Any, guild: Any) -> None:
        self.guild_id = int(guild_id)
        self.channel_id = int(channel_id)
        self.user = user
        self.guild = guild
        self.response = FakeInteractionResponse()
        self.followup = FakeInteractionFollowup()


@dataclass
class FakeMessage:
    guild: Any
    channel: Any
    author: Any
    clean_content: str

    def __init__(self, guild: Any, channel: Any, author: Any, clean_content: str) -> None:
        self.guild = guild
        self.channel = channel
        self.author = author
        self.clean_content = clean_content
        self.reactions: list[str] = []

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)
