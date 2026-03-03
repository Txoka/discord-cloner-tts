from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any


@dataclass
class FakeUser:
    id: int
    bot: bool = False


@dataclass
class FakeChannel:
    id: int
    name: str = "chan"


class FakeGuild:
    def __init__(self, guild_id: int, member: Any | None = None, channel: Any | None = None) -> None:
        self.id = int(guild_id)
        self.voice_client = None
        self._member = member
        self._channel = channel

    def get_member(self, member_id: int):
        return self._member

    def get_channel(self, channel_id: int):
        if self._channel and int(getattr(self._channel, "id", -1)) == int(channel_id):
            return self._channel
        return None


class FakeVoiceClient:
    def __init__(self) -> None:
        self.supports_receive = False
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


class FakeVoiceRecvClient(FakeVoiceClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_receive = True


class SpyVoiceClient(FakeVoiceClient):
    def __init__(self) -> None:
        super().__init__()
        self.play_sources: list[Any] = []

    def play(self, src: Any, after=None) -> None:
        self.play_sources.append(src)
        super().play(src, after=after)


class FakeGuildPCMStream:
    def __init__(self, guild_id: int):
        self.guild_id = int(guild_id)
        self.items: list[bytes] = []
        self.closed = False

    def is_opus(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True

    @property
    def _closed(self) -> bool:
        return self.closed

    def enqueue(self, _job: Any, pcm_bytes: bytes, _loop: Any) -> Any:
        self.items.append(pcm_bytes)
        event = asyncio.Event()
        event.set()
        return event

    def drain_for_tests(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class FakeVoiceChannel:
    def __init__(
        self,
        voice_client: FakeVoiceClient,
        channel_id: int = 1,
        members: list[Any] | None = None,
        recv_voice_client: FakeVoiceClient | None = None,
    ):
        self._voice_client = voice_client
        self._recv_voice_client = recv_voice_client if recv_voice_client is not None else voice_client
        self.name = "voice"
        self.id = int(channel_id)
        self.members = list(members) if members is not None else []
        self.connect_calls: list[dict[str, Any]] = []

    async def connect(self, cls=None, **kwargs):
        self.connect_calls.append({"cls": cls, "kwargs": dict(kwargs)})
        if cls is None:
            return self._voice_client
        return self._recv_voice_client


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
