#!/usr/bin/env python3
from __future__ import annotations

import logging
import os

import discord

from app.config import VOICES_DIR
from app.discord.bot import Bot
from app.tts.engine import TTSEngine


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Set DISCORD_TOKEN env var.")

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    tts = TTSEngine(VOICES_DIR)
    tts.load_model()

    bot = Bot(tts)

    @bot.tree.command(name="join", description="Join the voice channel you're currently in.")
    async def join(interaction: discord.Interaction):
        await bot._join(interaction)

    @bot.tree.command(name="leave", description="Leave voice chat.")
    async def leave(interaction: discord.Interaction):
        await bot._leave(interaction)

    @bot.tree.command(name="setchannel", description="Choose which text channel I should read aloud.")
    @discord.app_commands.describe(channel="The text channel to read messages from")
    async def setchannel(interaction: discord.Interaction, channel: discord.TextChannel):
        await bot._set_channel(interaction, channel)

    @bot.tree.command(name="clone", description="Enroll your voice by reading a Spanish sample in voice chat.")
    async def clone(interaction: discord.Interaction):
        await bot._clone(interaction)

    @bot.tree.command(name="forget", description="Forget your enrolled voice (delete prompt file).")
    async def forget(interaction: discord.Interaction):
        await bot._forget(interaction)

    bot.run(token)


if __name__ == "__main__":
    main()
