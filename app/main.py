#!/usr/bin/env python3
from __future__ import annotations

import logging
import os

import discord
from discord import app_commands

from app.config import (
    ADMIN_DB_PATH,
    DEVICE,
    DTYPE,
    LOG_LEVEL,
    MAX_BATCH_SIZE,
    MODEL_ID,
    NORM_MODE,
    SUPERADMIN_IDS,
    VOICES_DIR,
)
from app.discord.admin_store import AdminStore
from app.discord.bot import Bot
from app.tts.engine import TTSEngine


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("qwen-discord-tts")

    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Set DISCORD_TOKEN env var.")

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    tts = TTSEngine(VOICES_DIR)
    tts.load_model()
    admin_store = AdminStore(ADMIN_DB_PATH, SUPERADMIN_IDS)
    admin_store.init_schema()
    admin_store.bootstrap_superadmins()
    log.info(
        "Startup model=%s device=%s dtype=%s max_batch=%s norm_mode=%s voices_dir=%s",
        MODEL_ID,
        DEVICE,
        DTYPE,
        MAX_BATCH_SIZE,
        NORM_MODE,
        VOICES_DIR,
    )

    bot = Bot(tts, admin_store)

    @bot.tree.command(name="join", description="Join your current voice channel, or a specified one.")
    @discord.app_commands.describe(channel="Optional voice channel to join")
    async def join(interaction: discord.Interaction, channel: discord.VoiceChannel | None = None):
        await bot._join(interaction, channel)

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

    @bot.tree.command(name="disguise", description="Admins only: speak using someone else's voice.")
    @discord.app_commands.describe(target="User whose voice you want to use")
    async def disguise(interaction: discord.Interaction, target: discord.Member):
        await bot._disguise(interaction, target)

    @bot.tree.command(name="addadmin", description="Superadmins only: add an admin or superadmin.")
    @discord.app_commands.describe(target="User to grant admin access")
    @discord.app_commands.describe(role="admin or superadmin")
    @app_commands.choices(
        role=[
            app_commands.Choice(name="admin", value="admin"),
            app_commands.Choice(name="superadmin", value="superadmin"),
        ]
    )
    async def addadmin(
        interaction: discord.Interaction,
        target: discord.Member,
        role: app_commands.Choice[str] | None = None,
    ):
        await bot._add_admin(interaction, target, role.value if role else "admin")

    @bot.tree.command(name="removeadmin", description="Superadmins only: remove an admin or superadmin.")
    @discord.app_commands.describe(target="User to remove from admins")
    async def removeadmin(interaction: discord.Interaction, target: discord.Member):
        await bot._remove_admin(interaction, target)

    @bot.tree.command(name="sync", description="Owner only: sync slash commands.")
    async def sync(interaction: discord.Interaction):
        await bot._sync_commands(interaction)

    bot.run(token)


if __name__ == "__main__":
    main()
