from __future__ import annotations

import asyncio
from typing import Awaitable


async def cancel_task(task: asyncio.Task | Awaitable | None) -> None:
    if task is None:
        return
    if isinstance(task, asyncio.Task) or asyncio.isfuture(task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
