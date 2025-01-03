import asyncio
import redis.asyncio as redis
from datetime import datetime
from collections import defaultdict
from typing import Dict


class Analytics:
    def __init__(self, redis_url: str, sync_interval: int = 60):
        """
        Initializes the Analytics class with an async Redis connection and sync interval.

        Parameters:
        - redis_url: Redis connection URL (e.g., 'redis://localhost:6379/0')
        - sync_interval: Interval in seconds for syncing with Redis.
        """
        self.pool = redis.ConnectionPool.from_url(redis_url, decode_responses=True)
        self.redis_client = redis.Redis(connection_pool=self.pool)
        self.local_buffer = defaultdict(
            lambda: defaultdict(int)
        )  # {period: {model_id: count}}
        self.sync_interval = sync_interval
        self.lock = asyncio.Lock()  # Async lock for thread-safe updates
        asyncio.create_task(self._start_sync_task())

    def _get_period_keys(self) -> tuple:
        """
        Returns keys for day, week, month, and year based on the current date.
        """
        now = datetime.utcnow()
        day_key = now.strftime("%Y-%m-%d")
        week_key = f"{now.year}-W{now.strftime('%U')}"
        month_key = now.strftime("%Y-%m")
        year_key = now.strftime("%Y")
        return day_key, week_key, month_key, year_key

    async def access(self, model_id: str):
        """
        Records an access for a specific model_id.
        """
        day_key, week_key, month_key, year_key = self._get_period_keys()

        async with self.lock:
            self.local_buffer[day_key][model_id] += 1
            self.local_buffer[week_key][model_id] += 1
            self.local_buffer[month_key][model_id] += 1
            self.local_buffer[year_key][model_id] += 1
            self.local_buffer["total"][model_id] += 1

    async def stats(self) -> Dict[str, Dict[str, int]]:
        """
        Returns statistics for all models from the local buffer.
        """
        async with self.lock:
            return {
                period: dict(models) for period, models in self.local_buffer.items()
            }

    async def _sync_to_redis(self):
        """
        Synchronizes local buffer data with Redis.
        """
        async with self.lock:
            pipeline = self.redis_client.pipeline()
            for period, models in self.local_buffer.items():
                for model_id, count in models.items():
                    redis_key = f"analytics:{period}"
                    pipeline.hincrby(redis_key, model_id, count)
            await pipeline.execute()
            self.local_buffer.clear()  # Clear the buffer after sync

    async def _start_sync_task(self):
        """
        Starts a background task that periodically syncs data to Redis.
        """
        while True:
            await asyncio.sleep(self.sync_interval)
            await self._sync_to_redis()
