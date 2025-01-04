import logging
import asyncio
import redis.asyncio as redis
import redis.exceptions
from datetime import datetime
from collections import defaultdict
from typing import Dict


logger = logging.getLogger(__name__)


class Analytics:
    def __init__(self, redis_url: str, sync_interval: int = 60):
        """
        Initializes the Analytics class with an async Redis connection and sync interval.

        Parameters:
        - redis_url: Redis connection URL (e.g., 'redis://localhost:6379/0')
        - sync_interval: Interval in seconds for syncing with Redis.
        """
        self.redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            health_check_interval=10,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            socket_keepalive=True,
        )
        self.local_buffer = {
            "access": defaultdict(
                lambda: defaultdict(int)
            ),  # {period: {model_id: access_count}}
            "tokens": defaultdict(
                lambda: defaultdict(int)
            ),  # {period: {model_id: tokens_count}}
        }
        self.sync_interval = sync_interval
        self.lock = asyncio.Lock()  # Async lock for thread-safe updates
        asyncio.create_task(self._start_sync_task())
        
        logger.info("Initialized Analytics with Redis connection: %s", redis_url)

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

    async def access(self, model_id: str, tokens: int):
        """
        Records an access and token usage for a specific model_id.

        Parameters:
        - model_id: The ID of the model being accessed.
        - tokens: Number of tokens used in this access.
        """
        day_key, week_key, month_key, year_key = self._get_period_keys()

        async with self.lock:
            # Increment access count
            self.local_buffer["access"][day_key][model_id] += 1
            self.local_buffer["access"][week_key][model_id] += 1
            self.local_buffer["access"][month_key][model_id] += 1
            self.local_buffer["access"][year_key][model_id] += 1
            self.local_buffer["access"]["total"][model_id] += 1

            # Increment token count
            self.local_buffer["tokens"][day_key][model_id] += tokens
            self.local_buffer["tokens"][week_key][model_id] += tokens
            self.local_buffer["tokens"][month_key][model_id] += tokens
            self.local_buffer["tokens"][year_key][model_id] += tokens
            self.local_buffer["tokens"]["total"][model_id] += tokens

    async def stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Returns statistics for all models from the local buffer.

        Returns:
        - A dictionary with access counts and token usage for each period.
        """
        async with self.lock:
            return {
                "access": {
                    period: dict(models)
                    for period, models in self.local_buffer["access"].items()
                },
                "tokens": {
                    period: dict(models)
                    for period, models in self.local_buffer["tokens"].items()
                },
            }

    async def _sync_to_redis(self):
        """
        Synchronizes local buffer data with Redis.
        """
        async with self.lock:
            pipeline = self.redis_client.pipeline()

            # Sync access counts
            for period, models in self.local_buffer["access"].items():
                for model_id, count in models.items():
                    redis_key = f"analytics:access:{period}"
                    pipeline.hincrby(redis_key, model_id, count)

            # Sync token counts
            for period, models in self.local_buffer["tokens"].items():
                for model_id, count in models.items():
                    redis_key = f"analytics:tokens:{period}"
                    pipeline.hincrby(redis_key, model_id, count)

            pipeline.execute()
            self.local_buffer["access"].clear()  # Clear access buffer after sync
            self.local_buffer["tokens"].clear()  # Clear tokens buffer after sync
            logger.info("Synced analytics data to Redis.")

    async def _start_sync_task(self):
        """
        Starts a background task that periodically syncs data to Redis.
        """
        while True:
            await asyncio.sleep(self.sync_interval)
            try:
                await self._sync_to_redis()
            except redis.exceptions.ConnectionError as e:
                logger.error("Redis connection error: %s", e)
                await asyncio.sleep(5)
