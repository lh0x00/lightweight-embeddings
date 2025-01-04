import logging
import asyncio
import redis.asyncio as redis
import redis.exceptions
from datetime import datetime
from collections import defaultdict
from typing import Dict

logger = logging.getLogger(__name__)

class Analytics:
    def __init__(self, redis_url: str, sync_interval: int = 60, max_retries: int = 5):
        """
        Initializes the Analytics class with an async Redis connection and sync interval.

        Parameters:
        - redis_url: Redis connection URL (e.g., 'redis://localhost:6379/0')
        - sync_interval: Interval in seconds for syncing with Redis.
        - max_retries: Maximum number of retries for reconnecting to Redis.
        """
        self.redis_url = redis_url
        self.sync_interval = sync_interval
        self.max_retries = max_retries
        self.redis_client = self._create_redis_client()
        self.local_buffer = {
            "access": defaultdict(
                lambda: defaultdict(int)
            ),  # {period: {model_id: access_count}}
            "tokens": defaultdict(
                lambda: defaultdict(int)
            ),  # {period: {model_id: tokens_count}}
        }
        self.lock = asyncio.Lock()  # Async lock for thread-safe updates
        asyncio.create_task(self._start_sync_task())

        logger.info("Initialized Analytics with Redis connection: %s", redis_url)

    def _create_redis_client(self) -> redis.Redis:
        """
        Creates and returns a new Redis client.
        """
        return redis.from_url(
            self.redis_url,
            decode_responses=True,
            health_check_interval=10,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            socket_keepalive=True,
        )

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
            try:
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

            except redis.exceptions.ConnectionError as e:
                logger.error("Redis connection error during sync: %s", e)
                raise e
            except Exception as e:
                logger.error("Unexpected error during Redis sync: %s", e)
                raise e

    async def _start_sync_task(self):
        """
        Starts a background task that periodically syncs data to Redis.
        Implements retry logic with exponential backoff on connection failures.
        """
        retry_delay = 1  # Initial retry delay in seconds

        while True:
            await asyncio.sleep(self.sync_interval)
            try:
                await self._sync_to_redis()
                retry_delay = 1  # Reset retry delay after successful sync
            except redis.exceptions.ConnectionError as e:
                logger.error("Redis connection error: %s", e)
                await self._handle_redis_reconnection()
            except Exception as e:
                logger.error("Error during sync: %s", e)
                # Depending on the error, you might want to handle differently

    async def _handle_redis_reconnection(self):
        """
        Handles Redis reconnection with exponential backoff.
        """
        retry_count = 0
        delay = 1  # Start with 1 second delay

        while retry_count < self.max_retries:
            try:
                logger.info("Attempting to reconnect to Redis (Attempt %d)...", retry_count + 1)
                self.redis_client.close()
                self.redis_client = self._create_redis_client()
                # Optionally, perform a simple command to check connection
                self.redis_client.ping()
                logger.info("Successfully reconnected to Redis.")
                return
            except redis.exceptions.ConnectionError as e:
                logger.error("Reconnection attempt %d failed: %s", retry_count + 1, e)
                retry_count += 1
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        logger.critical("Max reconnection attempts reached. Unable to reconnect to Redis.")
        # Depending on your application's requirements, you might choose to exit or keep retrying indefinitely
        # For example, to keep retrying:
        while True:
            try:
                logger.info("Retrying to reconnect to Redis...")
                self.redis_client.close()
                self.redis_client = self._create_redis_client()
                self.redis_client.ping()
                logger.info("Successfully reconnected to Redis.")
                break
            except redis.exceptions.ConnectionError as e:
                logger.error("Reconnection attempt failed: %s", e)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)  # Cap the delay to 60 seconds

    async def close(self):
        """
        Closes the Redis connection gracefully.
        """
        self.redis_client.close()
        logger.info("Closed Redis connection.")
