import logging
import asyncio
import redis
import redis.exceptions
from datetime import datetime
from collections import defaultdict
from typing import Dict
from functools import partial

logger = logging.getLogger(__name__)


class Analytics:
    def __init__(self, redis_url: str, sync_interval: int = 60, max_retries: int = 5):
        """
        Initializes the Analytics class with a synchronous Redis client,
        wrapped in asynchronous methods by using run_in_executor.

        Parameters:
        - redis_url (str): Redis connection URL (e.g., 'redis://localhost:6379/0').
        - sync_interval (int): Interval in seconds for syncing with Redis.
        - max_retries (int): Maximum number of reconnection attempts to Redis.
        """
        self.redis_url = redis_url
        self.sync_interval = sync_interval
        self.max_retries = max_retries

        # Synchronous Redis client
        self.redis_client = self._create_redis_client()

        # Local buffer stores cumulative data for two-way sync
        self.local_buffer = {
            "access": defaultdict(lambda: defaultdict(int)),
            "tokens": defaultdict(lambda: defaultdict(int)),
        }

        # Asynchronous lock to protect shared data
        self.lock = asyncio.Lock()

        # Initialize data from Redis, then start the periodic sync loop
        asyncio.create_task(self._initialize())

        logger.info("Initialized Analytics with Redis connection: %s", redis_url)

    def _create_redis_client(self) -> redis.Redis:
        """
        Creates and returns a new synchronous Redis client.
        """
        return redis.from_url(
            self.redis_url,
            decode_responses=True,
            health_check_interval=10,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )

    async def _initialize(self):
        """
        Fetches existing data from Redis into the local buffer,
        then starts the periodic synchronization task.
        """
        try:
            await self._sync_from_redis()
            logger.info("Initial sync from Redis to local buffer completed.")
        except Exception as e:
            logger.error("Error during initial sync from Redis: %s", e)

        # Launch the periodic sync task
        asyncio.create_task(self._start_sync_task())

    def _get_period_keys(self) -> tuple:
        """
        Returns day, week, month, and year keys based on the current UTC date.
        """
        now = datetime.utcnow()
        day_key = now.strftime("%Y-%m-%d")
        week_key = f"{now.year}-W{now.strftime('%U')}"
        month_key = now.strftime("%Y-%m")
        year_key = now.strftime("%Y")
        return day_key, week_key, month_key, year_key

    async def access(self, model_id: str, tokens: int):
        """
        Records an access event and token usage for a specific model.

        Parameters:
        - model_id (str): The ID of the accessed model.
        - tokens (int): Number of tokens used in this access event.
        """
        day_key, week_key, month_key, year_key = self._get_period_keys()

        async with self.lock:
            # Access counts
            self.local_buffer["access"][day_key][model_id] += 1
            self.local_buffer["access"][week_key][model_id] += 1
            self.local_buffer["access"][month_key][model_id] += 1
            self.local_buffer["access"][year_key][model_id] += 1
            self.local_buffer["access"]["total"][model_id] += 1

            # Token usage
            self.local_buffer["tokens"][day_key][model_id] += tokens
            self.local_buffer["tokens"][week_key][model_id] += tokens
            self.local_buffer["tokens"][month_key][model_id] += tokens
            self.local_buffer["tokens"][year_key][model_id] += tokens
            self.local_buffer["tokens"]["total"][model_id] += tokens

    async def stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Returns a copy of current statistics from the local buffer.
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

    async def _sync_from_redis(self):
        """
        Pulls existing analytics data from Redis into the local buffer.
        Uses run_in_executor to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()

        async with self.lock:
            # Scan 'access' keys
            cursor = 0
            while True:
                cursor, keys = await loop.run_in_executor(
                    None,
                    partial(
                        self.redis_client.scan,
                        cursor=cursor,
                        match="analytics:access:*",
                        count=100,
                    ),
                )
                for key in keys:
                    # key is "analytics:access:<period>"
                    period = key.replace("analytics:access:", "")
                    data = await loop.run_in_executor(
                        None, partial(self.redis_client.hgetall, key)
                    )
                    for model_id, count_str in data.items():
                        self.local_buffer["access"][period][model_id] += int(count_str)
                if cursor == 0:
                    break

            # Scan 'tokens' keys
            cursor = 0
            while True:
                cursor, keys = await loop.run_in_executor(
                    None,
                    partial(
                        self.redis_client.scan,
                        cursor=cursor,
                        match="analytics:tokens:*",
                        count=100,
                    ),
                )
                for key in keys:
                    # key is "analytics:tokens:<period>"
                    period = key.replace("analytics:tokens:", "")
                    data = await loop.run_in_executor(
                        None, partial(self.redis_client.hgetall, key)
                    )
                    for model_id, count_str in data.items():
                        self.local_buffer["tokens"][period][model_id] += int(count_str)
                if cursor == 0:
                    break

    async def _sync_to_redis(self):
        """
        Pushes the local buffer data to Redis (local -> Redis).
        Uses a pipeline to minimize round trips and run_in_executor to avoid blocking.
        """
        loop = asyncio.get_running_loop()

        async with self.lock:
            try:
                pipeline = self.redis_client.pipeline(transaction=False)

                # Push 'access' data
                for period, models in self.local_buffer["access"].items():
                    redis_key = f"analytics:access:{period}"
                    for model_id, count in models.items():
                        pipeline.hincrby(redis_key, model_id, count)

                # Push 'tokens' data
                for period, models in self.local_buffer["tokens"].items():
                    redis_key = f"analytics:tokens:{period}"
                    for model_id, count in models.items():
                        pipeline.hincrby(redis_key, model_id, count)

                # Execute the pipeline in a separate thread
                await loop.run_in_executor(None, pipeline.execute)

                logger.info("Analytics data successfully synced to Redis.")
            except redis.exceptions.ConnectionError as e:
                logger.error("Redis connection error during sync: %s", e)
                raise e
            except Exception as e:
                logger.error("Unexpected error during Redis sync: %s", e)
                raise e

    async def _start_sync_task(self):
        """
        Periodically runs _sync_to_redis at a configurable interval.
        Also handles reconnections on ConnectionError.
        """
        while True:
            await asyncio.sleep(self.sync_interval)
            try:
                await self._sync_to_redis()
            except redis.exceptions.ConnectionError as e:
                logger.error("Redis connection error during scheduled sync: %s", e)
                await self._handle_redis_reconnection()
            except Exception as e:
                logger.error("Error during scheduled sync: %s", e)
                # Handle other errors as appropriate

    async def _handle_redis_reconnection(self):
        """
        Attempts to reconnect to Redis using exponential backoff.
        """
        loop = asyncio.get_running_loop()
        retry_count = 0
        delay = 1

        while retry_count < self.max_retries:
            try:
                logger.info(
                    "Attempting to reconnect to Redis (attempt %d)...", retry_count + 1
                )
                # Close existing connection
                await loop.run_in_executor(None, self.redis_client.close)
                # Create a new client
                self.redis_client = self._create_redis_client()
                # Test the new connection
                await loop.run_in_executor(None, self.redis_client.ping)
                logger.info("Successfully reconnected to Redis.")
                return
            except redis.exceptions.ConnectionError as e:
                logger.error("Reconnection attempt %d failed: %s", retry_count + 1, e)
                retry_count += 1
                await asyncio.sleep(delay)
                delay *= 2  # exponential backoff

        logger.critical(
            "Max reconnection attempts reached. Unable to reconnect to Redis."
        )

        # Optional: Keep retrying indefinitely instead of giving up.
        # while True:
        #     try:
        #         logger.info("Retrying to reconnect to Redis...")
        #         await loop.run_in_executor(None, self.redis_client.close)
        #         self.redis_client = self._create_redis_client()
        #         await loop.run_in_executor(None, self.redis_client.ping)
        #         logger.info("Reconnected to Redis after extended retries.")
        #         break
        #     except redis.exceptions.ConnectionError as e:
        #         logger.error("Extended reconnection attempt failed: %s", e)
        #         await asyncio.sleep(delay)
        #         delay = min(delay * 2, 60)  # Cap at 60 seconds or choose your own max

    async def close(self):
        """
        Closes the Redis client connection. Still wrapped in an async method to avoid blocking.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.redis_client.close)
        logger.info("Closed Redis connection.")
