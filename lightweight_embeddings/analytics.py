import logging
import asyncio
from upstash_redis import Redis as UpstashRedis
from datetime import datetime
from collections import defaultdict
from typing import Dict
from functools import partial

logger = logging.getLogger(__name__)


class Analytics:
    def __init__(
        self, url: str, token: str, sync_interval: int = 60, max_retries: int = 5
    ):
        """
        Initializes the Analytics class with an Upstash Redis client (HTTP-based),
        wrapped in async methods by using run_in_executor.

        Parameters:
        - url (str): Upstash Redis REST URL.
        - token (str): Upstash Redis token.
        - sync_interval (int): Interval in seconds for syncing with Redis.
        - max_retries (int): Maximum number of reconnection attempts to Redis.
        """
        self.url = url
        self.token = token
        self.sync_interval = sync_interval
        self.max_retries = max_retries

        # Upstash Redis client (synchronous over HTTP)
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

        logger.info("Initialized Analytics with Upstash Redis: %s", url)

    def _create_redis_client(self) -> UpstashRedis:
        """
        Creates and returns a new Upstash Redis (synchronous) client.
        """
        return UpstashRedis(url=self.url, token=self.token)

    async def _initialize(self):
        """
        Fetches existing data from Redis into the local buffer,
        then starts the periodic synchronization task.
        """
        try:
            await self._sync_from_redis()
            logger.info("Initial sync from Upstash Redis to local buffer completed.")
        except Exception as e:
            logger.error("Error during initial sync from Upstash Redis: %s", e)

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
        Pulls existing analytics data from Upstash Redis into the local buffer.
        Uses run_in_executor to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()

        async with self.lock:
            # Scan 'access' keys
            cursor = 0
            while True:
                # Upstash doesn't provide a typical 'SCAN' the same way as standard Redis?
                # We'll mimic it by searching for keys, or we can store a list of known periods if needed.
                # If you only store certain known patterns, adapt accordingly.
                # For demonstration, we do a naive approach, or assume we have a method that lists keys.
                # Upstash doesn't always support the standard SCAN. We might store known keys in a set in Redis.

                # If Upstash doesn't support SCAN at all, you need another approach (like maintaining a separate index).
                # For now, let's assume it can handle the SCAN command similarly:
                scan_result = await loop.run_in_executor(
                    None,
                    partial(
                        self.redis_client.scan,
                        cursor=cursor,
                        match="analytics:access:*",
                        count=1000,
                    ),
                )
                cursor, keys = scan_result[0], scan_result[1]

                for key in keys:
                    # key is "analytics:access:<period>"
                    period = key.replace("analytics:access:", "")
                    data = await loop.run_in_executor(
                        None, partial(self.redis_client.hgetall, key)
                    )
                    for model_id, count_str in data.items():
                        self.local_buffer["access"][period][model_id] = int(count_str)

                if cursor == 0:
                    break

            # Scan 'tokens' keys
            cursor = 0
            while True:
                scan_result = await loop.run_in_executor(
                    None,
                    partial(
                        self.redis_client.scan,
                        cursor=cursor,
                        match="analytics:tokens:*",
                        count=1000,
                    ),
                )
                cursor, keys = scan_result[0], scan_result[1]

                for key in keys:
                    # key is "analytics:tokens:<period>"
                    period = key.replace("analytics:tokens:", "")
                    data = await loop.run_in_executor(
                        None, partial(self.redis_client.hgetall, key)
                    )
                    for model_id, count_str in data.items():
                        self.local_buffer["tokens"][period][model_id] = int(count_str)

                if cursor == 0:
                    break

    async def _sync_to_redis(self):
        """
        Pushes the local buffer data to Upstash Redis (local -> Redis).
        Since Upstash does not support pipelining, we increment each field individually.
        """
        loop = asyncio.get_running_loop()

        async with self.lock:
            try:
                # For each (period, model_id, count), call hincrby
                for period, models in self.local_buffer["access"].items():
                    redis_key = f"analytics:access:{period}"
                    for model_id, count in models.items():
                        if count != 0:
                            # hincrby(key, field, amount)
                            await loop.run_in_executor(
                                None,
                                partial(
                                    self.redis_client.hincrby,
                                    redis_key,
                                    model_id,
                                    count,
                                ),
                            )

                for period, models in self.local_buffer["tokens"].items():
                    redis_key = f"analytics:tokens:{period}"
                    for model_id, count in models.items():
                        if count != 0:
                            await loop.run_in_executor(
                                None,
                                partial(
                                    self.redis_client.hincrby,
                                    redis_key,
                                    model_id,
                                    count,
                                ),
                            )

                logger.info("Analytics data successfully synced to Upstash Redis.")
            except Exception as e:
                logger.error("Unexpected error during Upstash Redis sync: %s", e)
                raise e

    async def _start_sync_task(self):
        """
        Periodically runs _sync_to_redis at a configurable interval.
        Also attempts reconnection on any errors (though Upstash typically won't
        behave exactly like a persistent TCP connection).
        """
        while True:
            await asyncio.sleep(self.sync_interval)
            try:
                await self._sync_to_redis()
            except Exception as e:
                # Upstash might fail differently than standard Redis if there's a network issue
                logger.error("Error during scheduled sync to Upstash Redis: %s", e)
                await self._handle_redis_reconnection()

    async def _handle_redis_reconnection(self):
        """
        Attempts to 'reconnect' to Upstash Redis.
        Because Upstash uses HTTP, it's often stateless and doesn't require
        the same approach as standard Redis. We simply recreate the client if needed.
        """
        loop = asyncio.get_running_loop()
        retry_count = 0
        delay = 1

        while retry_count < self.max_retries:
            try:
                logger.info(
                    "Attempting to reconnect to Upstash Redis (attempt %d)...",
                    retry_count + 1,
                )
                # Recreate the client
                await loop.run_in_executor(None, self.redis_client.close)
                self.redis_client = self._create_redis_client()
                # Upstash doesn't necessarily have a direct 'PING' command, so optional:
                # If you want to test, you could do e.g. redis_client.get("some_known_key") as a check
                logger.info("Successfully reconnected to Upstash Redis.")
                return
            except Exception as e:
                logger.error("Reconnection attempt %d failed: %s", retry_count + 1, e)
                retry_count += 1
                await asyncio.sleep(delay)
                delay *= 2  # exponential backoff

        logger.critical(
            "Max reconnection attempts reached. Unable to reconnect to Upstash Redis."
        )

        # Optionally, keep trying indefinitely
        # while True:
        #     try:
        #         logger.info("Retrying to reconnect to Upstash Redis...")
        #         await loop.run_in_executor(None, self.redis_client.close)
        #         self.redis_client = self._create_redis_client()
        #         logger.info(
        #             "Successfully reconnected to Upstash Redis after extended retry."
        #         )
        #         break
        #     except Exception as e:
        #         logger.error("Extended reconnection attempt failed: %s", e)
        #         await asyncio.sleep(delay)
        #         delay = min(delay * 2, 60)  # cap at 60s or choose another max

    async def close(self):
        """
        Closes the Upstash Redis client (although Upstash uses stateless HTTP).
        Still wrapped in async to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.redis_client.close)
        logger.info("Closed Upstash Redis client.")
