# filename: analytics.py

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

        # current_totals holds the absolute counters (loaded from Redis)
        self.current_totals = {
            "access": defaultdict(lambda: defaultdict(int)),
            "tokens": defaultdict(lambda: defaultdict(int)),
        }

        # new_increments holds only the new usage since last sync
        self.new_increments = {
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
        Fetches existing data from Redis into the current_totals buffer,
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
        Also includes "total" as a key for all-time tracking.
        """
        now = datetime.utcnow()
        day_key = now.strftime("%Y-%m-%d")
        # %U is the week number of year, with Sunday as the first day of the week
        # If you prefer ISO week, consider using %V or something else.
        week_key = f"{now.year}-W{now.strftime('%U')}"
        month_key = now.strftime("%Y-%m")
        year_key = now.strftime("%Y")
        return day_key, week_key, month_key, year_key, "total"

    async def access(self, model_id: str, tokens: int):
        """
        Records an access event and token usage for a specific model.

        Parameters:
        - model_id (str): The ID of the accessed model.
        - tokens (int): Number of tokens used in this access event.
        """
        # Validate inputs
        if not model_id or not isinstance(model_id, str):
            logger.warning("Invalid model_id provided: %s", model_id)
            return
        
        if tokens < 0:
            logger.warning("Negative token count provided for model %s: %d", model_id, tokens)
            tokens = 0
        
        keys = self._get_period_keys()

        async with self.lock:
            try:
                for period_key in keys:
                    # Increase new increments by the usage
                    self.new_increments["access"][period_key][model_id] += 1
                    self.new_increments["tokens"][period_key][model_id] += tokens

                    # Also update current_totals so that stats() are immediately up to date
                    self.current_totals["access"][period_key][model_id] += 1
                    self.current_totals["tokens"][period_key][model_id] += tokens
                    
                logger.debug("Recorded access for model %s: %d tokens", model_id, tokens)
            except Exception as e:
                logger.error("Error recording access for model %s: %s", model_id, e)

    async def stats(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Returns a copy of current statistics from the local buffer (absolute totals).
        """
        async with self.lock:
            try:
                # Return the current_totals, which includes everything loaded from Redis
                # plus all increments since the last sync.
                result = {
                    "access": {
                        period: dict(models)
                        for period, models in self.current_totals["access"].items()
                    },
                    "tokens": {
                        period: dict(models)
                        for period, models in self.current_totals["tokens"].items()
                    },
                }
                
                logger.debug("Retrieved stats for %d access periods and %d token periods", 
                           len(result["access"]), len(result["tokens"]))
                return result
            except Exception as e:
                logger.error("Error retrieving stats: %s", e)
                # Return empty structure if there's an error
                return {
                    "access": {},
                    "tokens": {},
                }

    async def _sync_from_redis(self):
        """
        Pulls existing analytics data from Upstash Redis into current_totals.
        Uses run_in_executor to avoid blocking the event loop.
        Also resets new_increments to avoid double counting after a restart.
        """
        loop = asyncio.get_running_loop()
        async with self.lock:
            logger.info("Starting sync from Upstash Redis...")
            
            # Reset local structures
            self.current_totals = {
                "access": defaultdict(lambda: defaultdict(int)),
                "tokens": defaultdict(lambda: defaultdict(int)),
            }
            self.new_increments = {
                "access": defaultdict(lambda: defaultdict(int)),
                "tokens": defaultdict(lambda: defaultdict(int)),
            }

            # Sync access data
            logger.debug("Scanning access analytics keys...")
            cursor = 0
            scan_attempts = 0
            max_scan_attempts = 100  # Prevent infinite loops
            
            while scan_attempts < max_scan_attempts:
                try:
                    scan_result = await loop.run_in_executor(
                        None,
                        partial(
                            self.redis_client.scan,
                            cursor=cursor,
                            match="analytics:access:*",
                            count=10000,
                        ),
                    )
                    
                    # Handle different return formats from Upstash Redis
                    if isinstance(scan_result, (list, tuple)) and len(scan_result) >= 2:
                        cursor, keys = scan_result[0], scan_result[1]
                    else:
                        logger.warning("Unexpected scan result format: %s", scan_result)
                        break
                        
                    # Convert cursor to int if it's a string
                    if isinstance(cursor, str):
                        cursor = int(cursor) if cursor.isdigit() else 0
                    
                    logger.debug("Scanned access keys: cursor=%s, found %d keys", cursor, len(keys) if keys else 0)
                    
                    if keys:
                        for key in keys:
                            # key is "analytics:access:<period>"
                            period = key.replace("analytics:access:", "")
                            data = await loop.run_in_executor(
                                None, partial(self.redis_client.hgetall, key)
                            )
                            # Ensure data is not None and handle empty results
                            if data:
                                for model_id, count_str in data.items():
                                    try:
                                        self.current_totals["access"][period][model_id] = int(count_str)
                                    except (ValueError, TypeError):
                                        logger.warning("Invalid count value for model %s in period %s: %s", model_id, period, count_str)
                                        self.current_totals["access"][period][model_id] = 0

                    if cursor == 0:
                        break
                        
                    scan_attempts += 1
                    
                except Exception as e:
                    logger.error("Error during access scan attempt %d: %s", scan_attempts, e)
                    break
                    
            if scan_attempts >= max_scan_attempts:
                logger.warning("Max scan attempts reached for access keys. Stopping scan.")

            # Sync token data
            logger.debug("Scanning token analytics keys...")
            cursor = 0
            scan_attempts = 0
            max_scan_attempts = 100  # Prevent infinite loops
            
            while scan_attempts < max_scan_attempts:
                try:
                    scan_result = await loop.run_in_executor(
                        None,
                        partial(
                            self.redis_client.scan,
                            cursor=cursor,
                            match="analytics:tokens:*",
                            count=1000,
                        ),
                    )
                    
                    # Handle different return formats from Upstash Redis
                    if isinstance(scan_result, (list, tuple)) and len(scan_result) >= 2:
                        cursor, keys = scan_result[0], scan_result[1]
                    else:
                        logger.warning("Unexpected scan result format: %s", scan_result)
                        break
                        
                    # Convert cursor to int if it's a string
                    if isinstance(cursor, str):
                        cursor = int(cursor) if cursor.isdigit() else 0
                    
                    logger.debug("Scanned token keys: cursor=%s, found %d keys", cursor, len(keys) if keys else 0)
                    
                    if keys:
                        for key in keys:
                            # key is "analytics:tokens:<period>"
                            period = key.replace("analytics:tokens:", "")
                            data = await loop.run_in_executor(
                                None, partial(self.redis_client.hgetall, key)
                            )
                            # Ensure data is not None and handle empty results
                            if data:
                                for model_id, count_str in data.items():
                                    try:
                                        self.current_totals["tokens"][period][model_id] = int(count_str)
                                    except (ValueError, TypeError):
                                        logger.warning("Invalid token count value for model %s in period %s: %s", model_id, period, count_str)
                                        self.current_totals["tokens"][period][model_id] = 0

                    if cursor == 0:
                        break
                        
                    scan_attempts += 1
                    
                except Exception as e:
                    logger.error("Error during token scan attempt %d: %s", scan_attempts, e)
                    break
                    
            if scan_attempts >= max_scan_attempts:
                logger.warning("Max scan attempts reached for token keys. Stopping scan.")
                
            logger.info("Completed sync from Upstash Redis. Loaded %d access periods, %d token periods.", 
                       len(self.current_totals["access"]), len(self.current_totals["tokens"]))

    async def _sync_to_redis(self):
        """
        Pushes only the new_increments to Upstash Redis (local -> Redis).
        We use HINCRBY to avoid double counting, ensuring we only add the difference.
        """
        loop = asyncio.get_running_loop()
        async with self.lock:
            try:
                sync_count = 0
                
                # For each (period, model_id, count) in new_increments, call HINCRBY
                for period, models in self.new_increments["access"].items():
                    redis_key = f"analytics:access:{period}"
                    for model_id, count in models.items():
                        if count > 0:  # Only sync positive counts
                            try:
                                await loop.run_in_executor(
                                    None,
                                    partial(
                                        self.redis_client.hincrby,
                                        redis_key,
                                        model_id,
                                        count,
                                    ),
                                )
                                sync_count += 1
                            except Exception as e:
                                logger.error("Failed to sync access count for model %s, period %s: %s", model_id, period, e)

                for period, models in self.new_increments["tokens"].items():
                    redis_key = f"analytics:tokens:{period}"
                    for model_id, count in models.items():
                        if count > 0:  # Only sync positive counts
                            try:
                                await loop.run_in_executor(
                                    None,
                                    partial(
                                        self.redis_client.hincrby,
                                        redis_key,
                                        model_id,
                                        count,
                                    ),
                                )
                                sync_count += 1
                            except Exception as e:
                                logger.error("Failed to sync token count for model %s, period %s: %s", model_id, period, e)

                # Reset new_increments after successful sync
                self.new_increments = {
                    "access": defaultdict(lambda: defaultdict(int)),
                    "tokens": defaultdict(lambda: defaultdict(int)),
                }

                logger.info("Analytics data successfully synced to Upstash Redis. Synced %d entries.", sync_count)
            except Exception as e:
                logger.error("Unexpected error during Upstash Redis sync: %s", e)
                raise e

    async def _start_sync_task(self):
        """
        Periodically runs _sync_to_redis at a configurable interval.
        Also attempts reconnection on any errors (though Upstash is HTTP-based,
        so it's stateless).
        """
        while True:
            await asyncio.sleep(self.sync_interval)
            try:
                await self._sync_to_redis()
            except Exception as e:
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
                # Optionally, do a test command if desired (Upstash has limited support).
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
        # Optionally, you can keep trying indefinitely here.

    async def close(self):
        """
        Closes the Upstash Redis client (although Upstash uses stateless HTTP).
        Still wrapped in async to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.redis_client.close)
        logger.info("Closed Upstash Redis client.")
