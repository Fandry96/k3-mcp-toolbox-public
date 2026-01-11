"""
K3 Firehose: Federated Bloom Filter Client
Connects to RedisBloom for distributed, multi-tenant probabilistic filtering.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Placeholder for when Redis is not available
class MockBloomFilter:
    """In-memory mock for local development without Redis."""

    def __init__(self):
        self._data: set = set()
        logger.warning("FederatedBloomFilter running in MOCK mode (no Redis).")

    def add(self, subreddit: str, item: str) -> bool:
        key = f"{subreddit}:{item}"
        self._data.add(key)
        return True

    def check(self, subreddit: str, item: str) -> bool:
        key = f"{subreddit}:{item}"
        return key in self._data


class FederatedBloomFilter:
    """
    Connects to a shared RedisBloom instance for cross-subreddit defense.

    Uses multi-tenant key prefixing: `{subreddit}:badactors`

    Environment Variables:
        REDIS_CLOUD_URL: Connection string for Redis Cloud (redis://user:pass@host:port)

    Usage:
        fdn = FederatedBloomFilter()
        fdn.add("news", "spammer_username")
        is_spammer = fdn.check("news", "spammer_username")  # True
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.environ.get("REDIS_CLOUD_URL")
        self._client = None
        self._mock_fallback = None

        if self.redis_url:
            try:
                import redis

                self._client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self._client.ping()
                logger.info("FederatedBloomFilter: Connected to Redis Cloud.")
            except ImportError:
                logger.error("redis library not installed. Run: pip install redis")
                self._mock_fallback = MockBloomFilter()
            except Exception as e:
                logger.error(f"FederatedBloomFilter: Redis connection failed: {e}")
                self._mock_fallback = MockBloomFilter()
        else:
            logger.warning("REDIS_CLOUD_URL not set. Using Mock mode.")
            self._mock_fallback = MockBloomFilter()

    def _get_key(self, subreddit: str) -> str:
        """Generates the multi-tenant key for a subreddit's ban filter."""
        return f"fdn:{subreddit}:badactors"

    def add(self, subreddit: str, item: str) -> bool:
        """
        Adds an item (username, phrase, etc.) to the subreddit's Bloom filter.

        Args:
            subreddit: The subreddit namespace.
            item: The item to add to the filter.

        Returns:
            True if successfully added, False otherwise.
        """
        if self._mock_fallback:
            return self._mock_fallback.add(subreddit, item)

        key = self._get_key(subreddit)
        try:
            # BF.ADD key item
            # Returns 1 if newly added, 0 if already exists (probably)
            self._client.execute_command("BF.ADD", key, item)
            return True
        except Exception as e:
            logger.error(f"BF.ADD failed for {key}: {e}")
            return False

    def check(self, subreddit: str, item: str) -> bool:
        """
        Checks if an item is probably in the subreddit's Bloom filter.

        Args:
            subreddit: The subreddit namespace.
            item: The item to check.

        Returns:
            True if probably in set (possible false positive).
            False if definitely NOT in set (guaranteed).
        """
        if self._mock_fallback:
            return self._mock_fallback.check(subreddit, item)

        key = self._get_key(subreddit)
        try:
            # BF.EXISTS key item
            # Returns 1 if probably exists, 0 if definitely not
            result = self._client.execute_command("BF.EXISTS", key, item)
            return result == 1
        except Exception as e:
            logger.error(f"BF.EXISTS failed for {key}: {e}")
            # Fail open: if we can't check, assume not in filter
            return False

    def bulk_add(self, subreddit: str, items: list[str]) -> int:
        """
        Adds multiple items to the filter at once.

        Args:
            subreddit: The subreddit namespace.
            items: List of items to add.

        Returns:
            Number of items successfully added.
        """
        if self._mock_fallback:
            for item in items:
                self._mock_fallback.add(subreddit, item)
            return len(items)

        key = self._get_key(subreddit)
        try:
            # BF.MADD key item1 item2 ...
            self._client.execute_command("BF.MADD", key, *items)
            return len(items)
        except Exception as e:
            logger.error(f"BF.MADD failed for {key}: {e}")
            return 0
