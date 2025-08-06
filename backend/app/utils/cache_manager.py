import os
from typing import Dict, Optional
from functools import wraps
from dotenv import load_dotenv
import hashlib
import json
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class CacheManager:
    """
    缓存管理类，用于缓存高频查询的检索结果和生成回答。
    使用带过期时间的内存缓存实现。
    """
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = int(os.getenv('CACHE_TTL', 3600))

    def _generate_cache_key(self, query: str, params: Dict) -> str:
        """
        生成缓存键，基于query哈希值和params组合。

        Args:
            query (str): 查询内容。
            params (Dict): 查询参数。

        Returns:
            str: 唯一的缓存键。
        """
        params_str = json.dumps(params, sort_keys=True)
        combined = f'{query}{params_str}'
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def cache_query(self, query: str, params: Dict, result: Dict) -> None:
        """
        缓存查询结果。

        Args:
            query (str): 查询内容。
            params (Dict): 查询参数。
            result (Dict): 查询结果。
        """
        cache_key = self._generate_cache_key(query, params)
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        logger.info(f'缓存查询结果，缓存键: {cache_key}')

    def get_cached(self, query: str, params: Dict) -> Optional[Dict]:
        """
        根据查询内容和参数获取缓存。

        Args:
            query (str): 查询内容。
            params (Dict): 查询参数。

        Returns:
            Optional[Dict]: 缓存结果，如果缓存不存在或已过期则返回None。
        """
        cache_key = self._generate_cache_key(query, params)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry['timestamp'] <= self.cache_ttl:
                logger.info(f'获取缓存结果，缓存键: {cache_key}')
                return entry['result']
            else:
                del self.cache[cache_key]
                logger.info(f'缓存已过期，删除缓存键: {cache_key}')
        return None

    def clear_expired_cache(self) -> None:
        """
        清除所有过期的缓存。
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry['timestamp'] > self.cache_ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        logger.info(f'清除了 {len(expired_keys)} 个过期缓存项')

# 装饰器示例，可用于函数缓存
def cacheable(func):
    """
    缓存装饰器，用于缓存函数调用结果。
    """
    cache_mgr = CacheManager()

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 简单起见，使用参数的字符串表示作为查询内容
        query = str(args) + str(kwargs)
        params = {}
        cached_result = cache_mgr.get_cached(query, params)
        if cached_result is not None:
            return cached_result
        result = func(*args, **kwargs)
        cache_mgr.cache_query(query, params, result)
        return result

    return wrapper