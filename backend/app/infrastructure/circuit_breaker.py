"""熔断器：判定 LLM 是否全部不可用，触发规则引擎降级。"""
from app.infrastructure.deepseek_keypool import KeyPool


async def should_fallback(pool: KeyPool) -> bool:
    """KeyPool.all_cooling() == True → 熔断。"""
    return await pool.all_cooling()
