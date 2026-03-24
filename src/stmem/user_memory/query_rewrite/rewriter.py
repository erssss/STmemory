"""查询重写器实现"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from src.stmem.prompts import build_query_rewrite_prompt

logger = logging.getLogger(__name__)


@dataclass
class QueryRewriteResult:
    """查询重写结果"""
    original_query: str               # 原始查询
    rewritten_query: str              # 重写后的查询
    is_rewritten: bool               # 是否成功重写
    profile_used: Optional[str] = None  # 使用的用户配置文件
    error: Optional[str] = None      # 错误信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


class QueryRewriter:
    """基于用户配置文件的查询重写器"""

    def __init__(
        self,
        llm,
        config: Dict[str, Any],
    ):
        """
        初始化重写器。

        参数:
            llm: 用于生成重写的大语言模型实例
            config: QueryRewriteConfig配置字典
        """
        self.llm = llm
        self.config = config
        self.enabled = config.get('enabled', False)  # 是否启用重写功能
        self.custom_instructions = config.get('prompt')  # 自定义指令（非完整prompt）

    def rewrite(
        self,
        query: str,
        profile_content: Optional[str] = None
    ) -> QueryRewriteResult:
        """
        基于用户配置文件重写查询。

        参数:
            query: 原始查询字符串
            profile_content: 来自user_profiles表的用户配置文件文本

        返回:
            QueryRewriteResult: 重写结果
        """
        # 如果没有用户配置文件，跳过重写
        if not profile_content or not profile_content.strip():
            logger.debug("未提供profile_content，跳过查询重写")
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                is_rewritten=False,
            )

        # 如果查询为空或太短，跳过重写
        if not query or len(query.strip()) < 3:
            logger.debug("查询太短，跳过重写")
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                is_rewritten=False,
            )

        try:
            start_time = time.time()  # 开始计时

            # 构建提示词
            prompt = build_query_rewrite_prompt(
                profile_content=profile_content,
                query=query,
                custom_instructions=self.custom_instructions,
            )

            # 调用LLM进行重写
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": "你是一个有用的查询重写助手。"},
                    {"role": "user", "content": prompt}
                ]
            )

            rewritten = response.strip()  # 去除空白字符
            elapsed = time.time() - start_time  # 计算耗时

            # 记录重写结果（始终记录）
            logger.info(f"查询重写: '{query}' -> '{rewritten}' (耗时 {elapsed:.2f}秒)")

            return QueryRewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                is_rewritten=True,
                profile_used=profile_content,
                metadata={"rewrite_time_seconds": elapsed}  # 记录重写耗时
            )

        except Exception as e:
            logger.error(f"查询重写失败: {e}，回退到原始查询")

            # 出错时回退到原始查询
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                is_rewritten=False,
                error=str(e)  # 记录错误信息
            )