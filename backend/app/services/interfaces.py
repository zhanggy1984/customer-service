"""对接层抽象接口。

Agent 只依赖这些 ABC 接口，不依赖具体实现：
- LocalOrderService（Phase 3）：直接操作 MySQL 模拟微服务
- RemoteOrderService（未来）：HTTP/gRPC 调真实微服务
切换方式：services/__init__.py 根据 SERVICE_MODE 注入实现，Agent 代码零改动。
"""
from abc import ABC, abstractmethod

from app.services.models import (
    ComplaintResult,
    ManualTicket,
    OrderInfo,
    RefundResult,
    ReturnResult,
)


class IOrderService(ABC):
    @abstractmethod
    async def query_order(self, order_id: str, user_id: int) -> OrderInfo | None:
        """按订单号查询订单，并校验归属（user_id 非本人返回 None）。"""

    @abstractmethod
    async def list_user_orders(self, user_id: int, limit: int = 5) -> list[OrderInfo]:
        """列出用户最近 limit 条订单（含商品明细）。"""


class IReturnService(ABC):
    @abstractmethod
    async def check_eligibility(self, order: OrderInfo, user_id: int) -> dict:
        """退货资格判定。

        返回 {"eligible": bool, "reason": str, "refund_amount": float, "items": [OrderItem]}
        校验项：订单归属 / 订单状态可退 / 未超 7 天 / 商品未退过 / 商品可退(returnable)。
        """

    @abstractmethod
    async def create_return(
        self,
        order: OrderInfo,
        user_id: int,
        items: list[str],
        reason: str,
        session_id: str,
    ) -> ReturnResult:
        """创建退货单（INSERT return_orders + 更新 order_items.status）。"""


class IRefundService(ABC):
    @abstractmethod
    async def check_refund_eligibility(self, order: OrderInfo) -> dict:
        """仅退款资格判定。

        三级判定：PAID=可退 / SHIPPED=需先拒收 / DELIVERED=必须走退货。
        返回 {"eligible": bool, "reason": str, "amount": float}
        """

    @abstractmethod
    async def create_refund(
        self,
        order: OrderInfo,
        user_id: int,
        reason: str,
        session_id: str,
    ) -> RefundResult:
        """创建仅退款单（INSERT refund_orders）。"""


class IComplaintService(ABC):
    @abstractmethod
    async def create_complaint(
        self,
        user_id: int,
        order_id: str | None,
        complaint_type: str,
        description: str,
        severity: str,
        session_id: str,
    ) -> ComplaintResult:
        """创建投诉工单（INSERT complaint_tickets）。"""


class ITicketService(ABC):
    @abstractmethod
    async def create_manual_ticket(
        self,
        user_id: int,
        order_id: str | None,
        session_id: str,
        message: str,
    ) -> ManualTicket:
        """创建人工工单（Agent 无法处理时兜底，当前写入 complaint_tickets 模拟）。"""
