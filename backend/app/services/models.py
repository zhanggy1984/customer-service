"""对接层领域模型。

Agent / 对接层之间传递的纯数据对象，不绑定数据库表结构，
Local 实现从 MySQL 行构造，未来 Remote 实现从 HTTP/gRPC 响应构造。
"""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OrderItem:
    id: int
    item_id: str
    name: str
    price: float
    quantity: int
    returnable: bool
    status: str = "NORMAL"  # NORMAL/RETURN_REQUESTED/RETURNED/REFUNDED


@dataclass
class OrderInfo:
    order_id: str
    user_id: int
    status: str  # PAID/SHIPPED/DELIVERED/CANCELLED
    total_amount: float
    shipping_address: str = ""
    created_at: datetime | None = None
    delivered_at: datetime | None = None
    db_id: int = 0  # 数据库主键，Local 实现内部用于关联 order_items
    items: list[OrderItem] = field(default_factory=list)

    def is_owner(self, user_id: int) -> bool:
        return self.user_id == user_id


@dataclass
class ReturnResult:
    success: bool
    return_id: str | None = None
    status: str = "PENDING"  # APPROVED/PENDING/REJECTED
    refund_amount: float = 0.0
    message: str = ""


@dataclass
class RefundResult:
    success: bool
    refund_id: str | None = None
    status: str = "PENDING"
    amount: float = 0.0
    message: str = ""


@dataclass
class ComplaintResult:
    success: bool
    ticket_id: str | None = None
    severity: str = "LOW"
    message: str = ""


@dataclass
class ManualTicket:
    ticket_id: str
    message: str = ""
