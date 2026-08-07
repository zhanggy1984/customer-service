"""对接层依赖注入。

SERVICE_MODE=local → Local 实现（直接操作 MySQL，Phase 3 使用）。
SERVICE_MODE=remote → Remote 实现（未来 HTTP/gRPC 调真实微服务）。
Agent 只 import 这里的服务对象，切换实现不改 Agent 代码。
"""
from app.config import settings

if settings.service_mode == "remote":
    raise NotImplementedError("SERVICE_MODE=remote 尚未实现，当前请使用 local")

from app.services.local_impl import (  # noqa: E402
    LocalComplaintService,
    LocalOrderService,
    LocalRefundService,
    LocalReturnService,
    LocalTicketService,
)
from app.services.interfaces import (  # noqa: E402
    IComplaintService,
    IOrderService,
    IRefundService,
    IReturnService,
    ITicketService,
)

order_service: IOrderService = LocalOrderService()
return_service: IReturnService = LocalReturnService()
refund_service: IRefundService = LocalRefundService()
complaint_service: IComplaintService = LocalComplaintService()
ticket_service: ITicketService = LocalTicketService()
