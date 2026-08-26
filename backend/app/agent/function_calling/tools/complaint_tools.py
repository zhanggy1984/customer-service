"""投诉工具。内部调用对接层 IComplaintService。"""
from app.services import complaint_service


async def create_complaint(params: dict, user_id: int, session_id: str) -> dict:
    description = params.get("description", "")
    if not description:
        return {"ok": False, "data": None, "error": {"code": "missing_description", "message": "缺少 description 参数"}}
    result = await complaint_service.create_complaint(
        user_id=user_id,
        order_id=params.get("order_id"),
        complaint_type=params.get("complaint_type", ""),
        description=description,
        severity=params.get("severity", "MEDIUM"),
        session_id=session_id,
    )
    return {
        "ok": True,
        "data": {
            "success": result.success,
            "ticket_id": result.ticket_id,
            "severity": result.severity,
            "message": result.message,
        },
        "error": None,
    }
