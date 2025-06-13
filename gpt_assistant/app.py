from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RefundRequest(BaseModel):
    order_id: str

class PickupRequest(BaseModel):
    order_id: str
    preferred_date: str

class LogCaseRequest(BaseModel):
    user_id: str
    issue_type: str

@app.post("/trigger_refund")
def trigger_refund(req: RefundRequest):
    return {"message": f"Refund for Order {req.order_id} has been initiated."}

@app.post("/reschedule_pickup")
def reschedule_pickup(req: PickupRequest):
    return {"message": f"Pickup for Order {req.order_id} rescheduled to {req.preferred_date}."}

@app.post("/log_case")
def log_case(req: LogCaseRequest):
    return {"message": f"Case logged for User {req.user_id} regarding '{req.issue_type}'."}
