import os
import json
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import traceback 

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"]) 

class DashboardViolationResponse(BaseModel):
    violation_sent: str;
    live_cameras: str;
    new_users: str;
    total_users: str;

@dashboard_router.get("/dashboard/response", response_model=DashboardViolationResponse)
async def get_dashboard_violations(
    limit: int = 100,
    offset: int = 0
):
    try:
        response = {
            "violation_sent": "10",
            "live_cameras": "5",
            "new_users": "3",
            "total_users": "50"
        }
        return response
    except Exception as e:
        print(f"Error getting dashboard violations: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")