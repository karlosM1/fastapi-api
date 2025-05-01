import os
import json
import asyncpg
from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import traceback 

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"]) 

@dashboard_router.get("/total-violations")
async def get_total_violations(request: Request):
    try:
        async with request.app.state.db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) AS total_violations FROM violations;")
            return {"total_violations": row["total_violations"]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch total violations")


@dashboard_router.get("/total-users")
async def get_total_users(request: Request):
    try:
        async with request.app.state.db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) AS total_users FROM users;")
            return {"total_users": row["total_users"]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch total users")
    
@dashboard_router.get("/violations-per-month")
async def get_violations_per_month(request: Request):
    try:
        async with request.app.state.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    DATE_TRUNC('month', detected_at) AS month,
                    COUNT(*) AS count
                FROM violations
                GROUP BY month
                ORDER BY month;
            """)
            return [
                {
                    "month": row["month"].strftime("%Y-%m"),
                    "count": row["count"]
                } for row in rows
            ]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch violations per month")
    
    


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
            "total_users": "20"
        }
        return response
    except Exception as e:
        print(f"Error getting dashboard violations: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")