import os
import json
from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import traceback 

mobile_router = APIRouter(prefix="/mobile")


@mobile_router.get("/userviolation/{clerk_id}")
async def get_violations(clerk_id: str, request: Request):
    async with request.app.state.db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT plate_number FROM users WHERE clerk_id = $1", clerk_id
        )

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        plate_number = user["plate_number"]

        violations = await conn.fetch(
            "SELECT * FROM violations WHERE plate_number = $1 ORDER BY detected_at DESC",
            plate_number
        )

        return {
            "plate_number": plate_number,
            "violations": [dict(v) for v in violations]
        }
# class ViolationResponse(BaseModel):
#     number_plate: str
#     timestamp: str
#     isHelmet: str
#     cropped_image: str

# class ViolationsListResponse(BaseModel):
#     violations: List[ViolationResponse]
#     total_count: int

# def get_violations_from_storage(
#     limit: int = 100,
#     offset: int = 0
# ):
#     try:
#         all_violations = []
        
#         violations_file = os.path.join("violations_data", "violations.json")
        
#         print(f"Looking for violations in: {violations_file}")
#         print(f"Current working directory: {os.getcwd()}")
        
#         if os.path.exists(violations_file):
#             print(f"File {violations_file} exists")
#             try:
#                 with open(violations_file, 'r', encoding='utf-8') as f:
#                     file_content = f.read()
#                     if not file_content.strip():
#                         print("File is empty")
#                         return {"violations": [], "total_count": 0}
                    
#                     all_violations = json.loads(file_content)
#                     print(f"Found {len(all_violations)} violations")
#             except json.JSONDecodeError as e:
#                 print(f"JSON decode error: {str(e)}")
#                 print(f"First 100 characters of file: {file_content[:100] if 'file_content' in locals() else 'Not available'}")
#                 return {"violations": [], "total_count": 0}
#             except Exception as e:
#                 print(f"Error reading violations: {str(e)}")
#                 traceback.print_exc()
#                 return {"violations": [], "total_count": 0}
#         else:
#             print(f"File {violations_file} does not exist")
#             try:
#                 if os.path.exists("violations_data"):
#                     print(f"Contents of violations_data directory: {os.listdir('violations_data')}")
#                 else:
#                     print("violations_data directory does not exist")
#                     print(f"Contents of current directory: {os.listdir('.')}")
#             except Exception as e:
#                 print(f"Error listing directory: {str(e)}")
            
#             return {"violations": [], "total_count": 0}
        
#         if not isinstance(all_violations, list):
#             print(f"all_violations is not a list, it's a {type(all_violations)}")
#             if isinstance(all_violations, dict) and "violations" in all_violations:
#                 all_violations = all_violations["violations"]
#             else:
#                 return {"violations": [], "total_count": 0}
        
#         try:
#             all_violations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
#         except Exception as e:
#             print(f"Error sorting violations: {str(e)}")
        
#         try:
#             paginated_violations = all_violations[offset:offset+limit]
#         except Exception as e:
#             print(f"Error applying pagination: {str(e)}")
#             paginated_violations = all_violations[:limit]
        
#         return {
#             "violations": paginated_violations,
#             "total_count": len(all_violations)
#         }
#     except Exception as e:
#         print(f"Unexpected error in get_violations_from_storage: {str(e)}")
#         traceback.print_exc()
#         return {"violations": [], "total_count": 0}

# @mobile_router.get("/violations", response_model=ViolationsListResponse)
# async def get_violations(
#     limit: int = 50,
#     offset: int = 0
# ):

#     try:
#         result = get_violations_from_storage(
#             limit=limit,
#             offset=offset
#         )
        
#         return result
#     except Exception as e:
#         print(f"Error in get_violations endpoint: {str(e)}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve violations: {str(e)}")

# @mobile_router.get("/test")
# async def test_endpoint():
#     return {"status": "ok", "message": "API is working"}



# @mobile_router.get("/violations/{clerk_id}")
# async def get_violations(clerk_id: str):
#     async with mobile_router.state.db.acquire() as conn:
#         user = await conn.fetchrow("SELECT plate_number FROM users WHERE clerk_id = $1", clerk_id)

#         if not user:
#             raise HTTPException(status_code=404, detail="User not found")

#         plate_number = user["plate_number"]

#         violations = await conn.fetch("""
#             SELECT * FROM violations
#             WHERE plate_number = $1
#             ORDER BY detected_at DESC
#         """, plate_number)

#         return {"plate_number": plate_number, "violations": [dict(v) for v in violations]}


