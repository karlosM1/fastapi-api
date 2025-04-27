# app.py
import traceback
import os
import asyncpg
import cv2
import numpy as np
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR

from mobile_api import mobile_router
from dashboard_api import dashboard_router
# from db import database, save_violation_db
# from contextlib import asynccontextmanager

from dotenv import load_dotenv

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(mobile_router)
app.include_router(dashboard_router)

# Load ML Models
ocr = PaddleOCR()
model = YOLO("best.pt")
names = model.names

area = [(1, 173), (62, 468), (608, 431), (364, 155)]
current_date = datetime.now().strftime('%Y-%m-%d')


DATABASE_URL = os.getenv("DATABASE_URL")

@app.on_event("startup")
async def startup():
    app.state.db_pool = await asyncpg.create_pool(
        dsn=DATABASE_URL
    )

# Close the connection pool on shutdown
@app.on_event("shutdown")
async def shutdown():
    await app.state.db_pool.close()



# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     app.state.db_pool = await asyncpg.create_pool(
#         dsn=DATABASE_URL
#     )
#     yield
#     await app.state.db_pool.close()


# app = FastAPI(lifespan=lifespan)

async def insert_violation(pool, violation: dict):
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO violations (plate_number, violation_type, detected_at, image_url)
                VALUES ($1, $2, $3, $4)
                """,
                violation["plate_number"],      # <-- match keys
                violation["violation_type"],
                violation["detected_at"],  
                violation["cropped_image"],       # <-- match key here
                False
            )
    except Exception as e:
        print(f"Error inserting violation: {e}")
        raise
    

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        video_path = "final.mp4"  
        with open(video_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Process the video and detect violations
        violations = process_video(video_path)
        print(f"Detected {len(violations)} violations from video")

        if os.path.exists(video_path):
            os.remove(video_path)
            
        if violations:
            # Insert each violation into NeonDB
            for violation in violations:
                await insert_violation(app.state.db_pool, violation)
            print("Violations saved to NeonDB successfully")
        else:
            print("No violations detected in the video")

        return {"violations": violations}

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def perform_ocr(image_array):
    if image_array is None or image_array.size == 0:
        raise ValueError("Invalid image for OCR")

    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    results = ocr.ocr(image_array, rec=True)
    detected_text = [result[1][0] for result in results[0] if result[1]]

    return ''.join(detected_text)

def process_video(video_path):
    processed_track_ids = set()
    cap = cv2.VideoCapture(video_path)
    violations = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.track(frame, persist=True)

        no_helmet_detected = False
        numberplate_box = None
        numberplate_track_id = None

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist() 

            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 0:
                    if c == 'no-helmet':
                        no_helmet_detected = True
                    elif c == 'numberplate':
                        numberplate_box = box
                        numberplate_track_id = track_id

            if no_helmet_detected and numberplate_box is not None and numberplate_track_id not in processed_track_ids:
                x1, y1, x2, y2 = numberplate_box
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (120, 85))
                text = perform_ocr(crop)
                print(f"Detected Number Plate: {text}")

                # Convert to base64
                _, buffer = cv2.imencode('.jpg', crop)
                crop_base64 = base64.b64encode(buffer).decode('utf-8')

                current_time = datetime.now().strftime('%H-%M-%S-%f')[:12]
                
                # Save cropped image locally (optional)
                if not os.path.exists(current_date):
                    os.makedirs(current_date)
                    
                crop_image_path = os.path.join(current_date, f"{text}_{current_time}.jpg")
                cv2.imwrite(crop_image_path, crop)

                # Append violation to list
                violations.append({
                    "plate_number": text,                    # <-- MATCHED to DB
                    "violation_type": "No Helmet",            # <-- MATCHED to DB
                    "detected_at": datetime.now(),            # <-- MATCHED to DB
                    "cropped_image": crop_base64 
                })
                processed_track_ids.add(numberplate_track_id)

    cap.release()
    return violations

