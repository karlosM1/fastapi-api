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
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from paddleocr import PaddleOCR

from mobile_api import mobile_router
from dashboard_api import dashboard_router
from db import database, save_violation_db
from contextlib import asynccontextmanager
from dotenv import load_dotenv
# from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI()
DATABASE_URL = os.getenv("DATABASE_URL")
# app.mount("/2025-05-02", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="."), name="images")

app.include_router(mobile_router)
app.include_router(dashboard_router)

ocr = PaddleOCR()
model = YOLO("best.pt")
names = model.names

area = [(1, 173), (62, 468), (608, 431), (364, 155)]
current_date = datetime.now().strftime('%Y-%m-%d')

@app.on_event("startup")
async def startup():
    app.state.db_pool = await asyncpg.create_pool(
        dsn=DATABASE_URL
    )

@app.on_event("shutdown")
async def shutdown():
    await app.state.db_pool.close()

async def insert_violation(pool, violation):
    try:
        if isinstance(violation['detected_at'], str):
            violation['detected_at'] = datetime.fromisoformat(violation['detected_at'])

        await pool.execute('''
            INSERT INTO violations (plate_number, violation_type, detected_at, image_url, is_notified)
            VALUES ($1, $2, $3, $4, $5)
        ''',
            violation['plate_number'],
            violation['violation_type'],
            violation['detected_at'],
            violation['image_url'],
            False
        )
    except KeyError as ke:
        print(f"KeyError when inserting violation: {ke}")
        print(f"Violation data that caused the error: {violation}")
        raise

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        video_path = "final.mp4"
        with open(video_path, "wb") as buffer:
            buffer.write(file.file.read())

        violations = process_video(video_path)
        print(f"Detected {len(violations)} violations from video")

        if os.path.exists(video_path):
            os.remove(video_path)

        if violations:
            for violation in violations:
                print("Violation data before insert:", violation)
                await insert_violation(app.state.db_pool, violation)
            print("Violations saved to NeonDB successfully")
        else:
            print("No violations detected in the video")

        return {"violations": violations}

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Internal Server Error: {str(e)}"}
        )

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

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
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

                current_time = datetime.now().strftime('%H-%M-%S-%f')[:12]

                if not os.path.exists(current_date):
                    os.makedirs(current_date)

                crop_image_filename = f"{text}_{current_time}.jpg"
                crop_image_path = os.path.join(current_date, crop_image_filename)
                cv2.imwrite(crop_image_path, crop)

                violations.append({
                    "plate_number": text,
                    "detected_at": datetime.now(),
                    "violation_type": "No Helmet",
                    "image_url": f"{current_date}/{crop_image_filename}",
                    "is_notified": False
                })
                processed_track_ids.add(numberplate_track_id)

    cap.release()
    return violations
