import traceback
from fastapi import Body, FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
import os
import json
from datetime import datetime
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from mobile_api import mobile_router
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins or specify a list of allowed origins ggs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mobile_router)
ocr = PaddleOCR()
model = YOLO("best.pt")
names = model.names

area = [(1, 173), (62, 468), (608, 431), (364, 155)]
current_date = datetime.now().strftime('%Y-%m-%d')

VIOLATIONS_DIR = "violations_data"
VIOLATIONS_FILE = os.path.join(VIOLATIONS_DIR, "violations.json")

if not os.path.exists(VIOLATIONS_DIR):
    os.makedirs(VIOLATIONS_DIR)

if not os.path.exists(VIOLATIONS_FILE):
    with open(VIOLATIONS_FILE, "w") as f:
        json.dump({"violations": []}, f)

def initialize_violations_file():
    
    try:
        with open(VIOLATIONS_FILE, "w") as f:
            json.dump({"violations": []}, f, indent=2)
        print(f"Initialized violations file at {VIOLATIONS_FILE}")
    except Exception as e:
        print(f"Error initializing violations file: {str(e)}")
        traceback.print_exc()

if not os.path.exists(VIOLATIONS_FILE):
    initialize_violations_file()

def save_violations(violations_list: List[Dict[str, Any]]):
    
    if not violations_list:
        print("No violations to save")
        return
    
    try:
        if not os.path.exists(VIOLATIONS_FILE) or os.path.getsize(VIOLATIONS_FILE) == 0:
            data = {"violations": []}
        else:
            try:
                with open(VIOLATIONS_FILE, "r") as f:
                    data = json.load(f)
                if "violations" not in data:
                    data = {"violations": []}
            except json.JSONDecodeError:
                print("JSON file was corrupted, resetting it")
                data = {"violations": []}
        
        data["violations"].extend(violations_list)
        
        with open(VIOLATIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Successfully saved {len(violations_list)} violations. Total: {len(data['violations'])}")
        
    except Exception as e:
        print(f"Error saving violations: {str(e)}")
        traceback.print_exc()

@app.get("/violations/")
async def get_violations(limit: Optional[int] = None, offset: Optional[int] = 0):
    """
    Get all recorded violations with optional pagination
    
    - limit: Maximum number of violations to return
    - offset: Number of violations to skip
    """
    try:
        
        if not os.path.exists(VIOLATIONS_FILE) or os.path.getsize(VIOLATIONS_FILE) == 0:
            print("Violations file doesn't exist or is empty")
            return {"violations": []}
        
        with open(VIOLATIONS_FILE, "r") as f:
            content = f.read()
            if not content.strip():
                print("Violations file is empty")
                return {"violations": []}
            
            
            f.seek(0)
            data = json.load(f)
        
        if "violations" not in data:
            print("Violations key not found in JSON data")
            return {"violations": []}
            
        violations = data["violations"]
        print(f"Found {len(violations)} violations in file")
        
        
        if limit is not None:
            violations = violations[offset:offset + limit]
        elif offset > 0:
            violations = violations[offset:]
            
        return {"violations": violations}
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        
        initialize_violations_file()
        return {"violations": []}
    except Exception as e:
        print(f"Error retrieving violations: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving violations: {str(e)}")

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
            save_violations(violations)
            print("Violations saved successfully")
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
                cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
                text = perform_ocr(crop)
                print(f"Detected Number Plate: {text}")

                _, buffer = cv2.imencode('.jpg', crop)
                crop_base64 = base64.b64encode(buffer).decode('utf-8')

                current_time = datetime.now().strftime('%H-%M-%S-%f')[:12]
                
                if not os.path.exists(current_date):
                    os.makedirs(current_date)
                    
                crop_image_path = os.path.join(current_date, f"{text}_{current_time}.jpg")
                cv2.imwrite(crop_image_path, crop)

                violations.append({
                    "number_plate": text,
                    "timestamp": datetime.now().isoformat(),
                    "isHelmet": "No Helmet",
                    "cropped_image": crop_base64
                })
                processed_track_ids.add(numberplate_track_id)
        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)

    cap.release()
    return violations


@app.get("/violations/debug/")
async def debug_violations_file():
    """Debug endpoint to check the status of the violations file"""
    try:
        file_exists = os.path.exists(VIOLATIONS_FILE)
        file_size = os.path.getsize(VIOLATIONS_FILE) if file_exists else 0
        
        file_content = None
        violation_count = 0
        
        if file_exists and file_size > 0:
            with open(VIOLATIONS_FILE, "r") as f:
                try:
                    data = json.load(f)
                    violation_count = len(data.get("violations", []))
                    # Only return a small sample to avoid huge responses
                    if "violations" in data and data["violations"]:
                        file_content = {"sample": data["violations"][:2]}
                except json.JSONDecodeError:
                    file_content = "Invalid JSON"
        
        return {
            "file_exists": file_exists,
            "file_size_bytes": file_size,
            "file_path": os.path.abspath(VIOLATIONS_FILE),
            "violation_count": violation_count,
            "file_content_sample": file_content
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/violations/reset/")
async def reset_violations_file():
    """Reset the violations file to an empty state"""
    try:
        initialize_violations_file()
        return {"message": "Violations file has been reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting violations file: {str(e)}")

# @app.delete("/violations/")
# async def clear_violations():
#     """Clear all recorded violations"""
#     try:
#         with open(VIOLATIONS_FILE, "w") as f:
#             json.dump({"violations": []}, f)
#         return {"message": "All violations cleared successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error clearing violations: {str(e)}")







@app.post("/send_notification/")
async def send_notification(payload: Dict[Any, Any] = Body(...)):
    """
    Send a notification to the mobile app.
    Payload should include:
    - title: notification title
    - body: notification message
    - data: additional data (optional)
    - token: device token (optional)
    """
    try:
        print(f"Sending notification: {payload}")
        
        # Extract notification details
        title = payload.get("title")
        body = payload.get("body")
        data = payload.get("data", {})
        token = payload.get("token")
        
        if not title or not body:
            raise HTTPException(status_code=400, detail="Title and body are required")
        
        # This is a placeholder endpoint that logs the notification
        # In a real implementation, you would integrate with a push notification service
        # such as Firebase Cloud Messaging, OneSignal, etc.
        
        notification_log = {
            "timestamp": datetime.now().isoformat(),
            "payload": payload
        }
        
        # You might want to store notifications in a separate file or database
        print(f"Notification received: {notification_log}")
        
        # Mock a successful response
        notification_id = f"notification_{datetime.now().timestamp()}"
        
        return {
            "status": "success",
            "message": "Notification request received successfully",
            "notification_id": notification_id
        }
    except Exception as e:
        print(f"Error processing notification: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing notification: {str(e)}")
