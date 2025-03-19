from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
from datetime import datetime
import shutil
import os

app = FastAPI()

ocr = PaddleOCR()
model = YOLO("best.pt")
names = model.names

area = [(1, 173), (62, 468), (608, 431), (364, 155)]



# @app.get("/violation")
# async def get_violation():
#     try:
#         response = requests.get(MODEL_API_URL)
#         # load model data
#         # load video
#         #predict on video
#         # return prediction response
#     except requests.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"Error contacting model API: {str(e)}")
    

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        video_path = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        with open(video_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Process the video and get results
        violations = process_video(video_path)

        # Clean up: remove video file after processing
        os.remove(video_path)

        return {"violations": violations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def perform_ocr(image_array):
    """Extract text from an image using OCR."""
    if image_array is None:
        return None
    
    results = ocr.ocr(image_array, rec=True)
    detected_text = []

    if results and results[0] is not None:
        for result in results[0]:
            text = result[1][0]
            detected_text.append(text)

    return "".join(detected_text)

def process_video(video_path):
    """Process video to detect helmet violations and number plates."""
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

            if no_helmet_detected and numberplate_box:
                x1, y1, x2, y2 = numberplate_box
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (120, 85))
                text = perform_ocr(crop)

                violations.append({
                    "number_plate": text,
                    "timestamp": datetime.now().isoformat(),
                    "isHelmet": "No Helmet"
                })

    cap.release()
    return violations

@app.get("/detection")
async def get_violation():
    sample_response = {
        "plate": "1234ABC",
        "isHelmet": "No"
    }
    return {"message": sample_response}