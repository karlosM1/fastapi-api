from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
import os
from datetime import datetime
import base64
from io import BytesIO

app = FastAPI()

ocr = PaddleOCR()
model = YOLO("best.pt")
names = model.names

area = [(1, 173), (62, 468), (608, 431), (364, 155)]
current_date = datetime.now().strftime('%Y-%m-%d')


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
        video_path = "final.mp4"  
        with open(video_path, "wb") as buffer:
            buffer.write(file.file.read())

        violations = process_video(video_path)

        os.remove(video_path)

        return {"violations": violations}

    except Exception as e:
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

# @app.get("/detection")
# async def get_violation():
#     sample_response = {
#         "plate": "1234ABC",
#         "isHelmet": "No"
#     }
#     return {"message": sample_response}



