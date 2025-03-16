from fastapi import FastAPI, HTTPException, requests
from pydantic import BaseModel

app = FastAPI()

MODEL_API_URL = "/best.pt"

@app.get("/violation")
async def get_violation():
    try:
        response = requests.get(MODEL_API_URL)
        # load model data
        # load video
        #predict on video
        # return prediction
        return {"message": model_data}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error contacting model API: {str(e)}")


@app.get("/detection")
async def get_violation():
    sample_response = {
        "plate": "1234ABC",
        "isHelmet": "No"
    }
    return {"message": sample_response}