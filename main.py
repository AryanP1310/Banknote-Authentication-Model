from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from predict import predict

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Banknote Authenticator")


class Note(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_note(note: Note):
    return predict(note.model_dump())