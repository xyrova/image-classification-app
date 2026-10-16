"""
FastAPI inference service for the digit classifier.

Applies the same preprocessing as the DVC pipeline (grayscale -> resize ->
normalize -> flatten -> feature selector) and returns the predicted class.

Run from the project root:
    uvicorn app.main:app --reload
"""
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
# model.joblib pickles a NumpyMLP from src/mlp.py, so `mlp` must be importable
sys.path.insert(0, str(ROOT / "src"))
from mlp import NumpyMLP  # noqa: F401 -- needed to unpickle model.joblib

params = yaml.safe_load((ROOT / "params.yaml").read_text())["data_processing"]
selector = joblib.load(ROOT / "models" / "feature_selector.joblib")
model = joblib.load(ROOT / "models" / "model.joblib")["model"]

app = FastAPI(title="Image Classification API")


def preprocess(image: Image.Image) -> np.ndarray:
    size = params["image_size"]
    img = image.convert("L" if params["grayscale"] else "RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)
    if params["normalize"]:
        x = x / 255.0
    return selector.transform(x.reshape(1, -1)).astype(np.float32)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    probs = model.predict_proba(preprocess(image))[0]
    pred = int(np.argmax(probs))
    return {"prediction": pred, "confidence": float(probs[pred])}
