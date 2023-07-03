import base64

from fastapi import FastAPI
from pydantic import BaseModel

from utils.index import align_crop, align_inputs, detect_markers, detect_qr, extract_data, highlight

app = FastAPI()

@app.get("/health")
def health():
    return {"status":"OK"}

@app.post("/scan")
def scan(images: list[str]):
    image = base64.b64decode(images[0].split(',')[1])
    
    markers = detect_markers(image)
    # print(markers)
    cropped_image = align_crop(image,markers)
    qr_data  = detect_qr(cropped_image)
    # print(qr_data)
    
    inputs = align_inputs(cropped_image, qr_data['page']['current'] * 90)
    # print(inputs)
    choices = extract_data(cropped_image, inputs)
    # print(data)
    highlights = [highlight(cropped_image, inputs, choices)]

    return { "data": {"name": qr_data["scale"], "choices":choices}, "highlights": highlights }