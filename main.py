import base64

from fastapi import FastAPI
from pydantic import BaseModel

from utils.index import align_crop, align_inputs, detect_markers, detect_qr, extract_data, highlight

app = FastAPI()

class Request(BaseModel):
    scale: str
    itemCount: int
    images: list[str]

@app.get("/health")
def health():
    return {"status":"OK"}

@app.post("/scan")
def scan(data: Request):
    image = base64.b64decode(data.images[0].split(',')[1])
    
    markers = detect_markers(image)
    # print(markers)
    cropped_image = align_crop(image,markers)
    qr_data  = detect_qr(cropped_image)
    # print(qr_data)

    if(qr_data['scale'] != data.scale):
      raise ValueError("scale_mismatch")
    
    if(qr_data['page']['total'] != len(data.images)):
      raise ValueError("page_mismatched")
    
    inputs = align_inputs(cropped_image, data.itemCount)
    # print(inputs)
    data = extract_data(cropped_image, inputs)
    # print(data)
    highlights = [highlight(cropped_image, inputs, data)]

    return { "data": data, "highlights": highlights }