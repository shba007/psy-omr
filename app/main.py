import base64

from fastapi import FastAPI, HTTPException

from utils.scan import align_crop, align_inputs, detect_markers, detect_qr, extract_data, highlight

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/scan")
def scan(images: list[str]):
    meta_data = []
    cropped_images = []
    for image in images:
        image = base64.b64decode(images[0].split(',')[1])

        markers = detect_markers(image)
        cropped_image = align_crop(image, markers)
        cropped_images.append(cropped_image)
        meta_data.append(detect_qr(cropped_image))

    if not all(item["scale"] == meta_data[0]["scale"] for item in meta_data):
        raise HTTPException(status_code=409, detail="Pages are not of a same scale")

    total_choice_indexes = set()
    for data in meta_data:
        start = data["choice"]["start"]
        count = data["choice"]["count"]

        choice_indexes = list(range(start, start + count))
        total_choice_indexes.update(choice_indexes)

    if not set(range(1, meta_data[0]["choice"]["total"])).issubset(total_choice_indexes):
        raise HTTPException(status_code=400, detail="Insufficient number of pages")

    highlights = []
    choices = []
    for index, cropped_image in enumerate(cropped_images):
        # print(meta_data)
        option_count = meta_data[index]['option']
        start = meta_data[index]['choice']['start']
        choice_count = meta_data[index]['choice']['count']
        inputs = align_inputs(cropped_image, option_count, start, choice_count)
        # print(inputs)
        choices.extend(extract_data(cropped_image, inputs))
        highlights.append(highlight(cropped_image, option_count, inputs, choices))

    # print(choices)
    return {"data": {"name": meta_data[0]["scale"], "choices": choices}, "highlights": highlights}
