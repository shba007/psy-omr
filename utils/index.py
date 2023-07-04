import json
import base64
from fastapi import HTTPException

import numpy as np
import cv2
from cv2 import aruco, dnn_superres
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from utils.helper import calculate_bw_ratio, choice_generator, is_circle_inside


def detect_markers(image_buffer, findNecessary=True):
    image_array = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    image = cv2.filter2D(image, -1, kernel)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()

    corners, ids, rejected = aruco.detectMarkers(image=image, dictionary=aruco_dict, parameters=parameters)

    if ids is None:
        raise HTTPException(status_code=404, detail="Unable to Detect any marker")

    sufficient = sum(num in ids for num in [1, 2, 9, 11]) >= 4

    if not (sufficient) and findNecessary:
        raise HTTPException(status_code=404, detail="Unable to Detect Corner markers")

    markers = [{"id": id[0].tolist(), "positions": [float(np.mean(corner[0, :, 0])), float(np.mean(corner[0, :, 1]))]} for id, corner in zip(ids, corners)]
    markers.sort(key=lambda x: x["id"])

    return markers


def align_crop(image_buffer, src_markers):
    image_array = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # cv2.imshow("align_crop", cv2.resize(image, (0,0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)

    factor = 4
    width, height = 595 * factor, 842 * factor

    corners = []
    for target_key in [1, 2, 11, 9]:
        src_marker = next((src_marker for src_marker in src_markers if src_marker['id'] == target_key), None)
        if src_marker is not None:
            corners.append(src_marker['positions'])

    src_points = np.array(corners, dtype=np.float32)
    dest_points = np.array([[70, 70], [width - 74, 70], [width - 74, height - 74], [70, height - 74]], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    cropped_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    success, encoded_image = cv2.imencode('.jpg', cropped_image)
    src_markers = detect_markers(encoded_image.tobytes())
    src_markers = [element for element in src_markers if element["id"] in [3, 5, 7, 9, 11]]

    dest_markers = [
        {'id': 1, 'positions': [70.0, 70.0]},
        {'id': 2, 'positions': [2306.0, 70.0]},
        {'id': 3, 'positions': [70.0, 390.5]},
        {'id': 4, 'positions': [1188.0, 390.5]},
        {'id': 5, 'positions': [2306.0, 390.5]},
        {'id': 6, 'positions': [70.25, 1842.25]},
        {'id': 7, 'positions': [1188.25, 1842.25]},
        {'id': 8, 'positions': [2306.25, 1842.25]},
        {'id': 9, 'positions': [70.0, 3294.0]},
        {'id': 10, 'positions': [1188.0, 3294.0]},
        {'id': 11, 'positions': [2306.0, 3294.0]}
    ]

    src_points = []
    dest_points = []
    # matches = []

    for index, src_marker in enumerate(src_markers):
        dest_marker = next((dest_marker for dest_marker in dest_markers if dest_marker['id'] == src_marker['id']), None)
        if dest_marker is not None:
            src_points.append(src_marker['positions'])
            dest_points.append(dest_marker['positions'])
            # matches.append((index, index))

    src_points = np.array(src_points)
    dest_points = np.array(dest_points)

    homography, _ = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(cropped_image, homography, (width, height))

    image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    image = cv2.convertScaleAbs(image, alpha=1.1, beta=0)
    _, buffer = cv2.imencode('.jpg', warped_image)

    return buffer.tobytes()


sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel("./model/ESPCN_x4.pb")
sr.setModel("espcn", 4)


def detect_qr(image_buffer):
    image_array = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    x = image.shape[1] - 105 - 380
    y = 55
    image = image[y:y+380, x:x+380]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale the image
    image = sr.upsample(image)
    image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    # cv2.imshow("detect_qr", image)
    # cv2.waitKey(0)

    detector = cv2.QRCodeDetector()
    retval, info, points, _ = detector.detectAndDecodeMulti(image)

    if retval is False:
        raise HTTPException(status_code=404, detail="Unable to detected QR Code")

    try:
        data = json.loads(info[0])
        return {"scale": data["scale"], "option": data["option"], "choice": {"start": data["start"], "count": data["count"], "total": data["total"]}}
    except:
        raise HTTPException(status_code=400, detail="Invalid QR Code detected")


def align_inputs(image_buffer, options_count, choice_start, choice_count):
    image_array = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # cv2.imshow("align_inputs", cv2.resize(image, (0,0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)

    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50
    )

    dest_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        for (x, y, r) in circles:
            if is_circle_inside((x, y)):
                dest_circles.append((x, y))

    choices = list(choice_generator(options_count, choice_start, choice_count))

    src_circles = np.array([choice["chord"] for data in choices for choice in data["choices"]])
    dest_circles = np.array(dest_circles)

    try:
        distances = np.linalg.norm(src_circles[:, np.newaxis] - dest_circles, axis=-1)
    except:
        raise HTTPException(status_code=500, detail="Unable to calculate")

    row_indices, col_indices = linear_sum_assignment(distances)

    matched_pairs = [(i, j) for i, j in zip(row_indices, col_indices)]

    for i in range(len(choices)):
        for j in range(len(choices[i]["choices"])):
            choices[i]["choices"][j]["chord"] = None

    for pair in matched_pairs:
        index = pair[0]
        choices[index//options_count]['choices'][index % options_count]['chord'] = dest_circles[pair[1]].tolist()

    return choices


def extract_data(image_buffer, inputs):
    image_array = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow("align_inputs", cv2.resize(image, (0,0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)

    factor = 4
    threshold = 0.12

    results = []
    for input_data in inputs:
        index = input_data['index']
        choices = input_data['choices']

        bw_ratios = []
        for choice in choices:
            chord = choice['chord']
            if chord is None:
                continue

            x, y = chord
            w, h = 12, 12
            left = int(x - (w / 2) * factor)
            top = int(y - (h / 2) * factor)
            width = int(w * factor)
            height = int(h * factor)

            crop_image = image[top:top+height, left:left+width]

            # cv2.imwrite(f'./temp/inputs/{index}-{choice["name"]}.jpg', crop_image)

            bw_ratio = calculate_bw_ratio(crop_image)
            bw_ratios.append(bw_ratio)

        choice_index = bw_ratios.index(min(bw_ratios)) if bw_ratios else None
        second_choice_index = bw_ratios.index(min(bw_ratios[:choice_index] + bw_ratios[choice_index+1:])) if len(bw_ratios) > 1 else None

        delta_bw_ratio = bw_ratios[choice_index] - bw_ratios[second_choice_index] if choice_index is not None and second_choice_index is not None else None

        if delta_bw_ratio is not None and abs(delta_bw_ratio) >= threshold:
            choice_index = choice_index if delta_bw_ratio < 0 else second_choice_index
        else:
            choice_index = None

        result = {
            'index': index,
            'value': choices[choice_index]['value'] if choice_index is not None else choice_index,
            # 'deltaBWRatio': delta_bw_ratio
        }
        results.append(result)

    return results


def draw_circle(canvas, x, y, circle_type, value=None):
    draw = ImageDraw.Draw(canvas)

    if circle_type == 'alignment':
        draw.ellipse((x - 27.5, y - 27.5, x + 27.5, y + 27.5), outline=(34, 197, 94), width=7)
    else:
        color_map = [(225, 29, 72), (192, 38, 211), (147, 51, 234), (79, 70, 229), (96, 165, 250)]
        color = color_map[value] if value is not None else color_map[0]
        draw.ellipse((x - 12.5, y - 12.5, x + 12.5, y + 12.5), fill=color, outline=(0, 0, 0), width=3)


def highlight(image_buffer, option_count, inputs, responses):
    image_array = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    is_alignment = True
    is_response = True if responses != None else False

    inputs = [[choice["chord"] for choice in input['choices']] for input in inputs]
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for q_index, input in enumerate(inputs):
        for d_index, dot in enumerate(input):
            if dot is None:
                continue

            x, y = dot

            if is_alignment:
                draw_circle(canvas, x, y, 'alignment')

            if is_response:
                choice = responses[q_index]['value'] if responses is not None else None
                if choice is not None:
                    if option_count == 2 and choice == 1 - d_index:
                        draw_circle(canvas, x, y, 'response', choice * 4)
                    elif option_count == 5 and choice == d_index:
                        draw_circle(canvas, x, y, 'response', choice)

    canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    height, width = canvas.shape[:2]
    new_width = int((720 / height) * width)
    canvas = cv2.resize(canvas, (new_width, 720))

    # cv2.imshow("highlighted", canvas)
    # cv2.waitKey(0)

    _, buffer = cv2.imencode('.jpg', canvas)
    image_base64 = base64.b64encode(buffer)
    image_str = image_base64.decode('utf-8')

    return f'data:image/jpeg;base64,{image_str}'
