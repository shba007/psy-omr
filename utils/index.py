import json

import numpy as np
import cv2
from cv2 import aruco
from scipy.optimize import linear_sum_assignment

from utils.helper import calculate_bw_ratio, choice_generator, is_circle_inside

def detect_markers(image_buffer, findNecessary=True):
    # Convert image bytes to an OpenCV image
    image_array = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    # Create a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])

    # Apply the sharpening kernel to the image
    image = cv2.filter2D(image, -1, kernel)

    # cv2.imshow("detect_markers", cv2.resize(image, (0,0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)

    # Define the dictionary for ArUco markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()

    # Detect ArUco markers
    corners, ids, rejected = aruco.detectMarkers(image=image, dictionary=aruco_dict, parameters=parameters)

    # Check if any markers are detected
    if ids is None:
        raise ValueError("No ArUco markers found")

    sufficient = sum(num in ids for num in [1, 2, 9, 11]) >= 4

    if not (sufficient) and findNecessary:
        raise ValueError(f"ArUco Corner markers not found, only found {ids}")

    markers = [{"id": id[0].tolist(), "positions": [float(np.mean(corner[0, :, 0])), float(np.mean(corner[0, :, 1]))]} for id, corner in zip(ids, corners)]
    # markers = [{"id": id[0].tolist(), "corners": corner[0].tolist()} for id, corner in zip(ids, corners)]
    markers.sort(key=lambda x: x["id"])

    return markers

def align_crop(image_buffer, src_markers):
    # Convert image bytes to an OpenCV image
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
    dest_points = np.array([[70, 70], [width -74, 70], [width - 74, height - 74] , [70, height - 74]], dtype=np.float32)
    # raise ValueError(src_points, dest_points)

    transform_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    cropped_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    # TODO: Homographs
    success, encoded_image = cv2.imencode('.jpg', cropped_image)
    src_markers = detect_markers(encoded_image.tobytes())
    src_markers = [element for element in src_markers if element["id"] in [3,5,7,9,11]]
    # raise ValueError(src_markers)
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
    
    # raise ValueError(src_points, dest_points)
    # Estimate the homography matrix using RANSAC algorithm
    homography, _ = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 5.0)
    # Warp the source image to align with the destination image using the homography matrix
    warped_image = cv2.warpPerspective(cropped_image, homography, (width, height))

    image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    warped_image = cv2.convertScaleAbs(warped_image, alpha=1.1, beta=0)
    """  # Create a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])
    # Apply the sharpening kernel to the warped_image
    warped_image = cv2.filter2D(warped_image, -1, kernel) """
    
    _, buffer = cv2.imencode('.jpg', warped_image)

    return buffer.tobytes()

def detect_qr(image_buffer):
    # Convert image bytes to an OpenCV image
    image_array = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Create a QR code detector
    detector = cv2.QRCodeDetector()

    # cv2.imshow("detect_qr", cv2.resize(image, (0,0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)
    
    # Detect and decode QR codes
    retval, info, points, _ = detector.detectAndDecodeMulti(image)

    if retval is False:
        raise ValueError("No QR Code found")
    
    data = json.loads(info[0])

    return { "scale": data["scale"], "page": { "current": data["curr"], "total": data["total"] } }

def align_inputs(image_buffer, choice_count):
    # Convert image bytes to an OpenCV image
    image_array = np.frombuffer(image_buffer, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Apply Gaussian blur to reduce noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # cv2.imshow("align_inputs", cv2.resize(image, (0,0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)

    # Apply HoughCircles to detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50
    )

    dest_circles = []
    # If circles are detected
    if circles is not None:
        # Convert the circles to integer coordinates
        circles = np.round(circles[0, :]).astype(int)

        # Draw circles and centers on the image
        for (x, y, r) in circles:
            if is_circle_inside((x,y)):
                dest_circles.append((x,y))

    # TODO: move into choice_generator
    factor = 4
    startX = 10
    startY = 90

    choices = [
        {
            'index': item['index'],
            'choices': [
                {
                    'name': choice['name'],
                    'value': choice['value'],
                    'chord': [((startX + (choice['chord'][0] if choice['chord'] is not None else 0)) * factor),
                              ((startY + (choice['chord'][1] if choice['chord'] is not None else 0)) * factor)]
                    if choice['chord'] is not None else None
                }
                for choice in item['choices']
            ]
        }
        for item in list(choice_generator(choice_count))
    ]
    # Find the circle in image buffer
    src_circles = np.array([choice["chord"] for data in choices for choice in data["choices"]])
    dest_circles = np.array(dest_circles)
    
    # Compute the pairwise distances between circles in the two sets
    try:
        distances = np.linalg.norm(src_circles[:, np.newaxis] - dest_circles, axis=-1)
    except:
        raise ValueError("unable to calculate distances", src_circles.shape, dest_circles.shape)

    # Solve the assignment problem using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(distances)

    # Retrieve the optimal matching pairs
    matched_pairs = [(i, j) for i, j in zip(row_indices, col_indices)]

    for i in range(len(choices)):
        for j in range(len(choices[i]["choices"])):
            choices[i]["choices"][j]["chord"] = None
    
    for pair in matched_pairs:
        index = pair[0]
        choices[index//2]['choices'][index%2]['chord'] = dest_circles[pair[1]].tolist()
    
    return choices

def extract_data(image_buffer, inputs):
    # Convert image bytes to an OpenCV image
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