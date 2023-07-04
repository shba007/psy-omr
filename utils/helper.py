import numpy as np
import cv2


def is_circle_inside(circle_center):
    # from markers 3,5,11,9
    boundary = [
        [70.0, 390.5],
        [2306.0, 390.5],
        [2306.0, 3294.0],
        [70.0, 3294.0],
    ]

    x, y = circle_center
    x_min, y_min = boundary[0]
    x_max, y_max = boundary[2]

    if x_min <= x <= x_max and y_min <= y <= y_max:
        return True
    else:
        return False


def choice_generator(option, index, total):
    factor = 4
    index = index - 1
    unit = 15
    x = 55
    y = 100

    while index < total:
        if index % 40 == 0 and index != 0:
            x += 110
            y = 100
        elif index % 5 == 0 and index != 0:
            y += 15

        y += unit

        choices = None
        if option == 2:
            choices = [
                {'value': 1, 'chord': [(x) * factor, (y) * factor]},
                {'value': 0, 'chord': [(x + unit) * factor, (y) * factor]}
            ]
        elif option == 5:
            choices = [
                {'value': 0, 'chord': [(x) * factor, (y) * factor]},
                {'value': 1, 'chord': [(x + 1 * unit) * factor, (y) * factor]},
                {'value': 2, 'chord': [(x + 2 * unit) * factor, (y) * factor]},
                {'value': 3, 'chord': [(x + 3 * unit) * factor, (y) * factor]},
                {'value': 4, 'chord': [(x + 4 * unit) * factor, (y) * factor]}
            ]

        yield {
            'index': index + 1,
            'choices': choices
        }

        index += 1


def calculate_bw_ratio(image):
    # Threshold the image to get binary image with white pixels
    _, binary = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)

    # Count the white pixels
    num_white_pixels = np.count_nonzero(binary == 255)

    # Calculate the ratio of white pixels to total pixels
    height, width = binary.shape
    num_pixels = width * height
    white_ratio = num_white_pixels / num_pixels

    return white_ratio
