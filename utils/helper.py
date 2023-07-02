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
    

def choice_generator(total):
    unit = 15
    index = 0
    x = 45
    y = 10

    while index < total:
        if index % 40 == 0 and index != 0:
            x += 70
            y = 10
        elif index % 5 == 0 and index != 0:
            y += 15

        y += unit

        yield {
            'index': index + 1,
            'choices': [
                {'name': 'True', 'value': 1, 'chord': [x, y]},
                {'name': 'False', 'value': 0, 'chord': [x + unit, y]}
            ]
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
    
""" 
function drawCircle(canvas: Canvas, x: number, y: number, type: 'alignment' | 'response', value?: number) {
  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.arc(x, y, type === 'alignment' ? 25 : 10, 0, 2 * Math.PI);

  if (type === 'alignment') {
    ctx.setLineDash([10, 10]);
    ctx.lineCap = 'round';
    ctx.lineWidth = 7.5;
    ctx.strokeStyle = '#22c55e'
    ctx.stroke()
  } else {
    const colorMap = ["#e11d48", "#c026d3", "#9333ea", "#4f46e5", "#3b82f6"]
    ctx.fillStyle = colorMap[value ?? 0];
    ctx.fill();
  }
}
 """