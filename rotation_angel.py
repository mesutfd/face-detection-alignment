import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_rotation_degrees(slope) -> float:
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)

    return angle_deg


# Load your image
def align_image(path: str, degrees: float) -> None:
    image = cv2.imread(path)

    # Define the angle of rotation in degrees (positive for right, negative for left)
    angle = degrees  # Change this angle as needed

    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    new_width = int(width * abs(np.cos(np.radians(angle))) + height * abs(np.sin(np.radians(angle))))
    new_height = int(width * abs(np.sin(np.radians(angle))) + height * abs(np.cos(np.radians(angle))))

    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    # Display the rotated image using matplotlib
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    cv2.imwrite('rotated_image.jpg', rotated_image)


