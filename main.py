import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from rotation_angel import align_image, find_rotation_degrees

app = FaceAnalysis(allowed_modules=['detection'])  # enable detection model only

app.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread("./semi_rotate.jpg")
faces = app.get(img)

if faces:
    face = faces[0]
    print(face)
    print(face.landmark)
    landmarks = face['kps']
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    print(f"Left Eye Coordinates: ({left_eye_x}, {left_eye_y})")
    print(f"Right Eye Coordinates: ({right_eye_x}, {right_eye_y})")
    slope = (left_eye_y - right_eye_y)/(left_eye_x - right_eye_x)
    print(slope)

    rotate_deg = find_rotation_degrees(slope)

    align_image('semi_rotate.jpg', rotate_deg)

else:
    print('No faces detected.')

# Method-2, load model directly
# detector = insightface.model_zoo.get_model('your_detection_model.onnx')
# detector.prepare(ctx_id=0, input_size=(640, 640))
