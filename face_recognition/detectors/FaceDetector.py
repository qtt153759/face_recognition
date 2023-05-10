import math
import time
from PIL import Image
import numpy as np
from face_recognition.commons import distance
from face_recognition.detectors import MtcnnWrapper



def detect_face(face_detector, detector_backend, img, align=True):

    obj = detect_faces(face_detector, detector_backend, img, align)

    if len(obj) > 0:
        face, region, confidence = obj[0]  # discard multiple faces
    else:  # len(obj) == 0
        face = None
        region = [0, 0, img.shape[1], img.shape[0]]

    return face, region, confidence


def detect_faces( img, align=True):
    tic = time.time()
    face_detector=MtcnnWrapper.build_model()
    obj = MtcnnWrapper.detect_face(face_detector, img, align)
        # obj stores list of (detected_face, region, confidence)
    print("MTCNN detect_face took ", time.time() - tic, " seconds")
    return obj
   

def alignment_procedure(img, left_eye, right_eye):

    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    # -----------------------

    return img  # return img anyway
