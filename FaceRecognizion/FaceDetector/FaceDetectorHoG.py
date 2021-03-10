# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordan BERTIEAUX"
__copyright__ = "Copyright 2019, Facial Recognition"
__credits__ = ["Jordan BERTIEAUX"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Jordan BERTIEAUX"
__email__ = "jordan.bertieaux@std.heh.be"
__status__ = "Production"


# ===========================================================================
#           Definition of Import
# ===========================================================================
from Helper.Colors import Colors

import cv2
import dlib


# ===========================================================================
#         Definition of Class FaceDetector
# ===========================================================================
class FaceDetectorHoG:

    def __init__(self, frame, img_path):
        self._color = Colors()
        self.faces = frame
        self._img_path = img_path

    # ===========================================================================
    #         Detect and return faces the face from Frame
    # ===========================================================================
    def detect_face(self, frame):
        frame_copy = frame.copy()

        hogFaceDetector = dlib.get_frontal_face_detector()
        faceRects = hogFaceDetector(frame_copy, 0)

        faces = []

        for faceRect in faceRects:
            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()
            face = frame_copy[y1:y2, x1:x2]
            # print(face)
            faces.append(face)
        # print(faces)
        return faces

    def detect_face_name(self, frame, names):
        frame_copy = frame.copy()

        hogFaceDetector = dlib.get_frontal_face_detector()
        faceRects = hogFaceDetector(frame_copy, 0)

        temp_name = ""
        for i, faceRect in enumerate(faceRects):
            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()
            frame_copy = self._create_rect(frame_copy, names[i], x1, x2, y1, y2)
            temp_name = names[i]

        return [frame_copy, temp_name]

    # ===========================================================================
    #         Create the Rect into the frame
    # ===========================================================================
    def _create_rect(self, frame, names, x1, x2, y1, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, names, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        return frame
