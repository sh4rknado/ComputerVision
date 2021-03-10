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


# ===========================================================================
#         Definition of Class FaceDetector
# ===========================================================================
class FaceDetectorDNN:

    def __init__(self, frame, img_path):
        # ===============
        # Use the Builder
        # ===============
        self._color = Colors()

        # ================================================================
        # OpenCV DNN supports 2 networks.
        # 1. FP16 version of the original caffe implementation ( 5.4 MB )
        # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
        # ================================================================
        self.DNN = "TF"

        # =======================================
        # Select the Network CAFFE or TensorFlow
        # ========================================
        if self.DNN == "CAFFE":
            self._modelFile = "Data/Model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            self._configFile = "Data/Model/deploy.prototxt"
            self._net = cv2.dnn.readNetFromCaffe(self._configFile, self._modelFile)
        else:
            self._modelFile = "Data/Model/opencv_face_detector_uint8.pb"
            self._configFile = "Data/Model/opencv_face_detector.pbtxt"
            self._net = cv2.dnn.readNetFromTensorflow(self._modelFile, self._configFile)

        # Select the confidence (0 to 1)
        self.conf_threshold = 0.8
        self.faces = frame
        self._img_path = img_path

    # ===========================================================================
    #         Detect and return faces the face from Frame
    # ===========================================================================
    def detect_face(self, frame):
        frame_copy = frame.copy()
        frameHeight = frame_copy.shape[0]
        frameWidth = frame_copy.shape[1]

        blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], False, False)
        self._net.setInput(blob)
        detections = self._net.forward()

        faces = []
        face = []
        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                face = frame_copy[y1:y2, x1:x2]
                faces.append(face)
        # print(faces)
        return faces

        # ===========================================================================
        #         Detect and return faces the face from Frame
        # ===========================================================================

    def detect_face_name(self, frame, names):
        frame_copy = frame.copy()
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], False, False)
        self._net.setInput(blob)
        detections = self._net.forward()

        faces = []
        temp_name = ""
        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faces = [x1, y1, x2, y2]
                temp_name = names[i]
                self.create_rect(frame_copy, names[i], x1, x2, y1, y2, frameHeight)
        # print(faces)
        return [frame_copy, temp_name]

    # ===========================================================================
    #         Create the Rect into the frame
    # ===========================================================================
    def create_rect(self, frame, names, x1, x2, y1, y2, frame_height):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
        cv2.putText(frame, names, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        return frame

