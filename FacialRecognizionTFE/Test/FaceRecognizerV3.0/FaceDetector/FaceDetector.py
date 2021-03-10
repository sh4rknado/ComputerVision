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
from threading import Thread
from DPObserver.Observer import Observer
from Helper.Colors import Colors
from imageio import imsave

import cv2
import os


# ===========================================================================
#         Definition of Class FaceDetector
# ===========================================================================
class FaceDetector(Thread, Observer):

    def __init__(self, frame, img_path, subject):
        # ===============
        # Use the Builder
        # ===============
        Thread.__init__(self)
        Observer.__init__(self)
        self._color = Colors()

        # ====================
        # Register the Subject
        # ====================
        self.register(subject)

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
    #         Override the start Thread
    # ===========================================================================
    def run(self):
        self._detect_face(self.faces)

    # ===========================================================================
    #         Detect and return faces the face from Frame
    # ===========================================================================
    def _detect_face(self, frame):
        frame_copy = frame.copy()
        frameHeight = frame_copy.shape[0]
        frameWidth = frame_copy.shape[1]

        blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], False, False)
        self._net.setInput(blob)
        detections = self._net.forward()

        faces = []
        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faces.append([x1, y1, x2, y2])
                try:
                    imsave(self._img_path, frame_copy[y1:y2, x1:x2])
                except:
                    self._color.printing("error", "Error write detecting Faces : " + self._img_path)
                    os.system("rm -rf " + self._img_path)
                finally:
                    pass

        # print(faces)
        self.update_observer(faces, "Finished")
