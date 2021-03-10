# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================================================
#           Infos developer
# ===========================================================================
__author__ = "Jordan BERTIEAUX"
__copyright__ = "Copyright 2020, Facial Recognition"
__credits__ = ["Jordan BERTIEAUX"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Jordan BERTIEAUX"
__email__ = "jordan.bertieaux@std.heh.be"
__status__ = "Production"


# ===========================================================================
#           Definition of Import
# ===========================================================================
from DP.Observable import Observable
from Helper.Colors import Colors
import numpy as np
import cv2
import re
import threading
import time


class ObjectDetector(threading.Thread):
    def __init__(self, confidence, threshsold, data_loaded, showPercent, override_zm, lock, main_observer, img, pattern=None):
        super(ObjectDetector, self).__init__()
        self._confidence = confidence
        self._threshold = threshsold
        self._detect_pattern = pattern
        self._show_percent = showPercent
        self._yolo_override_ZM = override_zm
        self.LABELS, self.COLORS, self.NET = data_loaded
        self._lock = lock
        # determine only the *output* layer names that we need from YOLO
        self._ln = self.NET.getLayerNames()
        self._ln = [self._ln[i[0] - 1] for i in self.NET.getUnconnectedOutLayers()]
        # Observer Pattern
        self._objectDetector = Observable()
        self._objectDetector.register(main_observer)
        self.IMG = img

    # *==========================*
    # |   Running   Threading    |
    # *==========================*
    def run(self):
        self.wait_lock()
        self._detector()

    # ===========================================================================
    #               Mise a jour et notification au sujet
    # ===========================================================================
    def update(self, result, image_result):
        self._objectDetector.update_observer(result, image_result)

    def wait_lock(self):
        if self._lock.locked():
            while self._lock.locked() is True:
                time.sleep(0.1)
                # Colors.print_infos("[INFOS] Waiting unlocked thread...")

    # *==========================*
    # |   Detector Algorithm     |
    # *==========================*
    def _detector(self):
        objImg = cv2.imread(self.IMG)
        (H, W) = objImg.shape[:2]

        # Construct the Matrice 4D from Image
        self.NET.setInput(cv2.dnn.blobFromImage(objImg, 1/255.0, (416, 416), swapRB=True, crop=False))
        self._lock.acquire()
        layerOutputs = self.NET.forward(self._ln)
        self._lock.release()

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self._confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self._confidence, self._threshold)
        result = ""

        # ensure at least one detection exists
        if len(idxs) > 0:
            for i in idxs.flatten():
                detect = self.LABELS[classIDs[i]]
                if re.match(self._detect_pattern, detect):

                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in self.COLORS[classIDs[i]]]

                    if self.LABELS[classIDs[i]] not in result:
                        result += "{0}".format(self.LABELS[classIDs[i]])

                    cv2.rectangle(objImg, (x, y), (x + w, y + h), color, 2)

                    if self._show_percent:
                        text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                    else:
                        text = "{0}".format(self.LABELS[classIDs[i]])

                    cv2.putText(objImg, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    self.update(result, self.IMG)

        if self._yolo_override_ZM:
            cv2.imwrite(self.IMG, objImg)

        # cleanning the RAM
        del objImg
        del H
        del W
        del layerOutputs
        del boxes
        del confidences
        del classIDs
        del idxs
