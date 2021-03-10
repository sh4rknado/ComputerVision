import numpy as np
import cv2
from os import path
import re
from configparser import ConfigParser


class ObjectDetector:
    def __init__(self, image=None, pattern=None):
        config = ConfigParser()
        config.read('Data/Config/detector.ini')
        config = config['object']

        self._confidence = float(config['confidence'])
        self._threshold = float(config['threshold'])

        self._yolo_labels_path = config['yolo_labels_path']
        self._yolo_config_path = config['yolo_config_path']
        self._yolo_weights_path = config['yolo_weights_path']

        self._detect_pattern = config['detect_pattern']
        self._show_percent = self._convert_boolean(config['yolo_show_percent'])
        self._yolo_override_ZM = self._convert_boolean(config['yolo_override_ZM'])

        if pattern is None:
            self._pattern = config['detect_pattern']
        else:
            self._pattern = pattern

        self._image_path = image
        del config

    # =============================*
    # | Convert String to Boolean  |
    # *============================*
    def _convert_boolean(self, string):
        if re.match('(y|Y|Yes|yes|True|true)', string):
            return True
        else:
            return False

    # =====================================*
    # | LOADING SECTION Init the Detector  |
    # *====================================*
    def _loading(self):

        if path.isfile(self._yolo_labels_path):
            LABELS = open(self._yolo_labels_path).read().strip().split("\n")

            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
        else:
            print("Error : LabelPath no such file or directory : {0}".format(self._yolo_labels_path))
            return None

        if path.isfile(self._yolo_weights_path) and path.isfile(self._yolo_config_path):
            NET = cv2.dnn.readNetFromDarknet(self._yolo_config_path, self._yolo_weights_path)
        else:
            print("Error : weightsPath no such file or directory : {0}".format(self._yolo_weights_path))
            print("Error : configPath or no such file or directory : {0}".format(self._yolo_config_path))
            return None

        del self._yolo_config_path
        del self._yolo_weights_path
        del self._yolo_labels_path

        return [LABELS, COLORS, NET]

    # ==================================*
    # |   Run the Detector Algorithm    |
    # *=================================*
    def run(self):
        # print("[INFO] loading YOLO from disk...")
        data_loaded = self._loading()

        # print("[INFO] Try to Detect Now...")
        result = self._detector(data_loaded[0], data_loaded[1], data_loaded[2])

        # print("[INFO] Clean the Object into the ram...")
        del data_loaded
        self._cleanning_ram()

        return result

    # *==========================*
    # |     Cleanning RAM        |
    # *==========================*
    def _cleanning_ram(self):
        del self._confidence
        del self._threshold
        del self._image_path
        del self._show_percent
        del self._yolo_override_ZM
        del self._pattern

    # *==========================*
    # |   Detector Algorithm     |
    # *==========================*
    def _detector(self, labels, colors, net):
        image = cv2.imread(self._image_path)
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # Construct the Matrice 4D from Image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)
        layerOutputs = net.forward(ln)

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
        result = "Detected : "

        # ensure at least one detection exists
        if len(idxs) > 0:
            for i in idxs.flatten():

                if re.match(self._pattern, labels[classIDs[i]]):

                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in colors[classIDs[i]]]

                    if labels[classIDs[i]] not in result:
                        result += "{0}-".format(labels[classIDs[i]])

                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                    if self._show_percent:
                        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                    else:
                        text = "{0}".format(labels[classIDs[i]])

                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self._yolo_override_ZM:
            # cv2.imshow("DEBUG", image)
            # cv2.waitKey(0)
            cv2.imwrite(self._image_path, image)

        # cleanning the RAM
        del image
        del H
        del W
        del ln
        del blob
        del net
        del layerOutputs
        del boxes
        del confidences
        del classIDs
        del idxs
        cv2.destroyAllWindows()
        return result

