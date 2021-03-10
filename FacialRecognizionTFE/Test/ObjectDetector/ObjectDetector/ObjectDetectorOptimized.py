import numpy as np
import cv2
import re
import os


class ObjectDetectorOptimized:
    def __init__(self, confidence, threshsold, data_loaded, showPercent, override_zm, pattern=None):
        self._confidence = confidence
        self._threshold = threshsold
        self._detect_pattern = pattern
        self._show_percent = showPercent
        self._yolo_override_ZM = override_zm
        self._data_loaded = data_loaded
        self.image_path = ""

    # ==================================*
    # |   Run the Detector Algorithm    |
    # *=================================*
    def run(self):
        # print("[INFO] Try to Detect Now...")
        result = self._detector(self._data_loaded[0], self._data_loaded[1], self._data_loaded[2])
        return result

    # *==========================*
    # |   Detector Algorithm     |
    # *==========================*
    def _detector(self, labels, colors, net):
        image = cv2.imread(self.image_path)
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
        result = " "

        # ensure at least one detection exists
        if len(idxs) > 0:
            for i in idxs.flatten():
                detect = labels[classIDs[i]]
                if re.match(self._detect_pattern, detect):

                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in colors[classIDs[i]]]

                    if labels[classIDs[i]] not in result:
                        result += "{0}".format(labels[classIDs[i]])

                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                    if self._show_percent:
                        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                    else:
                        text = "{0}".format(labels[classIDs[i]])

                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if not result.find(self._detect_pattern):
                os.remove(self.image_path)

        if self._yolo_override_ZM:
            # cv2.imshow("DEBUG", image)
            # cv2.waitKey(0)
            cv2.imwrite(self.image_path, image)

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

