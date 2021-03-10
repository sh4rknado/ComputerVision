import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths
from Model.PATH import PATH
from Model.Colors import Colors
import dlib


class Recognize:
    def __init__(self):
        self.path = PATH()
        self.color = Colors()

    def run(self):
        tab = self.Loading()  # return [detector, embedder, recognizer, le]
        tab2 = self.getListImages()

        imagePaths = tab2[0]
        total = tab2[1]
        i = 1
        for image in imagePaths:
            self.Recognize(image, tab[0], tab[1], tab[2], tab[3], i, total)
            # self.Recognize2(tab[0], image, tab[1])
            i += 1

        self.color.printing("success", " Recognizing Face Completed !")

    """
    @:return [images, total]
    """
    def getListImages(self):
        # grab the paths to the input images in our dataset
        self.color.printing("info", "[PROCESSING] Quantifying faces...")
        images = []
        total = 0

        for root, dirs, files in os.walk(self.path.IMAGE_TO_DETECT):
            for file in files:
                if file.endswith("png") | file.endswith("jpg"):
                    path = os.path.join(root, file)
                    images.append(path)
                    total += 1
        self.color.printing("info", "[PROCESSING] Total Faces : {}".format(total))
        self.color.printing("success", "[SUCCESS] Quantifying faces Completed\n")

        return [images, total]

    """
    @:return [detector, embedder, recognizer, le]
    """
    def Loading(self):
        # load our serialized face detector from disk
        self.color.printing("info", "[LOADING] Loading face detector...")
        detector = cv2.dnn.readNetFromCaffe(self.path.PROTO_PATH, self.path.MODEL_PATH)
        # detector = dlib.get_frontal_face_detector()

        # load our serialized face embedding model from disk
        self.color.printing("info", "[LOADING] Loading face recognizer...")

        embedder = cv2.dnn.readNetFromTorch(self.path.EMBEDDING_MODEL)
        # embedder = dlib.shape_predictor(self.path.PREDICATOR)

        # load the actual face recognition model along with the label encoder
        self.color.printing("info", "[LOADING] Data Serialised...")
        recognizer = pickle.loads(open(self.path.PICKLE_RECOGNIZER, "rb").read())
        le = pickle.loads(open(self.path.PICKLE_LE, "rb").read())
        self.color.printing("success", "[LOADING] Completed\n")

        return [detector, embedder, recognizer, le]

    def Recognize2(self, detector, image_path, predicator):
        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            landmarks = predicator(gray, face)

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

            cv2.imshow("Frame", frame)
            cv2.waitKey(0)


    """
    @:parameter image_path = Path of Image to Detect
    @:parameter detector = detector Facial
    @:parameter recognizer = recognizer Facial
    @:parameter le = Label of ID
    """
    def Recognize(self, image_path, detector, embedder, recognizer, le, i, total):
        self.color.printing("info", "[PROCESSING] processing image {}/{}".format(i, total))
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image dimensions
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > self.path.CONFIDENCE:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                 (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the associated
                # probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # show the output image
                cv2.imshow("Image", image)
                cv2.waitKey(0)

    """ 
    @:return True or False
    """
    def check_all_files(self):
        check = True
        self.color.printing("info", "[CHECKING] Verification of files and Directory ...")

        # check if file PROTOPATH Exits
        if not os.path.isfile(self.path.PROTO_PATH):
            self.color.printing("error", "File Not Found : " + self.path.PROTO_PATH)
            check = False

        # check if file MODEL_PATH Exits
        if not os.path.isfile(self.path.MODEL_PATH):
            self.color.printing("error", "File Not Found : " + self.path.MODEL_PATH)
            check = False

        # check if file EMBEDDEDING_MODEL Exists
        if not os.path.isfile(self.path.EMBEDDING_MODEL):
            self.color.printing("error", "File Not Found : " + self.path.EMBEDDING_MODEL)
            check = False

        # check if folder Imagedb Exits
        if not os.path.isdir(self.path.IMAGE_DB):
            self.color.printing("error", "Folder Not Found : " + self.path.IMAGE_DB)
            check = False

        # check if folder PICKLE_DIR Exits
        if not os.path.isdir(self.path.PICKLE_DIR):
            self.color.printing("error", "Folder Not Found : " + self.path.PICKLE_DIR)
            check = False

        return check
