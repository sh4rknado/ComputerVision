from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
from Model.PATH import PATH
from Model.Colors import Colors


class extractEmbeddings:
    def __init__(self):
        self.path = PATH()
        self.color = Colors()

    """
    Create The Model
    """
    def run(self):
        if self.check_all_files():
            self.color.printing("success", "[SUCCESS] Files and Directory is checked\n")
        else:
            self.color.printing("error", "[ERROR] Files is missing or not existing !")
            exit(0)

        tab = self.loading()
        self.extract_facial(tab[0], tab[1], tab[2])

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

    """
    @:return [detector, embedder, imagePaths]
    """
    def loading(self):
        # load our serialized face detector from disk
        self.color.printing("info", "[LOADING] Loading face detector...")
        detector = cv2.dnn.readNetFromCaffe(self.path.PROTO_PATH, self.path.MODEL_PATH)

        # load our serialized face embedding model from disk
        self.color.printing("info", "[LOADING] Loading face recognizer...")
        embedder = cv2.dnn.readNetFromTorch(self.path.EMBEDDING_MODEL)

        # grab the paths to the input images in our dataset
        self.color.printing("info", "[LOADING] Quantifying faces...")
        imagePaths = list(paths.list_images(self.path.IMAGE_DB))

        self.color.printing("success", "[SUCCESS] Face Recognizer and Detector is Loaded\n")

        return [detector, embedder, imagePaths]

    """
     @:parameter knownEmbeddings = tab of knownEmbeddings
     @:parameter embedder = detector facial 
     @:total     imagePaths = tab of pathImage
     """
    def extract_facial(self, detector, embedder, imagePaths):
        # initialize our lists of extracted facial embeddings and
        # corresponding people names
        knownEmbeddings = []
        knownNames = []

        # initialize the total number of faces processed
        total = 0

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):

            # extract the person name from the image path
            self.color.printing("info", "[PROCESSING] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load the image, resize it to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            detector.setInput(imageBlob)
            detections = detector.forward()

            # ensure at least one face was found
            if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out
                # weak detections)
                if confidence > self.path.CONFIDENCE:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI and grab the ROI dimensions
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    # add the name of the person + corresponding face
                    # embedding to their respective lists
                    knownNames.append(name)
                    knownEmbeddings.append(vec.flatten())
                    total += 1

        self.color.printing("success", "[SUCCESS] Extraction Completed\n")
        self.saving(knownEmbeddings, knownNames, total)

    """
    Serialise the data of Model (Label and Embeddings)
    @:parameter knownEmbeddings = tab of knownEmbeddings
    @:parameter knownNames =  tab of knownNames
    @:total     total = total of element 
    """
    def saving(self, knownEmbeddings, knownNames, total):
        # dump the facial embeddings + names to disk
        self.color.printing("info", "[SAVING] serializing {} encodings...".format(total))
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(self.path.PICKLE_EMBEDDED, "wb")
        f.write(pickle.dumps(data))
        f.close()
        self.color.printing("success", "[SUCCESS] serializing Completed\n")
