import os


class PATH:
    def __init__(self):
        self._BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self._PROTO_DIR = os.path.join(self._BASE_DIR, "../Data/face_detection_model")
        self.PICKLE_DIR = os.path.join(self._BASE_DIR, "../Data/Pickle")
        self.PREDICATOR = os.path.sep.join([self._PROTO_DIR, "shape_predictor_68_face_landmarks.dat"])
        self.PROTO_PATH = os.path.sep.join([self._PROTO_DIR, "deploy.prototxt"])
        self.MODEL_PATH = os.path.sep.join([self._PROTO_DIR, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.EMBEDDING_MODEL = os.path.sep.join([self._PROTO_DIR, "openface_nn4.small2.v1.t7"])
        self.IMAGE_DB = os.path.join(self._BASE_DIR, "../Data/ImagesDB")
        self.PICKLE_EMBEDDED = os.path.sep.join([self.PICKLE_DIR, "embeddings.Pickle"])
        self.PICKLE_LE = os.path.sep.join([self.PICKLE_DIR, "le.Pickle"])
        self.PICKLE_RECOGNIZER = os.path.sep.join([self.PICKLE_DIR, "recognizer.Pickle"])
        self.IMAGE_TO_DETECT = os.path.sep.join([self._BASE_DIR, "../ImageToDetect"])
        self.CONFIDENCE = 0.9
