import os


class PATH:
    def __init__(self):
        self._BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self._PICKLE_DIR = os.path.join(self._BASE_DIR, "../Data/Pickle")
        self._FACE_MODEL = os.path.join(self._BASE_DIR, "../Data/Model")
        self.IMAGE_DB = os.path.join(self._BASE_DIR, "../Data/IMAGE_DB")
        self.IMAGE_DB_RAW = os.path.join(self._BASE_DIR, "../IMAGE_DB_RAW")
        self.IMAGE_DB_RESULT = os.path.join(self._BASE_DIR, "../IMAGE_DB_RESULT")
        self.PICKLE_DATA = os.path.sep.join([self._PICKLE_DIR, "data.Pickle"])
        self.PICKLE_FACES = os.path.sep.join([self._PICKLE_DIR, "faces.Pickle"])
        self.PICKLE_EMBS = os.path.sep.join([self._PICKLE_DIR, "self._train_embs.npy"])
        self.OPENFACE_NN4_SMALL2_V1_H5 = os.path.sep.join([self._FACE_MODEL, "openface_nn4.small2.v1.h5"])
        self.OPENFACE_NN4_SMALL2_V1_T7 = os.path.sep.join([self._FACE_MODEL, "openface_nn4.small2.v1.t7"])
        self.SHAPE_PREDICATOR_68_FACE_LANDMARKS = os.path.sep.join([self._FACE_MODEL, "shape_predictor_68_face_landmarks.dat"])
        self.IMAGE_TO_DETECT = os.path.sep.join([self._BASE_DIR, "../IMAGE_TO_DETECT"])
