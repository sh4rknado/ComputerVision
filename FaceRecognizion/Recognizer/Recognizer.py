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
#         Definition of Import
# ===========================================================================
from tqdm import tqdm
from Helper import Colors, PATH, Serializer
from Recognizer.Model import create_model
from Recognizer.align import AlignDlib
from scipy.spatial import distance
from imutils import paths
from Recognizer.MotorRecognizer import MotorRecognizer
from DPObserver.Subject import Subject

import numpy as np
import cv2
import glob
import dlib
import matplotlib.pyplot as plt
import time


class Recognizer(Subject):
    def __init__(self):
        self._color = Colors.Colors()
        self._path = PATH.PATH()
        self._serializer = Serializer.Serializer()
        self._data = self._serializer.loading_data()
        self._faces = self._serializer.loading_faces()
        self._train_paths = glob.glob("Data/IMAGE_DB/*")
        self._nb_classes = len(self._train_paths)
        self._label_index = []
        # print(type(self._faces))
        # print(self._data)
        # print(self._faces)

        self._nn4_small2 = create_model()
        self._nn4_small2.summary()

        self._color.printing("info", "[LOADING] Load the model size of openface")
        self._nn4_small2.load_weights(self._path.OPENFACE_NN4_SMALL2_V1_H5)

        self._color.printing("info", "[LOADING] Align the face Predicator 68 Face Landmarks")
        self._alignment = AlignDlib(self._path.SHAPE_PREDICATOR_68_FACE_LANDMARKS)
        self._color.printing("success", "[LOADING] Loading Model Completed\n")

        self._reco_thread = []
        self._running = 0
        self._total = 0

    # *=================================================================*
    # |                 RUN THE FACIAL RECOGNIZION                      |
    # *=================================================================*
    def run(self):
        data = self._trainning()
        self._color.printing("infos", "[Analysing] Analysing Match/Unmatch distance \n")
        self._analysing(data[0])

        self._thread_init(data[0])
        self._color.printing("success", "[SUCCESS] Quantifying faces To Detect Completed\n")

        self._launch_recognizer()
        self._waiting_end_thread()
        self._color.printing("success", "[SUCCESS] Face Detection Completed\n")

    # *=================================================================*
    # |                    Method Helpers                               |
    # | Use the 68 Landmarks Predicator with face Alignement of Face    |
    # | with the list of Keras Model :                                  |
    # |   OPENFACE_NN4_SMALL2_V1_H5                                     |
    # |   OPENFACE_NN4_SMALL2_V1_T7                                     |
    # *=================================================================*
    def _align_face(self, face):
        # print(img.shape)
        (h, w, c) = face.shape
        bb = dlib.rectangle(0, 0, w, h)
        # print(bb)
        return self._alignment.align(96, face, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    def _load_and_align_images(self, filepaths):
        aligned_images = []
        for filepath in filepaths:
            # print(filepath)
            img = cv2.imread(filepath)
            aligned = self._align_face(img)
            aligned = (aligned / 255.).astype(np.float32)
            aligned = np.expand_dims(aligned, axis=0)
            aligned_images.append(aligned)

        return np.array(aligned_images)

    def _calc_embs(self, filepaths, batch_size=64):
        pd = []
        for start in tqdm(range(0, len(filepaths), batch_size)):
            aligned_images = self._load_and_align_images(filepaths[start:start + batch_size])
            pd.append(self._nn4_small2.predict_on_batch(np.squeeze(aligned_images)))
        # embs = l2_normalize(np.concatenate(pd))
        embs = np.array(pd)

        return np.array(embs)

    def _trainning(self):

        for i in tqdm(range(len(self._train_paths))):
            self._label_index.append(np.asarray(self._data[self._data.label == i].index))

        train_embs = self._calc_embs(self._data.image)
        np.save(self._path.PICKLE_EMBS, train_embs)
        train_embs = np.concatenate(train_embs)

        return [train_embs]

    # *=========================================*
    # | Analysing the Match / Unmatch Distance  |
    # *=========================================*
    def _analysing(self, train_embs):
        match_distances = []
        unmatch_distances = []

        for i in range(self._nb_classes):
            ids = self._label_index[i]
            distances = []
            for j in range(len(ids) - 1):
                for k in range(j + 1, len(ids)):
                    distances.append(distance.euclidean(train_embs[ids[j]].reshape(-1), train_embs[ids[k]].reshape(-1)))
            match_distances.extend(distances)

        for i in range(self._nb_classes):
            ids = self._label_index[i]
            distances = []
            for j in range(10):
                idx = np.random.randint(train_embs.shape[0])
                while idx in self._label_index[i]:
                    idx = np.random.randint(train_embs.shape[0])
                distances.append(
                    distance.euclidean(train_embs[ids[np.random.randint(len(ids))]].reshape(-1),
                                       train_embs[idx].reshape(-1)))
            unmatch_distances.extend(distances)

        _, _, _ = plt.hist(match_distances, bins=100)
        _, _, _ = plt.hist(unmatch_distances, bins=100, fc=(1, 0, 0, 0.5))
        plt.title("match/unmatch distances")
        # plt.show()

    # *================================================================*
    # |        Get the Notify from DP Observer                         |
    # *================================================================*
    """
    @:update
    """
    def update(self, value, message):
        self._running -= 1

    # *=================================================================*
    # |        Initialize the list of threads                           |
    # *=================================================================*
    """
    @:parameter train_embs = Trainning Data
    """
    def _thread_init(self, train_embs):
        images = list(paths.list_images(self._path.IMAGE_TO_DETECT))
        # print("Total Image : {0}".format(len(images)))
        total = 0
        for img in images:
            self._reco_thread.append(MotorRecognizer(self._label_index, self._nn4_small2,
                                                     self._alignment, img, train_embs, self, self._data))
            self._color.printing("info", "[LOADING] Create Threading Recognizing {}/{}".format(total + 1, len(images)))
            total += 1

        self._color.printing("success", "[SUCCESS] Create Threading Completed\n")

    def _waiting_end_thread(self):
        while self._running > 0:
            self._color.printing("info", "[WAITING] Waiting the end of Threads...")
            time.sleep(0.5)
        self._color.printing("success", "[SUCCESS] Thread Finished !\n")

    # *=================================================================*
    # |        Launch Thread for FaceDetection                          |
    # *=================================================================*
    """
    @:parameter data = the DataFrame from Panda
    @:parameter max = The maximum of Threads
    """
    def _launch_recognizer(self, max=15):
        self._total = 0
        while self._total < len(self._reco_thread):
            if self._running <= max:
                self._reco_thread[self._total].start()
                self._running += 1
                self._total += 1
                self._color.printing("info", "[PROCESSING] Processing image {}/{}".format(self._total, len(self._reco_thread)))
            else:
                while self._running == 5:
                    time.sleep(0.1)
