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


# *===========================================================================*
# |        Definition of Import                                               |
# *===========================================================================*
from imutils import paths, resize
from Helper.PATH import PATH
from Helper.Colors import Colors
from Helper.Serializer import Serializer
from FaceDetector.FaceDetectorDNN import FaceDetectorDNN
from FaceDetector.FaceDetectorHoG import FaceDetectorHoG
from scipy.spatial import distance
from glob import glob
from Recognizer.align import AlignDlib
from imageio import imsave
from threading import Thread
from DPObserver.Observer import Observer

import cv2
import numpy as np
import dlib


class MotorRecognizer(Thread, Observer):
    def __init__(self, labelindex, nn4_small2, alignment, image_path, train_embs, subject, data):
        Thread.__init__(self)
        Observer.__init__(self)

        # *======================*
        # | Register the Subject |
        # *======================*
        self.register(subject)

        self._path = PATH()
        self._color = Colors()
        self._serializer = Serializer()
        self._train_paths = glob("Data/IMAGE_DB/*")
        self._label_index = labelindex
        self._nn4_small2 = nn4_small2
        self._alignment = alignment
        self._data = data
        self._image_path = image_path
        self._train_embs = train_embs

    def run(self):
        self._recognize(self._train_embs, self._image_path)

    def _align_face(self, face):
        # print(img.shape)
        (h, w, c) = face.shape
        bb = dlib.rectangle(0, 0, w, h)
        # print(bb)
        return self._alignment.align(96, face, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    def _align_faces(self, faces):
        aligned_images = []
        for face in faces:
            # print(face.shape)
            aligned = self._align_face(face)
            aligned = (aligned / 255.).astype(np.float32)
            aligned = np.expand_dims(aligned, axis=0)
            aligned_images.append(aligned)

        return aligned_images

    def _calc_emb_test(self, faces):
        pd = []
        aligned_faces = self._align_faces(faces)
        # if face detected
        if len(faces) == 1:
            pd.append(self._nn4_small2.predict_on_batch(aligned_faces))
        elif len(faces) > 1:
            pd.append(self._nn4_small2.predict_on_batch(np.squeeze(aligned_faces)))
        # embs = l2_normalize(np.concatenate(pd))
        embs = np.array(pd)
        return np.array(embs)

    def _motor_recognizer(self, fd, image_copy, train_embs, total, threshold=0.8):
        faces = fd.detect_face(image_copy)

        if len(faces) == 0:
            print("no face detected!")
        else:
            test_embs = self._calc_emb_test(faces)
            test_embs = np.concatenate(test_embs)

            people = []
            for i in range(test_embs.shape[0]):
                distances = []
                for j in range(len(self._train_paths)):
                    distances.append(np.min(
                        [distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in
                         self._label_index[j]]))
                    # for k in label2idx[j]:
                    # print(distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)))
                if np.min(distances) > threshold:
                    people.append("inconnu")
                else:
                    res = np.argsort(distances)[:1]
                    people.append(res)

            names = []
            title = ""
            for p in people:
                if p == "inconnu":
                    name = "inconnu"
                else:
                    name = self._data[(self._data['label'] == p[0])].name.iloc[0]
                names.append(name)
                title = title + name + " "

            result = fd.detect_face_name(image_copy, names)
            temp_name = result[1]
            image_copy = result[0]
            image_copy = resize(image_copy, width=720)

            self._saving(temp_name, total, image_copy)

            # print(faces)
            self.update_observer(image_copy, temp_name)

    def _saving(self, temp_name, total, image_copy):
        print(self._path.IMAGE_DB_RESULT + "/" + temp_name + "{0}".format(total) + ".jpg")
        imsave(self._path.IMAGE_DB_RESULT + "/" + temp_name + "{0}".format(total) + ".jpg", image_copy)

        # cv2.imshow("result", image_copy)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

    # *========================================*
    # |         Recognizer Image               |
    # *========================================*
    def _recognize(self, train_embs, imagepath):
        total = 0

        image = cv2.imread(imagepath)
        image_copy = image.copy()

        fd_dnn = FaceDetectorDNN(image_copy, imagepath)
        fd_HoG = FaceDetectorHoG(image_copy, imagepath)

        data = []

        try:
            self._motor_recognizer(fd_dnn, image_copy, train_embs, total)
        except:
            self._motor_recognizer(fd_HoG, image_copy, train_embs, total)
        finally:
            total += 1