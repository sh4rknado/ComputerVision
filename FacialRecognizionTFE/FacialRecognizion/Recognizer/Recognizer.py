import glob
import os
import re

import cv2
import dlib
import imutils
import matplotlib.pyplot as plt
import numpy as np
from imageio import imsave
from imutils import paths
from scipy.spatial import distance
from tqdm import tqdm

from FaceDetector import tiny_face_model
from Helper.Serializer import Serializer
from Helper.Colors import Colors
from Recognizer.Model import create_model
from Recognizer.align import AlignDlib


class Recognizer:
    def __init__(self, pickle_data, pickle_embs, tiny_model, openface_model, predicator_landmarks):
        self.data = Serializer.loading_data(pickle_data)
        self._pickles_embs = pickle_embs
        self._train_paths = glob.glob("Data/IMAGE_DB/*")
        self._nb_classes = len(self._train_paths)
        self._label_index = []

        self._tinyFace_model = tiny_face_model.Model(tiny_model)
        self._nn4_small2 = create_model()

        Colors.print_infos("[LOADING] Load the model size of openface")
        Colors.print_infos("[LOADING] Align the face Predicator 68 Face Landmarks")
        # self._nn4_small2.summary()

        self._nn4_small2.load_weights(openface_model)
        self._alignment = AlignDlib(predicator_landmarks)

        Colors.print_sucess("[LOADING] Loading Model Completed\n")

    # *=================================================================*
    # |                 RUN THE FACIAL RECOGNIZION                      |
    # *=================================================================*
    def run(self, faces, fd_tiny, frame, refined_bboxes):
        data = self._trainning()
        self._analysing(data[0])
        data = self._recognize(data[0], faces, fd_tiny, frame, refined_bboxes)

        # return [image_copy, temp_name]
        if data is not None:
            return [data[0], data[1]]
        else:
            return None

    # *=================================================================*
    # |                    Method Helpers                               |
    # | Use the 68 Landmarks Predicator with face Alignement of Face    |
    # | with the list of Keras Model :                                  |
    # |   OPENFACE_NN4_SMALL2_V1_H5                                     |
    # |   OPENFACE_NN4_SMALL2_V1_T7                                     |
    # *=================================================================*
    def _l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

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
            if img is not None:
                aligned = self._align_face(img)
                aligned = (aligned / 255.).astype(np.float32)
                aligned = np.expand_dims(aligned, axis=0)
                aligned_images.append(aligned)
            else:
                Colors.print_error("[ERROR] File Not Found in ImageDatabases: " + filepath)

        return np.array(aligned_images)

    def _calc_embs(self, filepaths, batch_size=64):
        pd = []
        for start in tqdm(range(0, len(filepaths), batch_size)):
            aligned_images = self._load_and_align_images(filepaths[start:start + batch_size])
            pd.append(self._nn4_small2.predict_on_batch(np.squeeze(aligned_images)))
        # embs = l2_normalize(np.concatenate(pd))
        embs = np.array(pd)

        return np.array(embs)

    def _align_faces(self, faces):
        aligned_images = []
        for face in faces:
            # print(face.shape)
            try:
                aligned = self._align_face(face)
                aligned = (aligned / 255.).astype(np.float32)
                aligned = np.expand_dims(aligned, axis=0)
                aligned_images.append(aligned)
            except:
                continue

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

    def _trainning(self):
        for i in tqdm(range(len(self._train_paths))):
            self._label_index.append(np.asarray(self.data[self.data.label == i].index))

        train_embs = self._calc_embs(self.data.image)
        np.save(self._pickles_embs, train_embs)
        train_embs = np.concatenate(train_embs)

        return [train_embs]

    # =========================================
    # Analysing the Match / Unmatch Distance
    # =========================================
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
        plt.imsave('Result_Match.jpg')
        # plt.show()

    # =========================================
    # Analysing the Match / Unmatch Distance
    # =========================================
    def _recognize(self, train_embs, faces, fd_tiny, frame, refined_bboxes):
        threshold = 0.7

        try:
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
                dist = np.min(distances)
                if dist > threshold:
                    people.append("inconnu")
                else:
                    res = np.argsort(distances)[:1]
                    people.append(res)

            names = []
            title = ""
            for p in people:
                if re.match('inconnu', str(p)):
                    name = "inconnu"
                else:
                    name = self.data[(self.data['label'] == p[0])].name.iloc[0]
                names.append(name)
                title = title + name + " "
            try:
                data = fd_tiny.SetName(frame, refined_bboxes, names)
                image_copy = data[0]
                temp_name = data[1]
                image_copy = imutils.resize(image_copy, width=720)
                cv2.destroyAllWindows()
            except:
                return None
            return [image_copy, temp_name]
        except:
            return None
        finally:
            pass
