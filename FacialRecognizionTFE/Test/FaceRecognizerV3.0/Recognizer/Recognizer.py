from tqdm import tqdm
from Helper import Colors, PATH, Serializer
from Recognizer.Model import create_model
from Recognizer.align import AlignDlib
from scipy.spatial import distance
from imutils import paths
from FaceDetector.FaceDetectorDNN import FaceDetectorDNN
from FaceDetector.FaceDetectorHoG import FaceDetectorHoG
from imageio import imsave
from threading import Thread

import os
import numpy as np
import cv2
import glob
import imutils
import dlib
import matplotlib.pyplot as plt


class Recognizer:
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

    # *=================================================================*
    # |                 RUN THE FACIAL RECOGNIZION                      |
    # *=================================================================*
    def run(self):
        data = self._trainning()
        self._analysing(data[0])
        self._recognize(data[0])

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

    def _trainning(self):
        for i in tqdm(range(len(self._train_paths))):
            self._label_index.append(np.asarray(self._data[self._data.label == i].index))

        train_embs = self._calc_embs(self._data.image)
        np.save(self._path.PICKLE_EMBS, train_embs)
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
        # plt.show()

    # =========================================
    # Saving and get the old state
    # =========================================
    def _Saving(self, total):
        # *===================================*
        # |   Saving the pre processing File  |
        # *===================================*
        f = open("processing.dat", "w")
        f.write(str(total))
        f.close()

    def _Reading(self):
        totalSkip = 0
        # *=====================================*
        # | if File exist try to read the file  |
        # | and get the last processus          |
        # *=====================================*
        if os.path.isfile("processing.dat"):
            f = open("processing.dat", "r")
            totalSkip = int(f.read())
            f.close()
            del f
        return totalSkip

    # =========================================
    # Analysing the Match / Unmatch Distance
    # =========================================
    def _recognize(self, train_embs):
        threshold = 0.8

        # print("Total Image : {0}".format(len(images)))
        total = 0
        totalSkip = self._Reading()
        images = list(paths.list_images(self._path.IMAGE_TO_DETECT))

        for imagepath in images:

            # *==========================*
            # | skipping image the list  |
            # | and get the last session |
            # *==========================*
            if total < totalSkip:
                total += 1
                continue

            image = cv2.imread(imagepath)
            image_copy = image.copy()
            cv2.resize(image_copy, (1920, 1080), interpolation=cv2.INTER_LINEAR)

            try:
                fd_dnn = FaceDetectorDNN(image_copy, imagepath)

                faces = fd_dnn.detect_face(image_copy)

                self._color.printing("info", "Processing Images : {0}/{1} with Faces Detected : {2}".format(total, len(images), len(faces)))
                # self._color.printing("info", "Image Path : " + imagepath)

                if len(faces) == 0:
                    # print("no face detected!")
                    total += 1
                    self._Saving(total)
                    continue
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

                result = fd_dnn.detect_face_name(image_copy, names)
                temp_name = result[1]
                image_copy = result[0]
                image_copy = imutils.resize(image_copy, width=720)

                total += 1
                # print(self._path.IMAGE_DB_RESULT + "/" + temp_name + "{0}".format(total) + ".jpg")
                imsave(self._path.IMAGE_DB_RESULT + "/" + temp_name + "{0}".format(total) + ".jpg", image_copy)

                cv2.imshow("TEST", image_copy)
                cv2.waitKey(0)

                # cv2.imshow("result", image_copy)
                # cv2.waitKey(0)
                cv2.destroyAllWindows()
                self._Saving(total)

            except:
                fd_HoG = FaceDetectorHoG(image_copy, imagepath)

                faces = fd_HoG.detect_face(image_copy)

                self._color.printing("info", "Processing Images : {0}/{1} with Faces Detected : {2}".format(total, len(images), len(faces)))
                # self._color.printing("info", "Image Path : " + imagepath)

                if len(faces) == 0:
                    # print("no face detected!")
                    total += 1
                    self._Saving(total)
                    continue
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

                data = fd_HoG.detect_face_name(image_copy, names)
                image_copy = data[0]
                temp_name = data[1]
                image_copy = imutils.resize(image_copy, width=720)

                total += 1
                # print(self._path.IMAGE_DB_RESULT + "/" + temp_name + "{0}".format(total) + ".jpg")
                imsave(self._path.IMAGE_DB_RESULT + "/" + temp_name + "{0}".format(total) + ".jpg", image_copy)

                cv2.imshow("TEST", image_copy)
                cv2.waitKey(0)

                # cv2.imshow("result", image_copy)
                # cv2.waitKey(0)
                cv2.destroyAllWindows()
                self._Saving(total)

            finally:
                if total >= len(images):
                    os.system("rm -v processing.dat")
                pass
