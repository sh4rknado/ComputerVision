from tqdm import tqdm
from Helper import Colors,PATH,Serializer
from FaceDetector.Model import create_model
from FaceDetector.align import AlignDlib
from scipy.spatial import distance
from imageio import imsave
from imutils import paths
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
        self._train_paths = glob.glob("Data/IMAGE_DB/*")
        self._nb_classes = len(self._train_paths)
        self._label_index = []
        # print(type(self._faces))
        # print(self._data)
        # print(self._faces)

    def run(self):
        self._loading_model()
        data = self._trainning()
        self._analysing(data[0])
        self._recognize(data[0])

    """
    I use the 68 Face Landmarks predicator with alignement of Face 
    I based on openface modified Model
    List of Model : 
        OPENFACE_NN4_SMALL2_V1_H5
        OPENFACE_NN4_SMALL2_V1_T7
    """
    def _loading_model(self):
        self._color.printing("info", "[LOADING] Loading the model")

        self._color.printing("info", "[LOADING] Create the model\n")
        self._nn4_small2 = create_model()
        self._nn4_small2.summary()

        self._color.printing("info", "[LOADING] Load the model size of openface")
        self._nn4_small2.load_weights(self._path.OPENFACE_NN4_SMALL2_V1_H5)

        self._color.printing("info", "[LOADING] Align the face Predicator 68 Face Landmarks")
        self._alignment = AlignDlib(self._path.SHAPE_PREDICATOR_68_FACE_LANDMARKS)
        self._color.printing("success", "[LOADING] Loading Model Completed\n")

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
        plt.show()

    def _recognize(self, train_embs):
        threshold = 1
        images = list(paths.list_images(self._path.IMAGE_TO_DETECT))

        # print(images)
        total = 0

        for image in images:

            image = cv2.imread(image)
            image_copy = image.copy()

            hogFaceDetector = dlib.get_frontal_face_detector()
            faceRects = hogFaceDetector(image_copy, 0)

            faces = []

            for faceRect in faceRects:
                x1 = faceRect.left()
                y1 = faceRect.top()
                x2 = faceRect.right()
                y2 = faceRect.bottom()
                face = image_copy[y1:y2, x1:x2]

                faces.append(face)

            self._color.printing("info", "[PROCESSING] processing Recognizing {}/{}".format(total + 1, len(images)))
            self._color.printing("info", "[PROCESSING] Face Detected : {}\n".format(len(faces)))

            if len(faces) == 0:
                print("no face detected!")
                total += 1
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

            temp_name = ""
            for i, faceRect in enumerate(faceRects):
                x1 = faceRect.left()
                y1 = faceRect.top()
                x2 = faceRect.right()
                y2 = faceRect.bottom()
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(image_copy, names[i], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                            cv2.LINE_AA)
                temp_name = names[i]

            image_copy = imutils.resize(image_copy, width=720)

            total += 1
            # print(self._path.IMAGE_DB_RESULT + "/" + temp_name + "{0}".format(total) + ".jpg")
            imsave(self._path.IMAGE_DB_RESULT + "/" + temp_name + "{0}".format(total) + ".jpg", image_copy)

            # cv2.imshow("result", image_copy)
            # cv2.waitKey(0)
            cv2.destroyAllWindows()
