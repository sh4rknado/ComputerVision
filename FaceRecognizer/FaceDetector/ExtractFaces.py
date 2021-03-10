from Helper import Colors, PATH, Serializer
from imageio import imread, imsave
import dlib
import glob
import pandas as pd
from tqdm import tqdm
import os


class ExtractFaces:

    def __init__(self):
        self._color = Colors.Colors()
        self._path = PATH.PATH()
        self._serializer = Serializer.Serializer()

    def run(self):
        # grab the paths to the input images in our dataset
        self._color.printing("info", "[LOADING] Quantifying faces...")

        # Get list of Folder
        train_paths = glob.glob("IMAGE_DB_RAW/*")
        # print(train_paths)

        data = self._formated_data(train_paths)

        self._color.printing("success", "[SUCCESS] Quantifying faces Finished\n")

        # extract Facial
        self._extract_facial(data)

    def _formated_data(self, train_paths):
        # Create the Formated Data
        data = pd.DataFrame(columns=['image', 'label', 'name'])

        for i, train_path in tqdm(enumerate(train_paths)):
            name = train_path.split("/")[-1]
            images = glob.glob(train_path + "/*")
            for image in images:
                data.loc[len(data)] = [image, i, name]

        # print(data)
        return data

    def _extract_facial(self, data):
        # initialize the total number of faces processed
        total = 0
        faces = []

        for img_path in data.image:
            self._color.printing("info", "[PROCESSING] processing image {}/{}".format(total + 1, len(data.image)))
            # print(img_path)

            image = imread(img_path)
            hogFaceDetector = dlib.get_frontal_face_detector()
            faceRects = hogFaceDetector(image, 0)

            try:
                faceRect = faceRects[0]
                if faceRect == None:
                    continue
            except:
                self._color.printing("error", "[ERROR] NO FACES DETECTED : " + img_path +
                                     " {}/{}".format(total + 1, len(data.image)))
                os.system("rm -rf " + img_path)
                total += 1
                continue
            finally:
                pass

            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()

            face = image[y1:y2, x1:x2]
            faces.append(face)
            # print(face)

            imsave(img_path, face)
            total += 1

        os.system("rsync -a " + self._path.IMAGE_DB_RAW + "/*  " + self._path.IMAGE_DB)
        os.system("rm -rf " + self._path.IMAGE_DB_RAW + "/*")

        # Get list of Folder
        train_paths = glob.glob("Data/IMAGE_DB/*")
        # print(train_paths)
        data = self._formated_data(train_paths)

        self._color.printing("success", "[SUCCESS] Extraction Completed\n")
        self._serializer.saving_data(data)
        self._serializer.saving_faces(faces)
