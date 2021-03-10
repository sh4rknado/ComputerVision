# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# *===========================================================================*
# |                       Definition of Import                                |
# *===========================================================================*
import pickle
import pandas as pd
from Helper.Colors import Colors
from tqdm import tqdm
import glob
from os import path

# *===========================================================================*
# |                       Infos Developers                                    |
# *===========================================================================*
__author__ = "Jordan BERTIEAUX"
__copyright__ = "Copyright 2020, Facial Recognition"
__credits__ = ["Jordan BERTIEAUX"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Jordan BERTIEAUX"
__email__ = "jordan.bertieaux@std.heh.be"
__status__ = "Production"


# *========================================*
# |    Definition of Serializer Class      |
# *========================================*
class Serializer:
    """
    @:parameter faces = faces[]
    @:parameter pickle_faces = file face.pickle
    """
    @staticmethod
    def saving_faces(faces, pickle_face):
        Colors.print_infos("\n[SAVING] Serializing Faces...")
        f = open(pickle_face, "wb")
        f.write(pickle.dumps(faces))
        f.close()
        del f
        Colors.print_sucess("[SUCCESS] Serializing Faces Completed...\n")

    """
    @:parameter pickle_faces = file face.pickle
    @:return faces[]
    """
    @staticmethod
    def loading_faces(pickle_face):
        Colors.print_infos("[LOADING] Loading Faces Serialised...")
        faces = []

        # Load the serialised Data
        data = pickle.loads(open(pickle_face, "rb").read())
        for d in data:
            faces.append(d)
        del data

        Colors.print_sucess("[LOADING] Loading Faces Completed\n")
        return faces

    """
    Serialise the data of Model (Label and Embeddings)
    @:parameter data = preformated data by Panda 
    @:parameter pickle_data = file face.pickle
    """
    @staticmethod
    def saving_data(data, pickle_data):
        Colors.print_infos("\n[SAVING] Serializing Preformated Data...")
        # Serialize the model
        f = open(pickle_data, "wb")
        f.write(pickle.dumps(data))
        f.close()
        del f
        Colors.print_sucess("[SUCCESS] Serializing Completed\n")

    """
    @:parameter pickle_data = file data.pickle
    @:return Preformated Data from Pickle
    """
    @staticmethod
    def loading_data(pickle_data):
        if path.isfile(pickle_data):
            Colors.print_infos("[LOADING] Loading Data Serialised...")
            # Load the serialised Data
            data = pickle.loads(open(pickle_data, "rb").read())
            Colors.print_sucess("[LOADING] Loading Data Completed\n")
            return data
        else:
            Colors.print_error("[ERROR] File Not Found : " + str(pickle_data))
            return None

    """
    @:parameter train_path = Path from glog (UNIX LIKE)
    @:return preformated Data from Pandas
    """
    @staticmethod
    def format_data(train_paths):
        data = pd.DataFrame(columns=['image', 'label', 'name'])

        for i, train_path in tqdm(enumerate(train_paths)):
            name = train_path.split("/")[-1]
            images = glob.glob(train_path + "/*")
            for image in images:
                data.loc[len(data)] = [image, i, name]

            del name
            del images

        # print(data)
        return data

    """
    @:parameter obj = obj[]
    @:parameter pickle_obj = file static.pickle
    """
    @staticmethod
    def saving_static(obj, pickle_obj):
        Colors.print_infos("\n[SAVING] Serializing Static object...")
        f = open(pickle_obj, "wb")
        f.write(pickle.dumps(obj))
        f.close()
        del f
        Colors.print_sucess("[SUCCESS] Serializing Static object Completed...\n")


    """
    @:parameter pickle_obj = file static.pickle
    @:return obj[]
    """
    @staticmethod
    def loading_static(pickle_obj):
        Colors.print_infos("[LOADING] Loading Static object...")
        obj = []

        # Load the serialised Data
        data = pickle.loads(open(pickle_obj, "rb").read())
        for d in data:
            obj.append(d)
        del data

        Colors.print_sucess("[LOADING] Loading Static object Completed\n")
        return obj