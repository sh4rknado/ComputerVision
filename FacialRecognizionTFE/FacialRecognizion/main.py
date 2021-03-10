# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================================================
#           Infos developer
# ===========================================================================
__author__ = "Jordan BERTIEAUX"
__copyright__ = "Copyright 2020, Facial Recognition"
__credits__ = ["Jordan BERTIEAUX"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Jordan BERTIEAUX"
__email__ = "jordan.bertieaux@std.heh.be"
__status__ = "Production"


# ===========================================================================
#           Definition of Import
# ===========================================================================
import argparse
import os
import re
import time
import cv2
import dlib
import numpy as np
import threading
from configparser import ConfigParser, ExtendedInterpolation
from os import path
from imutils import paths
from FaceDetector.ExtractFaces import ExtractFaces
from FaceDetector.FaceDetectorDNN import FaceDetectorDNN
from FaceDetector.FaceDetectorHaar import FaceDetectorHaar
from FaceDetector.FaceDetectorHoG import FaceDetectorHoG
from FaceDetector.FaceDetectorMMOD import FaceDetectorMMOD
from FaceDetector.FaceDetectorTINY import FaceDetectorTiny
from Helper.Colors import Colors
from ObjectDetector.ObjectDetectorThread import ObjectDetectorThread
from Recognizer.Recognizer import Recognizer
from Helper.Serializer import Serializer

# =========================================== < HELPERS FUNCTION > ====================================================


# *============================*
# | Convert String to Boolean  |
# *============================*
def _convert_boolean(string):
    if re.match('(y|Y|Yes|yes|True|true)', string):
        return True
    else:
        return False


# =======================*
# | Affichage des infos  |
# *======================*
def _top():
    os.system("clear")
    Colors.print_infos("\n*-----------------------------------------------------*\n"
                       "| __author__ = Jordan BERTIEAUX                       |\n"
                       "| __copyright__ = Copyright 2020, Facial Recognition  |\n"
                       "| __credits__ = [Jordan BERTIEAUX]                    |\n"
                       "| __license__ = GPL                                   |\n"
                       "| __version__ = 1.0                                   |\n"
                       "| __maintainer__ = Jordan BERTIEAUX                   |\n"
                       "| __email__ = jordan.bertieaux@std.heh.be             |\n"
                       "| __status__ = Production                             |\n"
                       "*-----------------------------------------------------*\n")


# ====================*
# |   Getting Args    |
# *===================*
def _getting_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--eventpath', help='Directory of Events', default='IMAGE_TO_DETECT')
    ap.add_argument('-i', '--imgdb', help='Directory Image to Train', default='IMAGE_DB_RAW')
    ap.add_argument('-p', '--pattern', help='Override the YoloPattern by IPCAM Events', default='')
    ap.add_argument('-c', '--config', help='File config ini', default='Data/Config/detector.ini')
    args, u = ap.parse_known_args()
    args = vars(args)
    del ap

    # *=====================================*
    # |      Get All Values from Args       |
    # *=====================================*
    images = list(paths.list_images(args['eventpath']))
    imagesdb = list(paths.list_images(args['imgdb']))
    pattern = args['pattern']
    config = args['config']
    del args

    return [imagesdb, images, pattern, config]


# ====================*
# |  Getting  Config  |
# *===================*
def _getting_config():
    # *=============================*
    # |  Read the ini config file   |
    # *=============================*
    Colors.print_infos("[INFOS] Reading config detector.ini...")
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_path)

    # *=============================*
    # | Create Face/object detector |
    # *=============================*
    Colors.print_infos("[INFOS] Loading object/Face detector and Recognizer ...")
    object_detector = create_object_detector(config, max_thread=1)
    face_detector = create_face_detector(config)
    recognizer = Recognizer(config['Training']["data_Pickle"],
                            config['Training']['train_embs'],
                            config['FaceDetectorTiny']['Tiny_Face_detection_model'],
                            config['Model']['OPENFACE_NN4_SMALL2_V1_H5'],
                            config['Model']['PREDICATOR_68_FACE_LANDMARKS'])

    Colors.print_sucess("[SUCCESS] Object/Face detector and Recognizer Loaded !")
    return _convert_boolean(config['General']['use_facial_recognizion']), _convert_boolean(config['General']['use_alpr']), config['Training']["data_Pickle"], object_detector, face_detector, recognizer, config


# ======================================== < Read config.ini FUNCTION > ===============================================


# ==========================================*
# | Create Object Detector From config.ini  |
# *=========================================*
def create_object_detector(config, max_thread=1):
    obj = None
    yolo_weight = config['object']['yolo_weights_path']
    yolo_config = config['object']['yolo_config_path']
    yolo_labels = config['object']['yolo_labels_path']

    if path.isfile(yolo_labels) and path.isfile(yolo_weight) and path.isfile(yolo_config):
        LABELS = open(yolo_labels).read().strip().split("\n")
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
        NET = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weight)

        obj = ObjectDetectorThread(float(config['object']['confidence']),
                                   [LABELS, COLORS, NET], threading.Lock(),
                                   _convert_boolean(config['object']['yolo_override_ZM']),
                                   _convert_boolean(config['object']['yolo_show_percent']),
                                   float(config['object']['threshold']),
                                   pattern=config['object']['detect_pattern'], max_thread=max_thread)
        del LABELS
        del COLORS
        del NET

    elif not path.isfile(yolo_labels):
        raise Exception("Error : LabelPath no such file or directory : {0}".format(yolo_labels))
    elif not path.isfile(yolo_weight):
        raise Exception("Error : weightsPath no such file or directory : {0}".format(yolo_weight))
    elif not path.isfile(yolo_config):
        raise Exception("Error : config_path no such file or directory : {0}".format(yolo_config))

    # clean the RAM
    del yolo_weight
    del yolo_config
    del yolo_labels
    return obj


# *=======================================*
# | Create Face Detector From config.ini  |
# *=======================================*
def create_face_detector(config):
    face_detector = None
    process = config['General']['face_detector_process']

    # ELIF FACEDETECTOR == Tiny
    if process == "Tiny":
        if path.isfile(config['FaceDetectorTiny']['Tiny_Face_detection_model']):
            face_detector = FaceDetectorTiny(prob_thresh=float(config['FaceDetectorTiny']['prob_thresh']),
                                             nms_thres=float(config['FaceDetectorTiny']['nms_tresh']),
                                             lw=int(config['FaceDetectorTiny']['lw']),
                                             model=str(config['FaceDetectorTiny']['Tiny_Face_detection_model']))
        else:
            raise Exception("[ERROR] Tiny Model no such file or directory : " +
                            str(config['FaceDetectorTiny']['Tiny_Face_detection_model']))

    # IF FACEDETECTOR == DNN
    elif process == "DNN":
        modelFile = config['FaceDetectorDNN']['modelFile']
        configFile = config['FaceDetectorDNN']['configFile']

        if path.isfile(modelFile) and path.isfile(configFile):
            face_detector = FaceDetectorDNN(float(config['FaceDetectorDNN']['conf_threshold']),
                                            config['FaceDetectorDNN']['process_model'],
                                            modelFile,
                                            configFile)
        else:
            if not path.isfile(modelFile):
                raise Exception("[ERROR] No such file or Directory : {0}".format(modelFile))
            elif not path.isfile(configFile):
                raise Exception("[ERROR] No such file or Directory : {0}".format(configFile))

        del modelFile
        del configFile

    # ELIF FACEDETECTOR == HaarCascade
    elif process == "Haar":
        configFile = config['FaceDetectorHaar']['haarcascade_frontalface_default']
        if path.isfile(configFile):
            face_detector = FaceDetectorHaar(int(config['FaceDetectorHaar']['max_multiscale']),
                                             float(config['FaceDetectorHaar']['min_multiscale']),
                                             cv2.CascadeClassifier(configFile))
        else:
            raise Exception("[ERROR] HaarCasecade Model No such file or Directory : ".format(configFile))

        del configFile

    # ELIF FACEDETECTOR == MMOD
    elif process == "MMOD":
        configFile = config['FaceDetectorMMOD']['cnn_face_detection_model_v1']

        if path.isfile(configFile):
            face_detector = FaceDetectorMMOD(dlib.cnn_face_detection_model_v1(configFile))
        else:
            raise Exception("[ERROR] MMOD Model no such file or directory : ".format(configFile))

        del configFile

    # ELIF FACEDETECTOR == HoG
    elif process == "HoG":
        face_detector = FaceDetectorHoG()

    # ELSE FACEDETECTOR == ERROR
    else:
        raise Exception("ERROR No FaceDetector Found into detector.ini")

    del process
    return face_detector


# ========================================== < TRAINNING FUNCTION > ===================================================

# ===========================*
# | check files to TRAIN     |
# |        AND               |
# | LAUNCH TRAINNING PROCESS |
# *==========================*

def _training(object_detector, face_detector, pickle_data):
    Colors.print_sucess("[NEW] New Image Detected Run Analyse...\n")
    ex = ExtractFaces()
    ex.run(face_detector, object_detector, pickle_data)
    Colors.print_infos("[INFOS] Reloading the Serialized Data")
    recognizer.data = Serializer.loading_data(pickle_data)
    del ex


def _reconignizing(face_detector, reco, imgPath):
    # *==================*
    # | Extract the Face |
    # *==================*
    Colors.print_infos("[INFOS] Person Was detected !\n"
                       "[PROCESSING] Running Detect Face Process...\n")

    result = face_detector.detectFaceTiny(frame=cv2.imread(imgPath))
    faces = result[0]
    refined_bbox = result[1]
    del result

    Colors.print_sucess("\n[PROCESSING] " + str(len(faces)) + " Face Detected\n")

    if len(faces) > 0 and faces is not None:
        Colors.print_infos("[PROCESSING] Running Facial Recognizing...\n")

        result = reco.run(faces, face_detector, cv2.imread(imgPath), refined_bbox)

        if result is not None:
            Colors.print_sucess("\n[SUCESS] Detected Person: " + str(result[1]) + " \n")
            try:
                cv2.imwrite(imgPath, result[0])
                cv2.destroyAllWindows()
                return result[1]
            except:
                pass
        else:
            return None

# ============================================= < MAIN FUNCTION > =====================================================


if __name__ == "__main__":
    # *=====================*
    # | Getting Args/Config |
    # *=====================*
    _top()
    imagesdb, images, pattern, config_path = _getting_args()
    use_FacialRecognizer, use_ALPR, pickle_data, object_detector, face_detector, recognizer, config = _getting_config()

    # *==========================*
    # |  Check Training Process  |
    # *==========================*
    if len(imagesdb) > 0:
        _training(object_detector, face_detector, pickle_data)
    else:
        Colors.print_infos("\n[INFO] No Faces to Train now\n")

    t1 = time.time()
    result = ''

    # *========================*
    # | check files to Detect  |
    # |         AND            |
    # | Launch Infos Extractor |
    # *========================*
    if len(images) > 0:
        cpt = 0

        # *=============================*
        # | Running the Object Detector |
        # *=============================*
        object_detector.list_img = images
        yolo_result = object_detector.run()

        # *=================================*
        # | Foreach image in list of images |
        # *=================================*
        for x in yolo_result.Result:
            if x not in result:
                result += str(x) + " "

            # t2 = time.time()
            #
            # # *=============================*
            # # | Running the Object Detector |
            # # *=============================*
            # result = None
            #
            # if re.match('person', yolo_result) and use_FacialRecognizer:
            #     # print("Found : Person")
            #     if 'Person' not in final_val:
            #         final_val += " Person"
            #     # result = _reconignizing(face_detector, recognizer, img)
            #
            # elif re.match('car', yolo_result) and use_ALPR:
            #     # print("Found : CAR AND ALPR")
            #     if 'Car' not in final_val:
            #         final_val += " Car"
            #     # TODO ALPR RECOGNIZING
            #
            # elif yolo_result is None or yolo_result == "":
            #     # os.system("mv " + img + " /home/zerocool/PycharmProjects/FacialRecognizionTFE/Test/FaceRecognizerV4.0/IMG_DELETED")
            #     if verbose:
            #         Colors.print_error("[ERROR] Nothing Found: " + img)
            # cpt += 1
            #
            # if verbose:
            #     Colors.print_infos("\n[PROCESSING] Processing Image {0}/{1} in {2} s".format(cpt + 1, len(images), round(time.time()-t2, 3)))

    if result:
        print("Detected : " + result)

    Colors.print_sucess("\n[SUCCESS] Finished with Total processing time : " + str(round(time.time()-t1, 3)) + " s")
    del images
    del t1
