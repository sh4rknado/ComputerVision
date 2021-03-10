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
import pickle
import re
import time
import cv2
import dlib
import numpy as np
import threading
import tensorflow as tf
from Helper.SQL import SQLHelpers
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


# =========================================== < LOCKING RUNNING > ====================================================

# *==========================*
# | Check if already Running |
# *==========================*
def _check_running():
    if os.path.isfile("/tmp/zm_lock"):
        return True
    else:
        return False


# *==========================*
# | Check if already Running |
# *==========================*
def _enable_lock():
    os.system("touch /tmp/zm_lock")


# *==========================*
# | Check if already Running |
# *==========================*
def _remove_lock():
    os.system("rm -rfv /tmp/zm_lock")


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
    ap.add_argument('-t', '--threads', help='Number of Threads', type=int, default=1)
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
    threads = args['threads']

    del args
    del u

    return [imagesdb, images, pattern, config, threads]


# ====================*
# |  Getting  Config  |
# *===================*
def _getting_config(max_threads):
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
    object_detector = create_object_detector(config, max_threads)
    face_detector = create_face_detector(config)
    recognizer = Recognizer(config['Training']["data_Pickle"],
                            config['Training']['train_embs'],
                            config['FaceDetectorTiny']['Tiny_Face_detection_model'],
                            config['Model']['OPENFACE_NN4_SMALL2_V1_H5'],
                            config['Model']['PREDICATOR_68_FACE_LANDMARKS'])

    Colors.print_sucess("[SUCCESS] Object/Face detector and Recognizer Loaded !")
    return _convert_boolean(config['General']['use_facial_recognizion']), _convert_boolean(
        config['General']['use_alpr']), config['Training'][
               "data_Pickle"], object_detector, face_detector, recognizer, config


# ======================================== < Read config.ini FUNCTION > ===============================================


# ==========================================*
# | Create Object Detector From config.ini  |
# *=========================================*
def create_object_detector(config, max_thread):
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
                                   max_thread,
                                   pattern=config['object']['detect_pattern'])
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
    if len(imagesdb) > 0:
        Colors.print_sucess("[NEW] New Image Detected Run Analyse...\n")
        ex = ExtractFaces()
        ex.run(face_detector, object_detector, pickle_data)
        Colors.print_infos("[INFOS] Reloading the Serialized Data\n")
        recognizer.data = Serializer.loading_data(pickle_data)
        del ex
    else:
        Colors.print_infos("\n[INFO] No Faces to Train now\n")


def _reconignizing(face_detector, reco, imgPath, faces, refined_bbox):
    if len(faces) > 0 and faces is not None:
        Colors.print_infos("[PROCESSING] Running Facial Recognizing...\n")

        result = reco.run(faces, face_detector, cv2.imread(imgPath), refined_bbox)

        if result is not None:
            Colors.print_sucess("\n[SUCCESS] Found Person: " + str(result[1]) + " \n")
            try:
                cv2.imwrite(imgPath, result[0])
                cv2.destroyAllWindows()
                return result[1]
            except:
                pass
        else:
            return None


# *===========================*
# | Extract Faces From Images |
# *===========================*
def _extractfaces(yolo_result, face_detector):
    faces_extracted = []
    bbox_list = []
    faces_images = []
    cpt = 0

    score_final, average_image, clusters_w, clusters_h, normal_idx, clusters, session = _load_model(face_detector)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for x in yolo_result.Images:
            if "person" in yolo_result.Result[cpt]:
                Colors.print_infos("[PROCESSING] Person was detected !\n"
                                   "[PROCESSING] Running extract face process...\n")

                faces, refined_bbox = face_detector.detectFaceTiny(cv2.imread(x), score_final, average_image, clusters_w, clusters_h, normal_idx, clusters, session, sess)
                Colors.print_sucess("\n[PROCESSING] " + str(len(faces)) + " Face(s) Found(s)\n")

                faces_images.append(x)
                faces_extracted.append(faces)
                bbox_list.append(refined_bbox)

    Colors.print_sucess("[SUCESS] Extracted FINISHED")

    return [faces_extracted, bbox_list, faces_images]


# *=========================*
# | Load Model FaceDetector |
# *=========================*
def _load_model(facedetector):
    x = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])

    # Create the tiny face model which weights are loaded from a pretrained model.
    score_final = facedetector.model.tiny_face(x)

    # Load an average image and clusters(reference boxes of templates).
    with open(facedetector.model_path, "rb") as f:
        _, mat_params_dict = pickle.load(f)

    average_image = facedetector.model.get_data_by_key("average_image")
    clusters = facedetector.model.get_data_by_key("clusters")
    clusters_h = clusters[:, 3] - clusters[:, 1] + 1
    clusters_w = clusters[:, 2] - clusters[:, 0] + 1
    normal_idx = np.where(clusters[:, 4] == 1)

    return [ score_final, average_image, clusters_w, clusters_h, normal_idx, clusters, x ]


# ============================================= < MAIN FUNCTION > =====================================================


if __name__ == "__main__":
    while _check_running():
        time.sleep(1)
        Colors.print_infos("[INFOS] Waiting end !")

    # *=====================*
    # | Getting Args/Config |
    # *=====================*
    _enable_lock()
    _top()
    imagesdb, images, pattern, config_path, max_threads = _getting_args()
    use_FacialRecognizer, use_ALPR, pickle_data, object_detector, face_detector, recognizer, config = _getting_config(max_threads)

    # *==========================*
    # |  Check Training Process  |
    # *==========================*
    _training(object_detector, face_detector, pickle_data)

    t1 = time.time()
    result_pers = ''
    result_voit = ''

    # *========================*
    # | check files to Detect  |
    # | Launch Infos Extractor |
    # *========================*
    if len(images) >= 1:
        cpt = 0

        # *=============================*
        # | Running the Object Detector |
        # *=============================*
        test = time.time()
        object_detector.list_img = images
        yolo_result = object_detector.run()
        Colors.print_infos("[FINISHED] Extract Faces Finished in " + str(round(time.time() - test, 2)))

        # *===========================*
        # | Extract Faces From Images |
        # *===========================*
        test = time.time()
        with tf.Graph().as_default():
            faces, bbox, faces_image = _extractfaces(yolo_result, face_detector)
        Colors.print_infos("[FINISHED] Extract Faces Finished in " + str(round(time.time() - test, 2)))

        # *========================*
        # | Foreach Face Extracted |
        # |   Launch Recognizing   |
        # *========================*
        test = time.time()
        i = 0
        for x in faces_image:
            temp = _reconignizing(face_detector, recognizer, x, faces[i], bbox[i])
            i += 1
            if str(temp) not in str(result_pers) and temp is not None:
                result_pers += temp + " "

        Colors.print_infos("[FINISHED] Recognizing Faces Finished in " + str(round(time.time() - test, 2)))

        # *===================*
        # | Foreach CAR/TRUCK |
        # *===================*
        for x in yolo_result.Images:
            if "car" in yolo_result.Result[cpt] or "truck" in yolo_result.Result[cpt]:
                if yolo_result.Result[cpt] not in result_voit:
                    result_voit += " " + yolo_result.Result[cpt]

        # *======================================*
        # | Remove Images that nothings detected |
        # *======================================*
        cpt = 0
        # sql = SQLHelpers()
        for im1 in images:
            for im2 in yolo_result.Images:
                if im1 not in im2:
                    id = re.findall(r'\b\d+\b', im1)
                    if len(id) == 2:
                        Colors.print_error("[DELETE] Delete Images : " + im1 + " With Id : " + str(id[1]))
                        # sql.delete_frames(str(id[0]), str(id[1]))
                        os.system("rm -rfv " + im1)
                    cpt += 1

    if result_pers:
        Colors.print_sucess("Person Detected : " + result_pers)

    if result_voit:
        Colors.print_sucess("Voiture Detected : " + result_voit)

    Colors.print_sucess("\n[SUCCESS] Finished with Total processing time : " + str(round(time.time() - t1, 3)) + " s")

    _remove_lock()
    del images
    del t1
