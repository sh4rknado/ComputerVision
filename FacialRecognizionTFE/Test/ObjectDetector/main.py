# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import re

from configparser import ConfigParser
from os import path
from imutils import paths
from ObjectDetector.ObjectDetector import ObjectDetector
from ObjectDetector.ObjectDetectorOptimized import ObjectDetectorOptimized
from shutil import copyfile


# ==========================================*
# | Create Object Detector From config.ini  |
# *=========================================*
def create_object_detector():
    config = ConfigParser()
    config.read('Data/Config/detector.ini')
    config = config['object']
    NET = None

    if path.isfile(config['yolo_labels_path']):
        LABELS = open(config['yolo_labels_path']).read().strip().split("\n")
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    else:
        raise Exception("Error : LabelPath no such file or directory : {0}".format(config['yolo_labels_path']))

    if path.isfile(config['yolo_weights_path']) and path.isfile(config['yolo_config_path']):
        NET = cv2.dnn.readNetFromDarknet(config['yolo_config_path'], config['yolo_weights_path'])

    elif not path.isfile(config['yolo_weights_path']):
        raise Exception("Error : weightsPath no such file or directory : {0}".format(config['yolo_weights_path']))

    elif not path.isfile(config['yolo_config_path']):
        raise Exception("Error : config_path no such file or directory : {0}".format(config['yolo_config_path']))

    obj = ObjectDetectorOptimized(float(config['confidence']),
                                  float(config['threshold']),
                                  [LABELS, COLORS, NET],
                                  _convert_boolean(config['yolo_show_percent']),
                                  _convert_boolean(config['yolo_override_ZM']),
                                  config['detect_pattern'])
    # clean the RAM
    del config
    del LABELS
    del COLORS
    del NET

    return obj


# =============================*
# | Convert String to Boolean  |
# *============================*
def _convert_boolean(string):
    if re.match('(y|Y|Yes|yes|True|true)', string):
        return True
    else:
        return False


# *===============*
# | Check Folder  |
# *===============*
def check_img(folder):
    img = None

    if path.isdir(folder):
        img = list(paths.list_images(folder))

        if len(img) >= 1:
            i = 0

            for im in img:
                copyfile(im, "DB_RESULT/image_" + str(i) + ".jpg")
                i += 1

            img = list(paths.list_images("DB_RESULT"))

        else:
            img = None
            print("No Image Detected")
    else:
        print("No Folder Detected : " + str(folder))

    return img


def check_video(folder):
    video = None

    if path.isdir(folder):
        video = list(paths.list_files(folder))

        if len(video) < 1:
            video = None
            print("No Video Detected")
    else:
        print("No Folder Detected :" + folder)

    return video


# *===============*
# | Test Methods  |
# *===============*
def old_method(image):
    print("\n[INFOS] Old Method")
    cpt = 0
    t1 = time.time()

    for img in image:
        cpt += 1
        print("\n[INFOS] Proccessing : " + str(cpt) + " / " + str(len(image)))

        t3 = time.time()
        obj = ObjectDetector(image=img)
        obj.run()
        t4 = time.time()

        t_total = float("{0:.2f}".format(t4 - t3))
        print("[INFOS] Time Processing: " + str(t_total) + " second")

    t2 = time.time()
    t_total = float("{0:.2f}".format(t2 - t1))
    print("\n[INFOS] Proccessing Finished : " + str(t_total) + " second\n")
    return t_total


def new_method(image):
    new_detector = create_object_detector()
    print("\n[INFOS] New Method")
    cpt = 0
    t1 = time.time()

    for img in image:
        cpt += 1
        print("\n[INFOS] Proccessing : " + str(cpt) + " / " + str(len(image)))

        t3 = time.time()
        new_detector.image_path = img
        new_detector.run()
        t4 = time.time()

        t_total = float("{0:.2f}".format(t4 - t3))
        print("[INFOS] Time Processing: " + str(t_total) + " second")

    t2 = time.time()
    t_total = float("{0:.2f}".format(t2 - t1))
    print("\n[INFOS] Proccessing Finished : " + str(t_total) + " second\n\n")
    return t_total


if __name__ == "__main__":
    ap = argparse.ArgumentParser("main.py")
    ap.add_argument("-i", "--images", help="images folder", default="IMAGE_TO_DETECT")
    ap.add_argument("-v", "--videos", help="videos folder", default="VIDEO_TO_DETECT")
    args = ap.parse_args()

    # print(args.images)
    # print(args.videos)

    img = check_img(args.images)
    video = check_video(args.videos)

    # print(img)
    # print(video)

    if img is not None:
        time1 = old_method(img)
        time2 = new_method(img)

        t_total = float("{0:.2f}".format(time1 - time2))
        print("\n[INFOS] Method2 was " + str(t_total) + " second Fatest then Method1\n")

