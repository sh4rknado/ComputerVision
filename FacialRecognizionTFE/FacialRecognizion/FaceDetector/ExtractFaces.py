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
from Helper.Colors import Colors
from Helper.Serializer import Serializer
import glob
import cv2
import time
import os
import re


# ===========================================================================
#         Definition of class ExtractFaces
# ===========================================================================
class ExtractFaces:

    # *=================================*
    # |     Function of main            |
    # *=================================*
    def run(self, fd, obj, pickle_data):
        # Get list of files in Folder
        data = Serializer.format_data(glob.glob("IMAGE_DB_RAW/*"))

        # *======================*
        # | Create Database Tree |
        # *======================*
        Colors.print_infos("[INFO] Create Folders of databases...\n")
        for name in data.name:
            if not os.path.isdir("Data/IMAGE_DB/" + name):
                os.mkdir("Data/IMAGE_DB/" + name)

        # *=======================*
        # | Extract Faces Process |
        # *=======================*
        cpt = 0
        t2 = time.time()
        for img_path in data.image:
            Colors.print_infos("\n[PROCESSING] Try to Detect a Person...")

            # *===================*
            # | Performed Process |
            # *===================*
            yolo_result = obj.run(img_path)

            if re.match('person', yolo_result):
                Colors.print_sucess("[PROCESSING] Person Detected !")
                Colors.print_infos("[PROCESSING] Running Extract Faces Processing...")

                # print(img_path)
                Colors.print_infos("[PROCESSING] Extract Faces {}/{}".format(cpt + 1, len(data.image)))
                t1 = time.time()
                result = fd.ExtractFace(cv2.imread(img_path), "Data/IMAGE_DB/" + str(data.name[cpt]) + "/result_" + str(cpt))
                Colors.print_infos("[PROCESSING] Faces Detected : {} in {} s".format(result, time.time() - t1))
                del t1
                del result
            else:
                Colors.print_error("[PROCESSING] No Face Detected !")
            cpt += 1

        Colors.print_infos("[INFO] Remove file in IMG_DB_RAW...")
        os.system("rm -rfv IMAGE_DB_RAW/*")

        Colors.print_sucess("\n[SUCCESS] Extraction Completed in " + str(round(time.time()-t2, 4)) + " s\n")

        # Cleanning RAM
        del data
        del fd
        del cpt
        del t2

        data = Serializer.format_data(glob.glob("Data/IMAGE_DB/*"))

        # Saving Data
        Serializer.saving_data(data, pickle_data)
