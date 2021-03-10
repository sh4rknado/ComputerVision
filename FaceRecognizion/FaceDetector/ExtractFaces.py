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
from FaceDetector.FaceDetector import FaceDetector
from Helper.PATH import PATH
from tqdm import tqdm
from DPObserver.Subject import Subject

import glob
import pandas as pd
import cv2
import time
import os


# ===========================================================================
#         Definition of class ExtractFaces
# ===========================================================================
class ExtractFaces(Subject):
    def __init__(self):
        Subject.__init__(self)
        self._color = Colors()
        self._path = PATH()
        self._serializer = Serializer()
        self._facesThreads = []
        self._faces = []
        self._imgfaces = []
        self._running = 0
        self._total = 0

    # ===========================================================================
    #         Function of main
    # ===========================================================================
    def run(self):
        self._color.printing("info", "[LOADING] Quantifying faces...")

        # Get list of Folder
        train_paths = glob.glob("IMAGE_DB_RAW/*")
        # print(train_paths)

        data = self._format_data(train_paths)
        self._thread_init(data)
        self._color.printing("success", "[SUCCESS] Quantifying faces Finished\n")

        self._launch_detect_face()
        self._waiting_end_thread()

        # Saving Images
        self._saving()

    # ===========================================================================
    #         Create the Data Frame with Panda
    # ===========================================================================
    """
    @:parameter train_path = Path from glog (UNIX LIKE)
    """
    def _format_data(self, train_paths):
        data = pd.DataFrame(columns=['image', 'label', 'name'])

        for i, train_path in tqdm(enumerate(train_paths)):
            name = train_path.split("/")[-1]
            images = glob.glob(train_path + "/*")
            for image in images:
                data.loc[len(data)] = [image, i, name]

        # print(data)
        return data

    # ===========================================================================
    #         Get the Notify from DP Observer
    # ===========================================================================
    """
    @:update
    """
    def update(self, value, message):
        self._faces.append(value)
        self._imgfaces.append(message)
        self._running -= 1

    # ===========================================================================
    #         Initialize the list of threads
    # ===========================================================================
    """
    @:parameter data = DataFrame
    """
    def _thread_init(self, data):
        total = 0

        for img_path in data.image:
            # self._color.printing("info", "[LOADING] Create Threading {}/{}".format(total + 1, len(data.image)))
            # print(img_path)

            # Create the Thread
            frame = cv2.imread(img_path)
            self._facesThreads.append(FaceDetector(frame, img_path, self))
            total += 1

        self._color.printing("success", "[SUCCESS] Create Threading Completed\n")

    def _waiting_end_thread(self):
        while self._running > 0:
            self._color.printing("info", "[WAITING] Waiting the end of Threads...")
            time.sleep(0.5)
        self._color.printing("success", "[SUCCESS] Thread Finished !\n")

    # ===========================================================================
    #         Launch the Threads
    # ===========================================================================
    """
    @:parameter data = the DataFrame from Panda
    @:parameter max = The maximum of Threads
    """
    def _launch_detect_face(self, max=15):

        while self._total < len(self._facesThreads):
            if self._running <= max:
                self._facesThreads[self._total].start()
                self._running += 1
                self._total += 1
                self._color.printing("info", "[PROCESSING] Processing image {}/{}".format(self._total, len(self._facesThreads)))
            else:
                while self._running == 5:
                    time.sleep(0.1)

        self._color.printing("success", "[SUCCESS] Processing image completed\n")

    def _saving(self):
        os.system("rsync -a " + self._path.IMAGE_DB_RAW + "/*  " + self._path.IMAGE_DB)
        os.system("rm -rf " + self._path.IMAGE_DB_RAW + "/*")

        # Get list of Folder
        train_paths = glob.glob("Data/IMAGE_DB/*")
        data = self._format_data(train_paths)

        self._color.printing("success", "[SUCCESS] Extraction Completed\n")
        # print(data)
        # print(self._faces)
        self._serializer.saving_data(data)
        self._serializer.saving_faces(self._faces)
