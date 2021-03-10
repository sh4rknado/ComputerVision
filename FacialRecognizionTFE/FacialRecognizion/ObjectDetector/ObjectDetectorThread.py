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
from Helper.Colors import Colors
from ObjectDetector.ObjectDetector import ObjectDetector
from DP.Observer import Observer
import pandas as pd


# ===========================================================================
#           Definition of Class
# ===========================================================================
class ObjectDetectorThread(Observer):
    def __init__(self, confidence, data_load, lock, override_zm, show_percent, threshold, pattern, max_thread=20):
        self.list_img = None
        self.list_thread = []
        self.max_thread = max_thread
        self._confidence = confidence
        self._data_load = data_load
        self._lock = lock
        self._override_zm = override_zm
        self._show_percent = show_percent
        self.threshold = threshold
        self.thread_launched = 0
        self.pattern = pattern
        self._result_temp = []
        self._image_temp = []

    # ===========================================================================
    #         Running Threading
    # ===========================================================================
    def run(self):
        for img in self.list_img:
            if self.thread_launched < self.max_thread:
                self.list_thread.append(ObjectDetector(self._confidence, self.threshold, self._data_load, self._show_percent, self._override_zm, self._lock, self, img, self.pattern))
                self.thread_launched += 1
            else:
                self.start_threads()
                self.waiting_threads()
                self.clean_thread()

        data = {'Result': self._result_temp,
                'Images': self._image_temp}

        self.cleanning()
        return pd.DataFrame(data)

    # ===========================================================================
    #         Cleanning the RAM
    # ===========================================================================
    def cleanning(self):
        del self.list_img
        del self.list_thread
        del self.max_thread
        del self._confidence
        del self._data_load
        del self._lock
        del self._override_zm
        del self._show_percent
        del self.threshold
        del self.thread_launched
        del self.pattern
        del self._result_temp
        del self._image_temp

    # ===========================================================================
    #         Definition des functions de gestions des threads
    # ===========================================================================
    def waiting_threads(self):
        for thread in self.list_thread:
            if thread.is_alive():
                while thread.is_alive():
                    Colors.print_infos("[INFOS] Waiting Thread...\n")
                    self.thread_launched -= 1
                    thread.join()

    def start_threads(self):
        cpt = 0
        for thread in self.list_thread:
            if not thread.is_alive():
                thread.start()
                cpt += 1
                Colors.print_infos("[INFOS] Starting Thread {0}/{1}...".format(cpt, self.thread_launched))

    def clean_thread(self):
        self.list_thread.clear()
        self.list_thread = []
        self.thread_launched = 0

    # ===========================================================================
    #         Definition de la fonction de mise a jour
    # ===========================================================================
    def update(self, result, image):
        self._result_temp.append(result)
        self._image_temp.append(image)
