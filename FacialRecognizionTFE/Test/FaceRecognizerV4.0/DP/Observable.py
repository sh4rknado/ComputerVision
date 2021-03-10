#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jordan BERTIEAUX"
__copyright__ = "Copyright 2019, Projet FastBerry"
__credits__ = ["Jordan BERTIEAUX"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Jordan BERTIEAUX"
__email__ = "jordan.bertieaux@std.heh.be"
__status__ = "Production"


# ===========================================================================
#         Définition de la classe Osbervable
# ===========================================================================
class Observable:
    def __init__(self):
        self._osbservers = []

    def register(self, observer):
        if observer not in self._osbservers:
            self._osbservers.append(observer)

    def unregister(self, observer):
        self._osbservers.remove(observer)

    def unregister_all(self):
        if self._osbservers:
            del self._osbservers[:]

    def update_observer(self, result, image):
        for observer in self._osbservers:
            observer.update(result, image)
