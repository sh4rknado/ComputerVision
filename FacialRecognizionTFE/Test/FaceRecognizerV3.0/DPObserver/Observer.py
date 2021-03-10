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
#         Definition of Class Observer
# ===========================================================================
class Observer:
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

    def update_observer(self, value, message):
        for observer in self._osbservers:
            observer.update(value, message)
