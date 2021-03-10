# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(filename='FaceRecognizion.log', level=logging.DEBUG)

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


# *===========================================================================*
# |                    Definition of Class Colors                             |
# *===========================================================================*
class Colors:
    # def __init__(self):
    # self._HEADER = '\033[95m'
    # self._OKBLUE = '\033[94m'
    # self._OKGREEN = '\033[92m'
    # self._WARNING = '\033[93m'
    # self._FAIL = '\033[91m'
    # self._ENDC = '\033[0m'
    # self._BOLD = '\033[1m'
    # self._UNDERLINE = '\033[4m'

    @staticmethod
    def print_infos(message):
        print('\033[94m' + message + '\033[0m')
        logging.debug('\033[94m' + message + '\033[0m')

    @staticmethod
    def print_error(message):
        print('\033[91m' + message + '\033[0m')
        logging.warning('\033[94m' + message + '\033[0m')

    @staticmethod
    def print_sucess(message):
        print('\033[92m' + message + '\033[0m')
        logging.info('\033[94m' + message + '\033[0m')
