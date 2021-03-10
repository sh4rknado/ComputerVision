from Recognizer.Recognizer import Recognizer
from FaceDetector.ExtractFaces import ExtractFaces
from Helper.PATH import PATH
from imutils import paths
from Helper.Colors import Colors
from Recognizer.Recognizer import Recognizer


def check_new_files():
    path = PATH()
    # print(path.IMAGE_DB_RAW)
    if len(list(paths.list_images(path.IMAGE_DB_RAW))) > 0:
        del path
        return True
    else:
        del path
        return False


def check_file_to_detect():
    path = PATH()
    # print(path.IMAGE_DB_RAW)

    if len(list(paths.list_images(path.IMAGE_TO_DETECT))) > 0:
        del path
        return True
    else:
        del path
        return False


if __name__ == "__main__":
    color = Colors()
    if check_new_files():
        color.printing("info", "[NEW] New Image Detected Run Analyse...\n")
        fd = ExtractFaces()
        fd.run()
        del fd
    if check_file_to_detect():
        color.printing("info", "[NEW] New Image To Detect Run Recognizing...\n")
        reco = Recognizer()
        reco.run()
