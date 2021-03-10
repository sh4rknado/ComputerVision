from FaceDetector import ExtractFaces, Recognizer
from imutils import paths
from Helper.PATH import PATH
from Helper.Colors import Colors


def check_new_files():
    path = PATH()
    print(path.IMAGE_DB_RAW)
    imagePath = list(paths.list_images(path.IMAGE_DB_RAW))
    if len(imagePath) > 0:
        return True
    else:
        return False


def check_file_to_detect():
    path = PATH()
    # print(path.IMAGE_DB_RAW)
    imagePath = list(paths.list_images(path.IMAGE_TO_DETECT))
    # print(imagePath)
    if len(imagePath) > 0:
        return True
    else:
        return False


if __name__ == "__main__":
    color = Colors()
    if check_new_files():
        color.printing("info", "[NEW] New Image Detected Run Analyse...\n")
        fd = ExtractFaces.ExtractFaces()
        fd.run()
    if check_file_to_detect():
        color.printing("info", "[NEW] New Image To Detect Run Recognizing...\n")
        reco = Recognizer.Recognizer()
        reco.run()
