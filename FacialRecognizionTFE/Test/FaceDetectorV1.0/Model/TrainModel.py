from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from Model.PATH import PATH
from Model.Colors import Colors
import os


class TrainModel:
    def __init__(self):
        self.path = PATH()
        self.color = Colors()

    def run(self):
        if self.check_all_files():
            self.color.printing("success", "[SUCCESS] Files and Directory is checked\n")
        else:
            self.color.printing("error", "[ERROR] Files is missing or not existing !")
            exit(0)

        tab = self.Loading()
        recognizer = self.CreateSvm(tab[0], tab[1])
        self.Saving(recognizer, tab[2])

    """
    @:return [ data, labels, le ]
    """
    def Loading(self):
        # load the face embeddings
        self.color.printing("info", "[Loading] Loading face embeddings...")
        data = pickle.loads(open(self.path.PICKLE_EMBEDDED, "rb").read())

        # encode the labels
        self.color.printing("info", "[Loading] Encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        self.color.printing("success", "[Loading] Loading Completed\n")

        return [data, labels, le]

    """
    @:parameter data = tab of data
    @:parameter lables = tab of labels
    """
    def CreateSvm(self, data, labels):
        # train the model used to accept the 128-d embeddings of the face and
        # then produce the actual face recognition
        self.color.printing("info", "[TRAINING] Training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)
        self.color.printing("success", "[TRAINING] Training model Completed\n")
        return recognizer

    """
    @:parameter recognizer = recognizerFromModel
    @:parameter le = labelEncoder
    """
    def Saving(self, recognizer, le):
        self.color.printing("info", "[SAVING] Saving Serialised Data...")
        # write the actual face recognition model to disk
        f = open(self.path.PICKLE_RECOGNIZER, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # write the label encoder to disk
        f = open(self.path.PICKLE_LE, "wb")
        f.write(pickle.dumps(le))
        f.close()
        self.color.printing("success", "[SUCCESS] Saving Serialised Data completed\n")


    """ 
    @:return True or False
    """
    def check_all_files(self):
        check = True
        self.color.printing("info", "[CHECKING] Verification of files and Directory ...")

        # check if folder PICKLE_DIR Exits
        if not os.path.isdir(self.path.PICKLE_DIR):
            self.color.printing("error", "Folder Not Found : " + self.path.PICKLE_DIR)
            check = False

        # check if file PICKLE_EMBEDDED Exits
        if not os.path.isfile(self.path.PICKLE_EMBEDDED):
            self.color.printing("error", "File Not Found : " + self.path.PICKLE_EMBEDDED)
            check = False

        if len(os.listdir(self.path.IMAGE_DB)) <= 0:
            self.color.printing("error", "No Image into the DB Folder : " + self.path.IMAGE_DB )
            check = False

        return check
