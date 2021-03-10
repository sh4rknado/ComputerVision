import pickle
from Helper import PATH, Colors


class Serializer:

    def __init__(self):
        self._color = Colors.Colors()
        self._path = PATH.PATH()

    """
    @:parameter faces = faces[]
    """
    def saving_faces(self, faces):
        self._color.printing("info", "\n[SAVING] Serializing Faces...")
        f = open(self._path.PICKLE_FACES, "wb")
        f.write(pickle.dumps(faces))
        f.close()
        self._color.printing("success", "[SUCCESS] Serializing Faces Completed...\n")

    """
    @:return faces[]
    """
    def loading_faces(self):
        self._color.printing("info", "[LOADING] Loading Faces Serialised...")
        faces = []

        # Load the serialised Data
        data = pickle.loads(open(self._path.PICKLE_FACES, "rb").read())
        for d in data:
            faces.append(d)

        self._color.printing("success", "[LOADING] Loading Faces Completed\n")
        return faces

    """
    Serialise the data of Model (Label and Embeddings)
    @:data preformated by panda
    """
    def saving_data(self, data):
        self._color.printing("info", "\n[SAVING] Serializing Preformated Data...")
        # Serialize the model
        f = open(self._path.PICKLE_DATA, "wb")
        f.write(pickle.dumps(data))
        f.close()

        self._color.printing("success", "[SUCCESS] Serializing Completed\n")

    """
    @:return penda data[]
    """
    def loading_data(self):
        self._color.printing("info", "[LOADING] Loading Data Serialised...")
        # Load the serialised Data
        data = pickle.loads(open(self._path.PICKLE_DATA, "rb").read())
        self._color.printing("success", "[LOADING] Loading Data Completed\n")
        return data
