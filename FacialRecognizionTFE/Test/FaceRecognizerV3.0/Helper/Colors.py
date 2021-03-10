class Colors:
    def __init__(self):
        self._HEADER = '\033[95m'
        self._OKBLUE = '\033[94m'
        self._OKGREEN = '\033[92m'
        self._WARNING = '\033[93m'
        self._FAIL = '\033[91m'
        self._ENDC = '\033[0m'
        self._BOLD = '\033[1m'
        self._UNDERLINE = '\033[4m'

    """
    @:parameter type = type of message
    @:parameter message = message
    """
    def printing(self, type, message):
        if type == "error":
            print(self._FAIL + message + self._ENDC)

        elif type == "info":
            print(self._OKBLUE + message + self._ENDC)

        elif type == "success":
            print(self._OKGREEN + message + self._ENDC)
