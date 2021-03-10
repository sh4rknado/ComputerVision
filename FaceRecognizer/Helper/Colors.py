class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    """
    @:parameter type = type of message
    @:parameter message = message
    """
    def printing(self, type, message):
        if type == "error":
            print(Colors.FAIL + message + Colors.ENDC)

        elif type == "info":
            print(Colors.OKBLUE + message + Colors.ENDC)

        elif type == "success":
            print(Colors.OKGREEN + message + Colors.ENDC)
