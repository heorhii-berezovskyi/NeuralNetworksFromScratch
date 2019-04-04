class ParamAlreadyExistsException(Exception):
    def __init__(self, msg: str):
        self.msg = msg


class ParamNotFoundException(Exception):
    def __init__(self, msg: str):
        self.msg = msg
