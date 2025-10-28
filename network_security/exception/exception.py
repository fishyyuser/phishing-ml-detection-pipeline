import sys
from network_security.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message

        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = None
            self.file_name = None

        logger.error(self.__str__())

    def __str__(self):
        if self.lineno and self.file_name:
            return (
                f"Error occurred in Python script [{self.file_name}] "
                f"at line [{self.lineno}] "
                f"with message: {self.error_message}"
            )
        else:
            return str(self.error_message)